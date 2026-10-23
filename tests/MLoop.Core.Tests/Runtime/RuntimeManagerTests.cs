using MLoop.Core.Runtime;

namespace MLoop.Core.Tests.Runtime;

/// <summary>
/// Tests for <see cref="RuntimeManager.EnsureRuntimeForTask"/> — the shared guard that loads a
/// task's native runtime before any model load deserializes native parameters. The guard
/// must be a no-op for tabular tasks (so it can be called unconditionally on every load path) and
/// must fail loudly with an install hint for an uninstalled DL runtime.
/// </summary>
public class RuntimeManagerTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void EnsureRuntimeForTask_NullOrBlank_IsNoOp(string? taskType)
    {
        // Must never throw — inference paths call this even when the task is unknown/unset.
        RuntimeManager.EnsureRuntimeForTask(taskType!);
    }

    [Theory]
    [InlineData("regression")]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    [InlineData("anomaly-detection")]
    [InlineData("clustering")]
    [InlineData("forecasting")]
    [InlineData("not-a-real-task")]
    public void EnsureRuntimeForTask_TaskWithoutNativeRuntime_IsNoOp(string taskType)
    {
        // Tabular/unknown tasks require no on-demand runtime, so the guard must do nothing —
        // this is what makes it safe to call unconditionally before every Model.Load.
        RuntimeManager.EnsureRuntimeForTask(taskType);
    }

    [Theory]
    [InlineData("image-classification")]
    [InlineData("object-detection")]
    public void EnsureRuntimeForTask_DlTask_LoadsOrThrowsInstallHint(string taskType)
    {
        var runtime = RuntimeRegistry.GetRequiredByTask(taskType);
        Assert.NotNull(runtime);

        if (!runtime!.NuGetPackages.ContainsKey(PlatformDetector.GetRuntimeIdentifier()))
        {
            // No build for this platform: the guard says so and names where it runs — not an install that cannot work.
            var unavailable = Assert.Throws<InvalidOperationException>(() => RuntimeManager.EnsureRuntimeForTask(taskType));
            Assert.Contains("no build for", unavailable.Message);
            Assert.DoesNotContain("mloop runtime install", unavailable.Message);
        }
        else if (new RuntimeManager().IsInstalled(runtime))
        {
            // Installed: the guard registers the native search paths and returns without throwing.
            RuntimeManager.EnsureRuntimeForTask(taskType);
        }
        else
        {
            // Not installed: the guard must fail loudly with an actionable install hint rather than
            // letting the load reach a cryptic DllNotFoundException at deserialization time.
            var ex = Assert.Throws<InvalidOperationException>(
                () => RuntimeManager.EnsureRuntimeForTask(taskType));

            Assert.Contains("runtime", ex.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains($"mloop runtime install {runtime!.Id}", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
    }

    // A platform is supported by a runtime only as a whole: a package to download, the libraries to find in it and
    // where. An entry with one of the three and not the others was a platform «supported» that could not install.
    [Fact]
    public void EveryRuntime_NamesThePackageLibrariesAndPathForTheSamePlatforms()
    {
        Assert.All(RuntimeRegistry.All, runtime =>
        {
            var platforms = runtime.NuGetPackages.Keys.Order().ToArray();
            Assert.Equal(platforms, runtime.NativeLibraries.Keys.Order().ToArray());
            Assert.Equal(platforms, runtime.NativeSearchPaths.Keys.Order().ToArray());
        });
    }

    // TensorFlow's redistributable carries no macOS library (its osx-x64 folder is an empty placeholder): installing it
    // there downloaded 182 MB, unpacked every other platform's files beside nothing usable and failed with a stack trace.
    [Fact]
    public void TensorFlow_IsNotOffered_WhereItsPackageHasNoLibrary()
    {
        Assert.DoesNotContain("osx-arm64", RuntimeRegistry.TensorFlow.NuGetPackages.Keys);
        Assert.DoesNotContain("osx-x64", RuntimeRegistry.TensorFlow.NuGetPackages.Keys);
    }

    [Fact]
    public async Task ARuntimeWithoutAPackageForThisPlatform_IsRefusedBeforeAnythingIsDownloadedOrWritten()
    {
        var cache = Path.Combine(Path.GetTempPath(), $"mloop-runtimes-{Guid.NewGuid():N}");
        var nowhere = new RuntimeDefinition
        {
            Id = "nowhere",
            DisplayName = "Nowhere",
            Description = "",
            Version = "1.0.0",
            ApproximateSizeMB = 1,
            NuGetPackages = [],
            NativeLibraries = [],
            NativeSearchPaths = [],
            RequiredByTasks = [],
        };

        await Assert.ThrowsAsync<PlatformNotSupportedException>(() => new RuntimeManager(cache).InstallAsync(nowhere));
        Assert.False(Directory.Exists(cache));
    }


    // A P/Invoke asks for a library by a bare name; the cache holds platform file names. Every spelling of one library
    // must meet at one name, or a preloaded library stays out of reach of the import that needs it.
    [Theory]
    [InlineData("libtensorflow.so", "tensorflow")]
    [InlineData("tensorflow.dll", "tensorflow")]
    [InlineData("tensorflow", "tensorflow")]
    [InlineData("libtensorflow", "tensorflow")]
    [InlineData("libtensorflow_framework.so.2", "tensorflow_framework")]
    [InlineData("libtensorflow.2.dylib", "tensorflow")]
    [InlineData("libonnxruntime.so", "onnxruntime")]
    [InlineData("libc10.dylib", "c10")]
    [InlineData("torch_cpu.dll", "torch_cpu")]
    public void ANativeLibrary_IsKnownByOneName_WhateverThePlatformCallsIt(string fileName, string name)
        => Assert.Equal(name, RuntimeManager.NativeLibraryName(fileName));

}
