using System.Reflection;
using System.Xml.Linq;
using MLoop.Core.Runtime;

namespace MLoop.Core.Tests.Runtime;

/// <summary>
/// The native ONNX Runtime MLoop downloads must be the version of the managed assembly it runs
/// with, and a native library of another version on disk must not pass for an installed runtime —
/// the managed half asks the native one for its own API version and fails on anything older.
/// </summary>
public class RuntimeVersionTests
{
    [Fact]
    public void OnnxRuntimeDownload_IsTheLoadedManagedAssemblysVersion()
    {
        var informational = typeof(Microsoft.ML.OnnxRuntime.OrtEnv).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion;

        Assert.Equal(informational.Split('+')[0], RuntimeRegistry.OnnxRuntime.Version);
    }

    [Fact]
    public void OnnxRuntimeDownload_IsTheCentrallyDeclaredManagedPackage()
    {
        // The version this repository declares — compared with the authority rather than restated,
        // so a props change that the runtime does not follow is caught here.
        var props = XDocument.Load(FindUp("Directory.Packages.props"));
        var declared = props.Descendants("PackageVersion")
            .Single(e => (string?)e.Attribute("Include") == "Microsoft.ML.OnnxRuntime.Managed")
            .Attribute("Version")!.Value;

        Assert.Equal(declared, RuntimeRegistry.OnnxRuntime.Version);
    }

    [Fact]
    public void LibrariesOfAnotherVersion_AreNotAnInstalledRuntime()
    {
        using var cache = new TempCache();
        var runtime = RuntimeRegistry.OnnxRuntime;
        cache.Install(runtime, version: "1.0.0");
        var manager = new RuntimeManager(cache.Path);

        Assert.False(manager.IsInstalled(runtime));
        var status = manager.GetStatus(runtime);
        Assert.True(status.NeedsReinstall);
        Assert.Equal("1.0.0", status.InstalledVersion);
    }

    [Fact]
    public void LibrariesOfTheNeededVersion_AreAnInstalledRuntime()
    {
        using var cache = new TempCache();
        var runtime = RuntimeRegistry.OnnxRuntime;
        cache.Install(runtime, version: runtime.Version);
        var manager = new RuntimeManager(cache.Path);

        Assert.True(manager.IsInstalled(runtime));
        Assert.False(manager.GetStatus(runtime).NeedsReinstall);
    }

    [Fact]
    public void LibrariesWithoutARecordedVersion_AreTrustedAsBefore()
    {
        using var cache = new TempCache();
        var runtime = RuntimeRegistry.OnnxRuntime;
        cache.Install(runtime, version: null);
        var manager = new RuntimeManager(cache.Path);

        Assert.True(manager.IsInstalled(runtime));
        Assert.Null(manager.InstalledVersion(runtime));
    }

    private static string FindUp(string fileName)
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, fileName);
            if (File.Exists(candidate))
                return candidate;
        }
        throw new FileNotFoundException($"{fileName} not found above {AppContext.BaseDirectory}");
    }

    /// <summary>A runtime cache directory laid out as an install leaves it.</summary>
    private sealed class TempCache : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"mloop-rt-{Guid.NewGuid():N}");

        public void Install(RuntimeDefinition runtime, string? version)
        {
            var rid = PlatformDetector.GetRuntimeIdentifier();
            var dir = System.IO.Path.Combine(Path, runtime.Id, rid);
            Directory.CreateDirectory(dir);
            foreach (var lib in runtime.NativeLibraries[rid])
                File.WriteAllText(System.IO.Path.Combine(dir, lib), "");
            if (version is not null)
                File.WriteAllText(System.IO.Path.Combine(dir, "manifest.txt"), $"{runtime.Id}\n{version}\n{rid}\n2026-01-01T00:00:00Z");
        }

        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); } catch (IOException) { }
        }
    }
}
