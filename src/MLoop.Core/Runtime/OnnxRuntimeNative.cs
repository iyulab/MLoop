using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace MLoop.Core.Runtime;

/// <summary>
/// Whether ONNX Runtime can run in this process, and the native library it runs on — for every MLoop surface that
/// runs a graph (<c>mloop export onnx --verify</c>, the image encoders of <c>mloop embed</c> · <c>POST /embed</c>).
/// </summary>
/// <remarks>
/// The managed half ships with MLoop; the native library is resolved by explicit path — beside the application, or the
/// on-demand runtime (<c>mloop runtime install onnxruntime</c>) — never by the operating system's search.
/// </remarks>
public static class OnnxRuntimeNative
{
    private static int _resolverRegistered;

    /// <summary>
    /// Why ONNX Runtime cannot run here, with how to make it run, or <c>null</c> when it can.
    /// </summary>
    public static string? Unavailable()
    {
        // Decided from files, without touching ONNX Runtime: loading the wrong native library is
        // not an exception but a crash of the whole process (see NativeLibraryPath).
        if (NativeLibraryPath() is null)
        {
            var runtime = RuntimeRegistry.OnnxRuntime;
            var status = new RuntimeManager().GetStatus(runtime);
            return status.NeedsReinstall
                ? $"ONNX Runtime {status.InstalledVersion} is installed, but this mloop needs {runtime.Version}. "
                  + $"Install it again with: mloop runtime install {runtime.Id} (~{runtime.ApproximateSizeMB} MB download)"
                : $"ONNX Runtime's native library is not installed. Install it with: mloop runtime install {runtime.Id} "
                  + $"(~{runtime.ApproximateSizeMB} MB download)";
        }

        RegisterRuntimeResolver();
        try
        {
            _ = OrtEnv.Instance().GetVersionString();
            return null;
        }
        catch (Exception ex) when (IsMissingNative(ex))
        {
            return $"ONNX Runtime's native library could not be loaded ({ex.GetBaseException().Message}).";
        }
    }

    private static bool IsMissingNative(Exception ex)
    {
        for (var e = ex; e != null; e = e.InnerException)
        {
            if (e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
                return true;
        }
        return false;
    }

    /// <summary>
    /// The ONNX Runtime native library to load, by explicit path: the one the application ships
    /// beside itself (a test run, a self-contained deployment), else the on-demand runtime. Null
    /// when there is neither.
    /// </summary>
    /// <remarks>
    /// Never the operating system's search: Windows carries its own, older <c>onnxruntime.dll</c>
    /// in System32 for Windows ML, the default search finds it, and the managed half crashed the
    /// process against it with an access violation in its static initializer — not an exception
    /// anything can catch.
    /// </remarks>
    public static string? NativeLibraryPath()
    {
        var runtime = RuntimeRegistry.OnnxRuntime;
        var rid = PlatformDetector.GetRuntimeIdentifier();
        if (!runtime.NativeLibraries.TryGetValue(rid, out var libs) || libs.Length == 0)
            return null;
        var file = libs[0];

        foreach (var shipped in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, file),
                     Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", file),
                 })
        {
            if (File.Exists(shipped))
                return shipped;
        }

        var status = new RuntimeManager().GetStatus(runtime);
        var cached = Path.Combine(status.Path, file);
        return status.Installed && File.Exists(cached) ? cached : null;
    }

    /// <summary>
    /// Points the managed ONNX Runtime at <see cref="NativeLibraryPath"/>. Registered once, and only
    /// when there is a library to point at — with none, the caller stops at <see cref="Unavailable"/>.
    /// </summary>
    private static void RegisterRuntimeResolver()
    {
        if (NativeLibraryPath() is not { } libraryPath)
            return;
        if (Interlocked.Exchange(ref _resolverRegistered, 1) == 1)
            return;

        try
        {
            NativeLibrary.SetDllImportResolver(typeof(InferenceSession).Assembly, (name, _, _) =>
                name.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)
                && !name.Contains("extensions", StringComparison.OrdinalIgnoreCase)
                && NativeLibrary.TryLoad(libraryPath, out var handle)
                    ? handle
                    : IntPtr.Zero);
        }
        catch (InvalidOperationException)
        {
            // Another component already resolves this assembly's native imports; leave it to them.
        }
    }
}
