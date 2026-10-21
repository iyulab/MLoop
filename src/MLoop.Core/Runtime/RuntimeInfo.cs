using System.Runtime.InteropServices;

namespace MLoop.Core.Runtime;

/// <summary>
/// Defines available ML runtimes that can be downloaded on-demand
/// </summary>
public static class RuntimeRegistry
{
    public static readonly RuntimeDefinition TensorFlow = new()
    {
        Id = "tf",
        DisplayName = "TensorFlow CPU",
        Description = "TensorFlow native runtime for image classification",
        NuGetPackages = new Dictionary<string, string>
        {
            ["win-x64"] = "SciSharp.TensorFlow.Redist",
            ["linux-x64"] = "SciSharp.TensorFlow.Redist-Linux-Gpu",
            ["osx-x64"] = "SciSharp.TensorFlow.Redist",
            ["osx-arm64"] = "SciSharp.TensorFlow.Redist"
        },
        Version = "2.16.0",
        NativeLibraries = new Dictionary<string, string[]>
        {
            ["win-x64"] = ["tensorflow.dll"],
            ["linux-x64"] = ["libtensorflow.so", "libtensorflow_framework.so"],
            ["osx-x64"] = ["libtensorflow.dylib", "libtensorflow_framework.dylib"],
            ["osx-arm64"] = ["libtensorflow.dylib", "libtensorflow_framework.dylib"]
        },
        NativeSearchPaths = new Dictionary<string, string>
        {
            ["win-x64"] = "runtimes/win-x64/native",
            ["linux-x64"] = "runtimes/linux-x64/native",
            ["osx-x64"] = "runtimes/osx-x64/native",
            ["osx-arm64"] = "runtimes/osx-arm64/native"
        },
        ApproximateSizeMB = 182,
        RequiredByTasks = ["image-classification"]
    };

    public static readonly RuntimeDefinition Torch = new()
    {
        Id = "torch",
        DisplayName = "libtorch CPU",
        Description = "PyTorch native runtime for NLP and object detection",
        NuGetPackages = new Dictionary<string, string>
        {
            ["win-x64"] = "libtorch-cpu-win-x64",
            ["linux-x64"] = "libtorch-cpu-linux-x64",
            ["osx-x64"] = "libtorch-cpu-osx-x64",
            ["osx-arm64"] = "libtorch-cpu-osx-arm64"
        },
        // Must match the libtorch version Microsoft.ML.TorchSharp 0.23.0 (TorchSharp) binds to —
        // TorchSharp refuses to load a different libtorch ABI. 0.23.0 → libtorch 2.2.1.1.
        Version = "2.2.1.1",
        NativeLibraries = new Dictionary<string, string[]>
        {
            ["win-x64"] = ["torch_cpu.dll", "c10.dll", "torch.dll"],
            ["linux-x64"] = ["libtorch_cpu.so", "libc10.so", "libtorch.so"],
            ["osx-x64"] = ["libtorch_cpu.dylib", "libc10.dylib"],
            ["osx-arm64"] = ["libtorch_cpu.dylib", "libc10.dylib"]
        },
        NativeSearchPaths = new Dictionary<string, string>
        {
            ["win-x64"] = "runtimes/win-x64/native",
            ["linux-x64"] = "runtimes/linux-x64/native",
            ["osx-x64"] = "runtimes/osx-x64/native",
            ["osx-arm64"] = "runtimes/osx-arm64/native"
        },
        ApproximateSizeMB = 100,
        RequiredByTasks = ["object-detection", "text-classification", "sentence-similarity", "ner", "question-answering"]
    };

    /// <summary>
    /// ONNX Runtime, for checking an exported graph against the model it came from
    /// (<c>mloop export onnx --verify</c>). The managed half already ships with MLoop (through
    /// ML.NET AutoML); only the native library is downloaded.
    /// </summary>
    public static readonly RuntimeDefinition OnnxRuntime = new()
    {
        Id = "onnxruntime",
        DisplayName = "ONNX Runtime CPU",
        Description = "ONNX Runtime native library, to verify models exported with 'mloop export onnx'",
        NuGetPackages = new Dictionary<string, string>
        {
            ["win-x64"] = "Microsoft.ML.OnnxRuntime",
            ["win-arm64"] = "Microsoft.ML.OnnxRuntime",
            ["linux-x64"] = "Microsoft.ML.OnnxRuntime",
            ["linux-arm64"] = "Microsoft.ML.OnnxRuntime",
            ["osx-x64"] = "Microsoft.ML.OnnxRuntime",
            ["osx-arm64"] = "Microsoft.ML.OnnxRuntime",
        },
        // Must match the Microsoft.ML.OnnxRuntime.Managed that ML.NET AutoML brings in (the managed
        // half binds to this native version's API).
        Version = "1.23.2",
        NativeLibraries = new Dictionary<string, string[]>
        {
            ["win-x64"] = ["onnxruntime.dll"],
            ["win-arm64"] = ["onnxruntime.dll"],
            ["linux-x64"] = ["libonnxruntime.so"],
            ["linux-arm64"] = ["libonnxruntime.so"],
            ["osx-x64"] = ["libonnxruntime.dylib"],
            ["osx-arm64"] = ["libonnxruntime.dylib"],
        },
        NativeSearchPaths = new Dictionary<string, string>
        {
            ["win-x64"] = "runtimes/win-x64/native",
            ["win-arm64"] = "runtimes/win-arm64/native",
            ["linux-x64"] = "runtimes/linux-x64/native",
            ["linux-arm64"] = "runtimes/linux-arm64/native",
            ["osx-x64"] = "runtimes/osx-x64/native",
            ["osx-arm64"] = "runtimes/osx-arm64/native",
        },
        ApproximateSizeMB = 120,
        RequiredByTasks = []
    };

    public static IReadOnlyList<RuntimeDefinition> All { get; } = [TensorFlow, Torch, OnnxRuntime];

    public static RuntimeDefinition? GetById(string id) =>
        All.FirstOrDefault(r => r.Id.Equals(id, StringComparison.OrdinalIgnoreCase));

    public static RuntimeDefinition? GetRequiredByTask(string taskType) =>
        All.FirstOrDefault(r => r.RequiredByTasks.Contains(taskType, StringComparer.OrdinalIgnoreCase));
}

/// <summary>
/// Definition of a downloadable ML runtime
/// </summary>
public class RuntimeDefinition
{
    public required string Id { get; init; }
    public required string DisplayName { get; init; }
    public required string Description { get; init; }
    public required Dictionary<string, string> NuGetPackages { get; init; }
    public required string Version { get; init; }
    public required Dictionary<string, string[]> NativeLibraries { get; init; }
    public required Dictionary<string, string> NativeSearchPaths { get; init; }
    public required int ApproximateSizeMB { get; init; }
    public required string[] RequiredByTasks { get; init; }
}

/// <summary>
/// Detects current platform for runtime selection
/// </summary>
public static class PlatformDetector
{
    public static string GetRuntimeIdentifier()
    {
        var os = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "win"
            : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "linux"
            : RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "osx"
            : throw new PlatformNotSupportedException("Unsupported OS platform");

        var arch = RuntimeInformation.OSArchitecture switch
        {
            Architecture.X64 => "x64",
            Architecture.Arm64 => "arm64",
            _ => throw new PlatformNotSupportedException($"Unsupported architecture: {RuntimeInformation.OSArchitecture}")
        };

        return $"{os}-{arch}";
    }
}
