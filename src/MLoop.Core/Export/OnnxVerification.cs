using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLoop.Core.Runtime;

namespace MLoop.Core.Export;

/// <summary>How an exported graph compared with the model it came from, on the same rows.</summary>
public sealed record OnnxVerificationResult
{
    /// <summary>Rows both were run on.</summary>
    public required int Rows { get; init; }

    /// <summary>The largest absolute difference per numeric output (<c>Score</c>, <c>Probability</c>).</summary>
    public required IReadOnlyDictionary<string, double> MaxAbsDifference { get; init; }

    /// <summary>Rows whose <c>PredictedLabel</c> differs.</summary>
    public required int LabelMismatches { get; init; }

    /// <summary>The largest numeric difference that still passes.</summary>
    public required double Tolerance { get; init; }

    /// <summary>The ONNX Runtime version the graph ran on.</summary>
    public string? Runtime { get; init; }

    /// <summary>Every numeric output within <see cref="Tolerance"/> and no label differs.</summary>
    public bool Passed => LabelMismatches == 0 && MaxAbsDifference.Values.All(d => d <= Tolerance);
}

/// <summary>
/// Runs an exported graph in ONNX Runtime on data rows and compares its outputs with the ML.NET
/// model's — the check that the conversion said what the model says.
/// </summary>
/// <remarks>
/// A conversion that reports success can still be wrong: ML.NET 5.0.1's converter writes a reloaded
/// all-lengths n-gram extractor as counting only its longest length, and the graph then answered
/// differently from the model on every row while every step reported success. Only running both on
/// the same rows shows it. ONNX Runtime's native library is not part of MLoop; it is resolved from
/// the on-demand runtime (<c>mloop runtime install onnxruntime</c>) when installed, and otherwise from
/// wherever the process can already load it.
/// </remarks>
public static class OnnxVerification
{
    /// <summary>Default numeric tolerance: float32 arithmetic in a different order, not a different answer.</summary>
    public const double DefaultTolerance = 1e-4;

    /// <summary>Default number of rows compared.</summary>
    public const int DefaultRows = 1000;

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
            return $"ONNX Runtime's native library is not installed. Install it with: mloop runtime install {runtime.Id} "
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

    /// <summary>
    /// Runs <paramref name="onnxPath"/> and <paramref name="model"/> on the first
    /// <paramref name="maxRows"/> rows of <paramref name="input"/> and compares the outputs
    /// <see cref="OnnxExport.Write"/> listed in <paramref name="export"/>.
    /// </summary>
    public static OnnxVerificationResult Compare(
        ITransformer model,
        IDataView input,
        OnnxExportResult export,
        string onnxPath,
        int maxRows = DefaultRows,
        double tolerance = DefaultTolerance)
    {
        if (Unavailable() is { } unavailable)
            throw new InvalidOperationException(unavailable);

        var rows = Math.Min(maxRows, CountRows(input, maxRows));
        var expected = model.Transform(input);

        // ONNX Runtime warns, on stderr, about shape metadata it merges leniently — which ML.NET's
        // converter writes for every one-hot column. Noise to a user; errors still surface.
        using var options = new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
        using var session = new InferenceSession(onnxPath, options);
        var feeds = export.Inputs.Concat(export.UnusedInputs).Select(spec => Feed(spec, input, rows)).ToList();
        using var outputs = session.Run(feeds);
        var byName = outputs.ToDictionary(o => o.Name, o => (NamedOnnxValue)o);

        var differences = new Dictionary<string, double>();
        var labelMismatches = 0;
        foreach (var output in export.Outputs)
        {
            if (!byName.TryGetValue(output.Name, out var actual))
                throw new InvalidOperationException($"The graph did not produce '{output.Name}'.");

            switch (actual.Value)
            {
                case Tensor<float> floats when output.Column != "PredictedLabel":
                    differences[output.Column] = MaxDifference(ExpectedFloats(expected, output.Column, rows), floats.ToArray());
                    break;
                case Tensor<bool> bools:
                    labelMismatches = Math.Max(labelMismatches, Mismatches(
                        expected.GetColumn<bool>(output.Column).Take(rows).ToArray(), bools.ToArray()));
                    break;
                case Tensor<string> strings:
                    labelMismatches = Math.Max(labelMismatches, Mismatches(
                        expected.GetColumn<ReadOnlyMemory<char>>(output.Column).Take(rows).Select(v => v.ToString()).ToArray(),
                        strings.ToArray()));
                    break;
                case Tensor<uint> keys:
                    labelMismatches = Math.Max(labelMismatches, Mismatches(
                        expected.GetColumn<uint>(output.Column).Take(rows).ToArray(), keys.ToArray()));
                    break;
                case Tensor<float> floatLabels:
                    labelMismatches = Math.Max(labelMismatches, Mismatches(
                        expected.GetColumn<float>(output.Column).Take(rows).ToArray(), floatLabels.ToArray()));
                    break;
                default:
                    throw new InvalidOperationException(
                        $"Cannot compare output '{output.Name}' of type {actual.Value?.GetType().Name}.");
            }
        }

        return new OnnxVerificationResult
        {
            Rows = rows,
            MaxAbsDifference = differences,
            LabelMismatches = labelMismatches,
            Tolerance = tolerance,
            Runtime = SafeVersion(),
        };
    }

    private static NamedOnnxValue Feed(OnnxTensorSpec spec, IDataView data, int rows)
    {
        var column = data.Schema.GetColumnOrNull(spec.Name)
            ?? throw new InvalidOperationException($"The data has no column '{spec.Name}' to feed the graph.");
        int[] shape = [rows, spec.Width];
        // A vector column (a merged numeric range, the loader's group of yes/no columns) and a
        // scalar one are fed the same way: rows × width values in slot order.
        return spec.ElementType switch
        {
            "float32" => NamedOnnxValue.CreateFromTensor(spec.Name, new DenseTensor<float>(Values<float>(data, column, rows), shape)),
            "float64" => NamedOnnxValue.CreateFromTensor(spec.Name, new DenseTensor<double>(Values<double>(data, column, rows), shape)),
            "bool" => NamedOnnxValue.CreateFromTensor(spec.Name, new DenseTensor<bool>(Values<bool>(data, column, rows), shape)),
            "int32" => NamedOnnxValue.CreateFromTensor(spec.Name, new DenseTensor<int>(Values<int>(data, column, rows), shape)),
            "int64" => NamedOnnxValue.CreateFromTensor(spec.Name, new DenseTensor<long>(Values<long>(data, column, rows), shape)),
            "string" => NamedOnnxValue.CreateFromTensor(spec.Name, new DenseTensor<string>(
                Values<ReadOnlyMemory<char>>(data, column, rows).Select(v => v.ToString()).ToArray(), shape)),
            _ => throw new InvalidOperationException($"Cannot feed input '{spec.Name}' of element type {spec.ElementType}."),
        };
    }

    private static T[] Values<T>(IDataView data, DataViewSchema.Column column, int rows) =>
        column.Type is VectorDataViewType
            ? data.GetColumn<VBuffer<T>>(column).Take(rows).SelectMany(b => b.DenseValues()).ToArray()
            : data.GetColumn<T>(column).Take(rows).ToArray();

    private static float[] ExpectedFloats(IDataView scored, string column, int rows) =>
        scored.Schema[column].Type is VectorDataViewType
            ? scored.GetColumn<VBuffer<float>>(column).Take(rows).SelectMany(b => b.DenseValues()).ToArray()
            : scored.GetColumn<float>(column).Take(rows).ToArray();

    private static double MaxDifference(float[] expected, float[] actual)
    {
        if (expected.Length != actual.Length)
            return double.PositiveInfinity;
        double worst = 0;
        for (var i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i]) && float.IsNaN(actual[i]))
                continue;
            worst = Math.Max(worst, Math.Abs((double)expected[i] - actual[i]));
        }
        return worst;
    }

    private static int Mismatches<T>(T[] expected, T[] actual)
    {
        if (expected.Length != actual.Length)
            return Math.Max(expected.Length, actual.Length);
        var count = 0;
        for (var i = 0; i < expected.Length; i++)
            if (!EqualityComparer<T>.Default.Equals(expected[i], actual[i]))
                count++;
        return count;
    }

    private static int CountRows(IDataView data, int cap)
    {
        var count = 0;
        using var cursor = data.GetRowCursor([]);
        while (count < cap && cursor.MoveNext())
            count++;
        return count;
    }

    private static string? SafeVersion()
    {
        try { return "onnxruntime " + OrtEnv.Instance().GetVersionString(); }
        catch (Exception) { return null; }
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
