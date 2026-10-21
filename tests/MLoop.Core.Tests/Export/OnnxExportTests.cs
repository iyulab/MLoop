using System.Globalization;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLoop.Core.Export;
using MLoop.Tests;

namespace MLoop.Core.Tests.Export;

/// <summary>
/// An exported graph must answer what the ML.NET model answers. Each test fits a small pipeline of
/// the shape training builds, exports it, runs the graph in ONNX Runtime on the same rows, and
/// compares the outputs value by value.
/// </summary>
public sealed class OnnxExportTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop-onnx-" + Guid.NewGuid().ToString("N"));
    private readonly MLContext _ml = new(seed: 7);

    public OnnxExportTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void Binary_NumericFeatures_GraphMatchesModel_AndDropsTheLabel()
    {
        var rng = new Random(3);
        var path = WriteCsv("x1,x2,label", Enumerable.Range(0, 200).Select(_ =>
        {
            double a = rng.NextDouble(), b = rng.NextDouble();
            return Row(a, b, a + b + (rng.NextDouble() - 0.5) * 0.4 > 1 ? "true" : "false");
        }));
        var loader = new TextLoader.Options
        {
            HasHeader = true, Separators = [','],
            Columns =
            [
                new TextLoader.Column("Features", DataKind.Single, 0, 1),
                new TextLoader.Column("label", DataKind.Boolean, 2),
            ],
        };
        var data = _ml.Data.CreateTextLoader(loader).Load(path);
        var model = _ml.Transforms.ReplaceMissingValues("Features")
            .Append(_ml.BinaryClassification.Trainers.SingleThreadSdcaLogisticRegression("label"))
            .Fit(data);

        var (result, onnx) = Export(model, data, loader, ["x1", "x2", "label"], "label");

        var input = Assert.Single(result.Inputs);
        Assert.Equal("Features", input.Name);
        Assert.Equal("float32", input.ElementType);
        Assert.Equal(2, input.Width);
        Assert.Equal(["x1", "x2"], input.SourceColumns);
        Assert.Empty(result.UnusedInputs); // the binary pipeline scores without the label
        Assert.Contains(result.Outputs, o => o.Name == "Probability.output" && o.ElementType == "float32");

        var outputs = Run(onnx, data, result);
        AssertClose(Floats(model.Transform(data), "Probability"), outputs["Probability.output"].Floats);
        Assert.Equal(Bools(model.Transform(data), "PredictedLabel"), outputs["PredictedLabel.output"].Bools);
    }

    [Fact]
    public void Multiclass_OneHotCategory_GraphMatchesModel_LabelIsDeclaredUnused()
    {
        var rng = new Random(5);
        string[] cats = ["x", "y", "z"];
        var path = WriteCsv("a,cat,label", Enumerable.Range(0, 240).Select(i =>
        {
            double a = rng.NextDouble();
            var cat = cats[i % 3];
            var label = cat == "z" ? "lo" : (a > 0.5 ? "hi" : "mid");
            return Row(a, cat, label);
        }));
        var loader = new TextLoader.Options
        {
            HasHeader = true, Separators = [','],
            Columns =
            [
                new TextLoader.Column("a", DataKind.Single, 0),
                new TextLoader.Column("cat", DataKind.String, 1),
                new TextLoader.Column("label", DataKind.String, 2),
            ],
        };
        var data = _ml.Data.CreateTextLoader(loader).Load(path);
        var model = _ml.Transforms.Conversion.MapValueToKey("label")
            .Append(_ml.Transforms.Categorical.OneHotEncoding("cat_encoded", "cat"))
            .Append(_ml.Transforms.Concatenate("Features", "a", "cat_encoded"))
            .Append(_ml.MulticlassClassification.Trainers.SingleThreadSdcaMaximumEntropy("label"))
            .Append(_ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
            .Fit(data);

        var (result, onnx) = Export(model, data, loader, ["a", "cat", "label"], "label");

        Assert.Equal(["a", "cat"], result.Inputs.Select(i => i.Name));
        Assert.Equal("string", result.Inputs.Single(i => i.Name == "cat").ElementType);
        var unused = Assert.Single(result.UnusedInputs);
        Assert.Equal("label", unused.Name); // MapValueToKey reads it, no output depends on it

        var outputs = Run(onnx, data, result);
        var scored = model.Transform(data);
        AssertClose(FloatVectors(scored, "Score").SelectMany(v => v).ToArray(), outputs["Score.output"].Floats);
        Assert.Equal(scored.GetColumn<ReadOnlyMemory<char>>("PredictedLabel").Select(v => v.ToString()).ToArray(),
            outputs["PredictedLabel.output"].Strings);
    }

    [Fact]
    public void Regression_TreeEnsemble_GraphMatchesModel()
    {
        var rng = new Random(11);
        var path = WriteCsv("a,b,y", Enumerable.Range(0, 200).Select(_ =>
        {
            double a = rng.NextDouble(), b = rng.NextDouble();
            return Row(a, b, 3 * a - 2 * b + rng.NextDouble() * 0.1);
        }));
        var loader = new TextLoader.Options
        {
            HasHeader = true, Separators = [','],
            Columns =
            [
                new TextLoader.Column("Features", DataKind.Single, 0, 1),
                new TextLoader.Column("y", DataKind.Single, 2),
            ],
        };
        var data = _ml.Data.CreateTextLoader(loader).Load(path);
        var model = _ml.Transforms.ReplaceMissingValues("Features")
            .Append(_ml.Regression.Trainers.FastTree(labelColumnName: "y", numberOfLeaves: 4, numberOfTrees: 20, minimumExampleCountPerLeaf: 5))
            .Fit(data);

        var (result, onnx) = Export(model, data, loader, ["a", "b", "y"], "y");

        Assert.Empty(result.UnusedInputs);
        var outputs = Run(onnx, data, result);
        AssertClose(Floats(model.Transform(data), "Score"), outputs["Score.output"].Floats);
    }

    [Fact]
    public void FreeTextFeaturization_IsRefusedWithTheStringColumnsAndTheWayOut()
    {
        var path = WriteCsv("note,level,label", Enumerable.Range(0, 60).Select(i =>
            Row(i % 2 == 0 ? "disk failure on volume" : "backup completed", i % 3 == 0 ? "WARN" : "INFO", i % 2 == 0 ? "true" : "false")));
        var loader = new TextLoader.Options
        {
            HasHeader = true, Separators = [','],
            Columns =
            [
                new TextLoader.Column("note", DataKind.String, 0),
                new TextLoader.Column("level", DataKind.String, 1),
                new TextLoader.Column("label", DataKind.Boolean, 2),
            ],
        };
        var data = _ml.Data.CreateTextLoader(loader).Load(path);
        var model = _ml.Transforms.Text.FeaturizeText("note_f", "note")
            .Append(_ml.Transforms.Text.FeaturizeText("level_f", "level"))
            .Append(_ml.Transforms.Concatenate("Features", "note_f", "level_f"))
            .Append(_ml.BinaryClassification.Trainers.SingleThreadSdcaLogisticRegression("label"))
            .Fit(data);

        Assert.NotNull(OnnxExport.FindNotExportable(model));
        using var sink = new MemoryStream();
        var ex = Assert.Throws<OnnxExportException>(() =>
            OnnxExport.Write(_ml, model, data, loader, ["note", "level", "label"], "label", sink));

        Assert.Contains("FeaturizeText", ex.Message);
        Assert.Contains("'note', 'level'", ex.Message);
        Assert.Contains("type: categorical", ex.Message);
        Assert.Equal(0, sink.Length); // nothing partial is written
    }

    [Fact]
    public void Manifest_RoundTrips()
    {
        var manifest = new OnnxManifest
        {
            OnnxFile = "model.onnx",
            OnnxSha256 = "ab",
            SourceModelSha256 = "cd",
            ModelName = "default",
            ExperimentId = "exp-001",
            Task = "binary-classification",
            Inputs = [new OnnxTensorSpec("Features", "float32", 2, ["x1", "x2"])],
            Outputs = [new OnnxOutputSpec("Score.output", "Score", "float32", 1)],
            Label = new OnnxLabelInfo { Column = "label", Classes = ["no", "yes"], Decision = "Score > 0" },
        };

        var json = manifest.ToJson();
        var back = OnnxManifest.FromJson(json);

        Assert.Contains("\"sourceColumns\"", json);
        Assert.Contains("Score > 0", json); // not >
        Assert.Equal(manifest.Inputs.Single().SourceColumns, back.Inputs.Single().SourceColumns);
        Assert.Equal(["no", "yes"], back.Label.Classes);
    }

    // ---- helpers ----

    private (OnnxExportResult Result, string OnnxPath) Export(
        ITransformer model, IDataView data, TextLoader.Options loader, string[] header, string label)
    {
        var onnxPath = Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".onnx");
        using var stream = File.Create(onnxPath);
        var result = OnnxExport.Write(_ml, model, data, loader, header, label, stream);
        return (result, onnxPath);
    }

    private sealed record Output(float[] Floats, bool[] Bools, string[] Strings);

    private static Dictionary<string, Output> Run(string onnxPath, IDataView data, OnnxExportResult result)
    {
        using var session = new InferenceSession(onnxPath);
        var rows = 0;
        using (var cursor = data.GetRowCursor([]))
            while (cursor.MoveNext()) rows++;
        var feeds = result.Inputs.Concat(result.UnusedInputs).Select(spec => Feed(spec, data, rows)).ToList();
        using var outputs = session.Run(feeds);

        return outputs.ToDictionary(o => o.Name, o =>
        {
            var value = (NamedOnnxValue)o;
            return value.Value switch
            {
                Tensor<float> t => new Output(t.ToArray(), [], []),
                Tensor<bool> t => new Output([], t.ToArray(), []),
                Tensor<string> t => new Output([], [], t.ToArray()),
                _ => new Output([], [], []),
            };
        });
    }

    private static NamedOnnxValue Feed(OnnxTensorSpec spec, IDataView data, int rows)
    {
        var column = data.Schema[spec.Name];
        switch (spec.ElementType)
        {
            case "float32" when column.Type is VectorDataViewType:
            {
                var values = data.GetColumn<VBuffer<float>>(column).Select(b => b.DenseValues().ToArray()).ToList();
                var tensor = new DenseTensor<float>([rows, spec.Width]);
                for (int r = 0; r < rows; r++)
                    for (int j = 0; j < spec.Width; j++)
                        tensor[r, j] = values[r][j];
                return NamedOnnxValue.CreateFromTensor(spec.Name, tensor);
            }
            case "float32":
            {
                var values = data.GetColumn<float>(column).ToArray();
                return NamedOnnxValue.CreateFromTensor(spec.Name, new DenseTensor<float>(values, [rows, 1]));
            }
            case "bool":
            {
                var values = data.GetColumn<bool>(column).ToArray();
                return NamedOnnxValue.CreateFromTensor(spec.Name, new DenseTensor<bool>(values, [rows, 1]));
            }
            case "string":
            {
                var values = data.GetColumn<ReadOnlyMemory<char>>(column).Select(v => v.ToString()).ToArray();
                return NamedOnnxValue.CreateFromTensor(spec.Name, new DenseTensor<string>(values, [rows, 1]));
            }
            default:
                throw new NotSupportedException(spec.ElementType);
        }
    }

    private static float[] Floats(IDataView view, string column) => view.GetColumn<float>(column).ToArray();
    private static bool[] Bools(IDataView view, string column) => view.GetColumn<bool>(column).ToArray();
    private static List<float[]> FloatVectors(IDataView view, string column) =>
        view.GetColumn<VBuffer<float>>(column).Select(b => b.DenseValues().ToArray()).ToList();

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        var worst = expected.Zip(actual, (e, a) => Math.Abs(e - a)).Max();
        Assert.True(worst < 1e-5, $"max |ML.NET - ONNX| = {worst}");
    }

    private string WriteCsv(string header, IEnumerable<string> rows)
    {
        var path = Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".csv");
        File.WriteAllLines(path, rows.Prepend(header));
        return path;
    }

    private static string Row(params object[] values) =>
        string.Join(",", values.Select(v => Convert.ToString(v, CultureInfo.InvariantCulture)));
}
