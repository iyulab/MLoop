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
            HasHeader = true,
            Separators = [','],
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
            HasHeader = true,
            Separators = [','],
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

    /// <summary>
    /// The graph of the trainer search's own featurization — a number and a one-hot category — answers
    /// what the model answers.
    /// </summary>
    [Fact]
    public void Binary_Category_FromTheSearchFeaturizer_GraphMatchesModel()
    {
        var rng = new Random(9);
        string[] cats = ["red", "green", "blue", "grey"];
        var path = WriteCsv("a,colour,label", Enumerable.Range(0, 240).Select(i =>
        {
            double a = rng.NextDouble();
            var colour = cats[i % 4];
            return Row(a, colour, a + (colour == "red" ? 0.4 : 0) > 0.7 ? "true" : "false");
        }));
        var loader = new TextLoader.Options
        {
            HasHeader = true,
            Separators = [','],
            Columns =
            [
                new TextLoader.Column("a", DataKind.Single, 0),
                new TextLoader.Column("colour", DataKind.String, 1),
                new TextLoader.Column("label", DataKind.Boolean, 2),
            ],
        };
        var data = _ml.Data.CreateTextLoader(loader).Load(path);
        var roles = new MLoop.Search.ColumnRoles("label",
            [new("a", MLoop.Search.ColumnRole.Numeric), new("colour", MLoop.Search.ColumnRole.Categorical)]);
        var featurizer = MLoop.Search.Candidates.Featurizer.Build(_ml, data.Schema, roles)!;
        var model = featurizer
            .Append(_ml.BinaryClassification.Trainers.LbfgsLogisticRegression("label", "Features"))
            .Fit(data);

        var (result, onnx) = Export(model, data, loader, ["a", "colour", "label"], "label");

        Assert.Equal("string", result.Inputs.Single(i => i.Name == "colour").ElementType);
        var outputs = Run(onnx, data, result);
        AssertClose(Floats(model.Transform(data), "Probability"), outputs["Probability.output"].Floats);
        Assert.Equal(Bools(model.Transform(data), "PredictedLabel"), outputs["PredictedLabel.output"].Bools);
    }

    [Fact]
    public void Verification_FeedsAVectorOfBooleans()
    {
        // The loader groups yes/no columns into one Boolean vector; the graph takes it as one input.
        var rng = new Random(17);
        var path = WriteCsv("a,f1,f2,label", Enumerable.Range(0, 160).Select(_ =>
        {
            double a = rng.NextDouble();
            bool f1 = rng.NextDouble() < 0.5, f2 = rng.NextDouble() < 0.5;
            return Row(a, f1 ? "true" : "false", f2 ? "true" : "false", a + (f1 ? 0.4 : 0) > 0.7 ? "true" : "false");
        }));
        var loader = new TextLoader.Options
        {
            HasHeader = true,
            Separators = [','],
            Columns =
            [
                new TextLoader.Column("a", DataKind.Single, 0),
                new TextLoader.Column("BooleanFeatures", DataKind.Boolean, 1, 2),
                new TextLoader.Column("label", DataKind.Boolean, 3),
            ],
        };
        var data = _ml.Data.CreateTextLoader(loader).Load(path);
        var model = _ml.Transforms.Conversion.ConvertType("flags", "BooleanFeatures", DataKind.Single)
            .Append(_ml.Transforms.Concatenate("Features", "a", "flags"))
            .Append(_ml.BinaryClassification.Trainers.SingleThreadSdcaLogisticRegression("label"))
            .Fit(data);

        var (result, onnx) = Export(model, data, loader, ["a", "f1", "f2", "label"], "label");
        var flags = Assert.Single(result.Inputs, i => i.Name == "BooleanFeatures");
        Assert.Equal(("bool", 2), (flags.ElementType, flags.Width));
        Assert.Equal(["f1", "f2"], flags.SourceColumns);

        var verification = OnnxVerification.Compare(model, data, result, onnx);

        Assert.True(verification.Passed, string.Join(", ", verification.MaxAbsDifference.Select(kv => $"{kv.Key}={kv.Value}")));
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
            HasHeader = true,
            Separators = [','],
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
            HasHeader = true,
            Separators = [','],
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
    public void PortableText_SavedAndReloaded_GraphMatchesModel()
    {
        // The exported model is always one loaded from model.zip. A single n-gram extractor over all
        // lengths exports correctly in memory but, once reloaded, as counting only the longest
        // length — the graph then disagreed with the model on every row. The portable featurizer
        // must survive the round trip.
        string[] ham = ["see you at the station at six", "ok thanks, call me later", "lunch tomorrow?"];
        string[] spam = ["WIN a FREE prize now!! call 0800 123", "URGENT: claim your cash reward", "Txt STOP to opt out of offers"];
        var rng = new Random(9);
        var path = WriteCsv("text,label", Enumerable.Range(0, 240).Select(i =>
        {
            var isSpam = i % 3 == 0;
            var pool = isSpam ? spam : ham;
            return Row($"\"{pool[rng.Next(pool.Length)]} {rng.Next(100)}\"", isSpam ? "true" : "false");
        }));
        var loader = new TextLoader.Options
        {
            HasHeader = true,
            Separators = [','],
            AllowQuoting = true,
            Columns =
            [
                new TextLoader.Column("text", DataKind.String, 0),
                new TextLoader.Column("label", DataKind.Boolean, 1),
            ],
        };
        var data = _ml.Data.CreateTextLoader(loader).Load(path);
        var trained = MLoop.Core.Training.PortableTextFeaturizer.For(_ml, "text")
            .Append(_ml.Transforms.Concatenate("Features", "text"))
            .Append(_ml.BinaryClassification.Trainers.SingleThreadSdcaLogisticRegression("label"))
            .Fit(data);
        var zip = Path.Combine(_dir, "model.zip");
        _ml.Model.Save(trained, data.Schema, zip);
        var model = _ml.Model.Load(zip, out _);

        Assert.Null(OnnxExport.FindNotExportable(model));
        var (result, onnx) = Export(model, data, loader, ["text", "label"], "label");

        var input = Assert.Single(result.Inputs);
        Assert.Equal(("text", "string"), (input.Name, input.ElementType));
        var outputs = Run(onnx, data, result);
        AssertClose(Floats(model.Transform(data), "Probability"), outputs["Probability.output"].Floats);
    }

    [Theory]
    [InlineData(true)]   // the portable featurizer: one extractor per length
    [InlineData(false)]  // one all-lengths extractor: exported wrongly once reloaded
    public void Verification_PassesAFaithfulGraph_AndFailsAWrongOne(bool portable)
    {
        string[] ham = ["see you at the station at six", "ok thanks, call me later", "lunch tomorrow?"];
        string[] spam = ["WIN a FREE prize now!! call 0800 123", "URGENT: claim your cash reward", "Txt STOP to opt out of offers"];
        var rng = new Random(13);
        var path = WriteCsv("text,label", Enumerable.Range(0, 240).Select(i =>
        {
            var isSpam = i % 3 == 0;
            var pool = isSpam ? spam : ham;
            return Row($"\"{pool[rng.Next(pool.Length)]} {rng.Next(100)}\"", isSpam ? "true" : "false");
        }));
        var loader = new TextLoader.Options
        {
            HasHeader = true,
            Separators = [','],
            AllowQuoting = true,
            Columns =
            [
                new TextLoader.Column("text", DataKind.String, 0),
                new TextLoader.Column("label", DataKind.Boolean, 1),
            ],
        };
        var data = _ml.Data.CreateTextLoader(loader).Load(path);
        IEstimator<ITransformer> featurizer = portable
            ? MLoop.Core.Training.PortableTextFeaturizer.For(_ml, "text")
            : _ml.Transforms.Text.TokenizeIntoCharactersAsKeys("chars", "text")
                .Append(_ml.Transforms.Text.ProduceNgrams("text", "chars", ngramLength: 4, useAllLengths: true,
                    weighting: Microsoft.ML.Transforms.Text.NgramExtractingEstimator.WeightingCriteria.Tf));
        var trained = featurizer
            .Append(_ml.Transforms.Concatenate("Features", "text"))
            .Append(_ml.BinaryClassification.Trainers.SingleThreadSdcaLogisticRegression("label"))
            .Fit(data);
        var zip = Path.Combine(_dir, $"model-{portable}.zip");
        _ml.Model.Save(trained, data.Schema, zip);
        var model = _ml.Model.Load(zip, out _);
        var (result, onnx) = Export(model, data, loader, ["text", "label"], "label");

        var verification = OnnxVerification.Compare(model, data, result, onnx);

        Assert.Equal(240, verification.Rows);
        Assert.Equal(portable, verification.Passed);
        if (!portable)
            Assert.True(verification.MaxAbsDifference["Probability"] > 1e-3, $"{verification.MaxAbsDifference["Probability"]}");
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
