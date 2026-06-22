using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Infrastructure.ML;

public class EvaluationEngineTests : IDisposable
{
    private readonly string _testDir;
    private readonly EvaluationEngine _engine;
    private readonly MLContext _mlContext;

    public EvaluationEngineTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), "mloop-eval-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testDir);
        _engine = new EvaluationEngine();
        _mlContext = new MLContext(seed: 42);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, recursive: true); } catch { }
        }
    }

    #region ConvertStringLabelToBoolean

    [Fact]
    public void ConvertStringLabelToBoolean_WithTwoUniqueValues_ConvertsToBool()
    {
        // Arrange: OK/NG labels → should map NG=false, OK=true (alphabetical)
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new StringLabelRow { Label = "OK", Feature1 = 1f },
            new StringLabelRow { Label = "NG", Feature1 = 2f },
            new StringLabelRow { Label = "OK", Feature1 = 3f },
            new StringLabelRow { Label = "NG", Feature1 = 4f },
        });

        // Act
        var result = _engine.ConvertStringLabelToBoolean(data, "Label");

        // Assert: Label column should now be Boolean type
        var labelCol = result.Schema["Label"];
        Assert.Equal(typeof(bool), labelCol.Type.RawType);

        // Verify mapping: NG (first alphabetically) → false, OK → true
        var values = ReadBoolColumn(result, "Label");
        Assert.Equal(new[] { true, false, true, false }, values);
    }

    [Fact]
    public void ConvertStringLabelToBoolean_WithMoreThanTwoValues_ReturnsOriginal()
    {
        // Arrange: 3 unique values → not binary, should return original
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new StringLabelRow { Label = "A", Feature1 = 1f },
            new StringLabelRow { Label = "B", Feature1 = 2f },
            new StringLabelRow { Label = "C", Feature1 = 3f },
        });

        // Act
        var result = _engine.ConvertStringLabelToBoolean(data, "Label");

        // Assert: Label column should still be String type (unchanged)
        var labelCol = result.Schema["Label"];
        Assert.Equal(typeof(ReadOnlyMemory<char>), labelCol.Type.RawType);
    }

    [Fact]
    public void ConvertStringLabelToBoolean_WithOneUniqueValue_ReturnsOriginal()
    {
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new StringLabelRow { Label = "OK", Feature1 = 1f },
            new StringLabelRow { Label = "OK", Feature1 = 2f },
        });

        var result = _engine.ConvertStringLabelToBoolean(data, "Label");

        var labelCol = result.Schema["Label"];
        Assert.Equal(typeof(ReadOnlyMemory<char>), labelCol.Type.RawType);
    }

    [Fact]
    public void ConvertStringLabelToBoolean_AlphabeticalOrder_FirstIsFalse()
    {
        // "Fail" < "Pass" alphabetically → Fail=false, Pass=true
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new StringLabelRow { Label = "Pass", Feature1 = 1f },
            new StringLabelRow { Label = "Fail", Feature1 = 2f },
            new StringLabelRow { Label = "Pass", Feature1 = 3f },
        });

        var result = _engine.ConvertStringLabelToBoolean(data, "Label");

        var values = ReadBoolColumn(result, "Label");
        Assert.Equal(new[] { true, false, true }, values);
    }

    [Fact]
    public void ConvertStringLabelToBoolean_CaseInsensitive_StillTwoValues()
    {
        // "ok" and "OK" are distinct strings but should still be detected as 2 values
        var data = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new StringLabelRow { Label = "ok", Feature1 = 1f },
            new StringLabelRow { Label = "OK", Feature1 = 2f },
            new StringLabelRow { Label = "NG", Feature1 = 3f },
        });

        var result = _engine.ConvertStringLabelToBoolean(data, "Label");

        // 3 unique string values (ok, OK, NG) → not binary, returns original
        var labelCol = result.Schema["Label"];
        Assert.Equal(typeof(ReadOnlyMemory<char>), labelCol.Type.RawType);
    }

    #endregion

    #region EvaluateAsync

    [Fact]
    public async Task EvaluateAsync_UnsupportedTaskType_ThrowsNotSupportedException()
    {
        var modelPath = Path.Combine(_testDir, "dummy.zip");
        var dataPath = CreateBomCsv("unsup.csv", "Feature1,Label\n1,A\n2,B\n");

        // Create a minimal model to satisfy Load
        var trainData = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new NumericRow { Feature1 = 1f, Label = 1f },
            new NumericRow { Feature1 = 2f, Label = 2f },
        });
        var pipeline = _mlContext.Transforms.Concatenate("Features", "Feature1");
        var model = pipeline.Fit(trainData);
        _mlContext.Model.Save(model, trainData.Schema, modelPath);

        var ex = await Assert.ThrowsAsync<NotSupportedException>(() =>
            _engine.EvaluateAsync(modelPath, dataPath, "Label", "unsupported-type"));

        Assert.Contains("unsupported-type", ex.Message);
    }

    [Fact]
    public async Task EvaluateAsync_Regression_ReturnsRegressionMetrics()
    {
        // Train a simple regression model
        var (modelPath, testDataPath) = TrainSimpleRegressionModel();

        var metrics = await _engine.EvaluateAsync(modelPath, testDataPath, "Label", "regression");

        Assert.Contains("r_squared", metrics.Keys);
        Assert.Contains("rmse", metrics.Keys);
        Assert.Contains("mae", metrics.Keys);
        Assert.Contains("mse", metrics.Keys);
        Assert.Equal(4, metrics.Count);
    }

    [Fact]
    public async Task EvaluateAsync_BinaryClassification_ReturnsMetrics()
    {
        var (modelPath, testDataPath) = TrainSimpleBinaryModel();

        var metrics = await _engine.EvaluateAsync(modelPath, testDataPath, "Label", "binary-classification");

        Assert.Contains("accuracy", metrics.Keys);
        Assert.Contains("auc", metrics.Keys);
        Assert.Contains("f1_score", metrics.Keys);
        Assert.Contains("precision", metrics.Keys);
        Assert.Contains("recall", metrics.Keys);
        Assert.Equal(5, metrics.Count);
    }

    [Fact]
    public async Task EvaluateAsync_MulticlassClassification_ReturnsMetrics()
    {
        var (modelPath, testDataPath) = TrainSimpleMulticlassModel();

        var metrics = await _engine.EvaluateAsync(modelPath, testDataPath, "Label", "multiclass-classification");

        Assert.Contains("macro_accuracy", metrics.Keys);
        Assert.Contains("micro_accuracy", metrics.Keys);
        Assert.Contains("log_loss", metrics.Keys);
        Assert.Equal(3, metrics.Count);
    }

    [Fact]
    public async Task EvaluateAsync_Regression_Cp949TestData_KoreanLabel_FindsLabelAndReturnsMetrics()
    {
        // BUG-43: evaluate must run the test CSV through EncodingDetector (like train/predict),
        // not force UTF-8. A CP949-encoded file with a Korean label column (e.g. '출력값') was
        // garbled into mojibake, so the label column "could not be found" even though training
        // accepted the same data. Guards both LoadTestData and the schema read against regression.
        System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
        const string korFeature = "입력값";
        const string korLabel = "출력값";

        // Train a tiny regression model whose schema uses Korean column names (UTF-8 training file).
        var trainCsv = Path.Combine(_testDir, "train_kor.csv");
        var rng = new Random(42);
        File.WriteAllText(trainCsv, BuildCsv(
            $"{korFeature},{korLabel}",
            Enumerable.Range(0, 50).Select(_ => { var f = (float)rng.NextDouble() * 10; return $"{f},{f * 2 + 1}"; })),
            new System.Text.UTF8Encoding(true));

        var loaderColumns = new[]
        {
            new TextLoader.Column(korFeature, DataKind.Single, 0),
            new TextLoader.Column(korLabel, DataKind.Single, 1),
        };
        var trainData = _mlContext.Data.CreateTextLoader(loaderColumns, hasHeader: true, separatorChar: ',').Load(trainCsv);
        var pipeline = _mlContext.Transforms.Concatenate("Features", korFeature)
            .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: korLabel, featureColumnName: "Features"));
        var model = pipeline.Fit(trainData);
        var modelPath = Path.Combine(_testDir, "regression_kor.zip");
        _mlContext.Model.Save(model, trainData.Schema, modelPath);

        // Test data written in CP949 (the pre-fix code read it as UTF-8 → garbled Korean header).
        var testCsv = Path.Combine(_testDir, "test_kor_cp949.csv");
        File.WriteAllText(testCsv, BuildCsv(
            $"{korFeature},{korLabel}",
            Enumerable.Range(0, 10).Select(_ => { var f = (float)rng.NextDouble() * 10; return $"{f},{f * 2 + 1}"; })),
            System.Text.Encoding.GetEncoding(949));

        // Act + Assert: label is found, metrics returned (pre-fix this threw "label not found").
        var metrics = await _engine.EvaluateAsync(modelPath, testCsv, korLabel, "regression");
        Assert.Contains("r_squared", metrics.Keys);
        Assert.Equal(4, metrics.Count);
    }

    #endregion

    #region Helpers

    private string CreateBomCsv(string fileName, string content)
    {
        var path = Path.Combine(_testDir, fileName);
        File.WriteAllText(path, content, new System.Text.UTF8Encoding(true));
        return path;
    }

    private (string modelPath, string testDataPath) TrainSimpleRegressionModel()
    {
        var data = new List<NumericRow>();
        var rng = new Random(42);
        for (int i = 0; i < 50; i++)
        {
            var f = (float)rng.NextDouble() * 10;
            data.Add(new NumericRow { Feature1 = f, Label = f * 2 + 1 });
        }

        var trainData = _mlContext.Data.LoadFromEnumerable(data);
        var pipeline = _mlContext.Transforms.Concatenate("Features", "Feature1")
            .Append(_mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"));

        var model = pipeline.Fit(trainData);

        var modelPath = Path.Combine(_testDir, "regression.zip");
        _mlContext.Model.Save(model, trainData.Schema, modelPath);

        // Create test data CSV
        var testPath = CreateBomCsv("regression_test.csv", BuildCsv(
            "Feature1,Label",
            data.Take(10).Select(r => $"{r.Feature1},{r.Label}")));

        return (modelPath, testPath);
    }

    private (string modelPath, string testDataPath) TrainSimpleBinaryModel()
    {
        var data = new List<BinaryRow>();
        var rng = new Random(42);
        for (int i = 0; i < 50; i++)
        {
            var f = (float)rng.NextDouble() * 10;
            data.Add(new BinaryRow { Feature1 = f, Label = f > 5 });
        }

        var trainData = _mlContext.Data.LoadFromEnumerable(data);
        var pipeline = _mlContext.Transforms.Concatenate("Features", "Feature1")
            .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "Label", featureColumnName: "Features"));

        var model = pipeline.Fit(trainData);

        var modelPath = Path.Combine(_testDir, "binary.zip");
        _mlContext.Model.Save(model, trainData.Schema, modelPath);

        var testPath = CreateBomCsv("binary_test.csv", BuildCsv(
            "Feature1,Label",
            data.Take(10).Select(r => $"{r.Feature1},{(r.Label ? "true" : "false")}")));

        return (modelPath, testPath);
    }

    private (string modelPath, string testDataPath) TrainSimpleMulticlassModel()
    {
        var data = new List<MulticlassRow>();
        var rng = new Random(42);
        string[] classes = { "A", "B", "C" };
        for (int i = 0; i < 60; i++)
        {
            var classIdx = i % 3;
            var f = (float)(classIdx * 10 + rng.NextDouble() * 3);
            data.Add(new MulticlassRow { Feature1 = f, Label = classes[classIdx] });
        }

        var trainData = _mlContext.Data.LoadFromEnumerable(data);
        var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label")
            .Append(_mlContext.Transforms.Concatenate("Features", "Feature1"))
            .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                labelColumnName: "Label", featureColumnName: "Features"))
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        var model = pipeline.Fit(trainData);

        var modelPath = Path.Combine(_testDir, "multiclass.zip");
        _mlContext.Model.Save(model, trainData.Schema, modelPath);

        var testPath = CreateBomCsv("multiclass_test.csv", BuildCsv(
            "Feature1,Label",
            data.Take(15).Select(r => $"{r.Feature1},{r.Label}")));

        return (modelPath, testPath);
    }

    private static string BuildCsv(string header, IEnumerable<string> rows)
    {
        return header + "\n" + string.Join("\n", rows) + "\n";
    }

    private bool[] ReadBoolColumn(IDataView data, string columnName)
    {
        var col = data.Schema[columnName];
        var values = new List<bool>();
        using var cursor = data.GetRowCursor(new[] { col });
        var getter = cursor.GetGetter<bool>(col);
        while (cursor.MoveNext())
        {
            bool val = default;
            getter(ref val);
            values.Add(val);
        }
        return values.ToArray();
    }

    private sealed class StringLabelRow
    {
        public string Label { get; set; } = "";
        public float Feature1 { get; set; }
    }

    private sealed class NumericRow
    {
        public float Feature1 { get; set; }
        public float Label { get; set; }
    }

    private sealed class BinaryRow
    {
        public float Feature1 { get; set; }
        public bool Label { get; set; }
    }

    private sealed class MulticlassRow
    {
        public float Feature1 { get; set; }
        public string Label { get; set; } = "";
    }

    #endregion
}
