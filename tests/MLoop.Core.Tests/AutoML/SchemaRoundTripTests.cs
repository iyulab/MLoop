using Microsoft.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// upstream-008 round-trip guards: the InputSchemaInfo that RunAsync captures
/// (CaptureInputSchema) must be directly consumable by PredictionService — the
/// natural in-process path `RunAsync → result.Schema → Predict(rows, schema, …)`.
///
/// Two historical failure modes this pins down:
/// 1. Vocabulary drift — CaptureInputSchema emitted raw .NET type names ("Single")
///    while PredictionService/CsvDataLoader/CategoricalMapper expect the semantic
///    vocabulary ("Numeric"/"Categorical"/"Text"/"Boolean"), so every numeric
///    feature fell through to DataKind.String and Transform threw.
/// 2. Merged-vector capture — InferColumns merges adjacent numeric CSV columns
///    into a ranged "Features" vector, and capturing that merged view loses the
///    named columns row-based predict needs to reconstruct.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class SchemaRoundTripTests : IDisposable
{
    private static readonly string[] SemanticDataTypes = ["Numeric", "Categorical", "Text", "Boolean"];

    private readonly string _tmpDir;

    public SchemaRoundTripTests()
    {
        _tmpDir = Path.Combine(Path.GetTempPath(), $"mloop_roundtrip_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tmpDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tmpDir))
            Directory.Delete(_tmpDir, recursive: true);
    }

    private async Task<string> WriteRegressionCsvAsync()
    {
        // Two adjacent numeric feature columns — the exact shape that makes
        // InferColumns merge them into a ranged Features vector.
        var csvPath = Path.Combine(_tmpDir, "train.csv");
        var lines = new List<string> { "age,score,response" };
        for (int i = 0; i < 150; i++)
        {
            var age = 20 + (i % 50);
            var score = i % 30;
            var response = age * 2.0 + score * 0.5;
            lines.Add($"{age},{score},{response}");
        }
        await File.WriteAllLinesAsync(csvPath, lines);
        return csvPath;
    }

    private async Task<string> WriteAnomalyCsvAsync()
    {
        var csvPath = Path.Combine(_tmpDir, "anomaly.csv");
        var lines = new List<string> { "v1,v2,label" };
        for (int i = 0; i < 150; i++)
        {
            // Mostly tight cluster with a few obvious outliers
            bool outlier = i % 30 == 0;
            var v1 = outlier ? 500 + i : 10 + (i % 5);
            var v2 = outlier ? 400 + i : 20 + (i % 7);
            lines.Add($"{v1},{v2},{(outlier ? 1 : 0)}");
        }
        await File.WriteAllLinesAsync(csvPath, lines);
        return csvPath;
    }

    private async Task<AutoMLResult> TrainAsync(MLContext ctx, string csvPath, string task, string label)
    {
        var loader = new CsvDataLoader(ctx);
        var runner = new AutoMLRunner(ctx, loader, _tmpDir);
        var config = new TrainingConfig
        {
            ModelName = "test",
            DataFile = csvPath,
            LabelColumn = label,
            Task = task,
            TimeLimitSeconds = 10
        };
        return await runner.RunAsync(config, cancellationToken: CancellationToken.None);
    }

    [Fact]
    public async Task Regression_captured_schema_uses_semantic_datatype_vocabulary()
    {
        var ctx = new MLContext(seed: 42);
        var csvPath = await WriteRegressionCsvAsync();

        var result = await TrainAsync(ctx, csvPath, "regression", "response");

        Assert.NotNull(result.Schema);
        Assert.All(result.Schema!.Columns, col =>
            Assert.True(SemanticDataTypes.Contains(col.DataType),
                $"Column '{col.Name}' has DataType '{col.DataType}' — expected one of " +
                $"[{string.Join(", ", SemanticDataTypes)}]. Raw .NET type names are not " +
                "consumable by PredictionService/CsvDataLoader."));
    }

    [Fact]
    public async Task Regression_captured_schema_exposes_named_feature_columns()
    {
        var ctx = new MLContext(seed: 42);
        var csvPath = await WriteRegressionCsvAsync();

        var result = await TrainAsync(ctx, csvPath, "regression", "response");

        Assert.NotNull(result.Schema);
        var names = result.Schema!.Columns.Select(c => c.Name).ToList();
        Assert.Contains("age", names);
        Assert.Contains("score", names);
        Assert.Contains("response", names);
    }

    [Fact]
    public async Task Regression_captured_schema_roundtrips_through_predictionservice()
    {
        var ctx = new MLContext(seed: 42);
        var csvPath = await WriteRegressionCsvAsync();

        var result = await TrainAsync(ctx, csvPath, "regression", "response");
        Assert.NotNull(result.Schema);

        var rows = new[]
        {
            new Dictionary<string, object> { ["age"] = 30, ["score"] = 10 },
            new Dictionary<string, object> { ["age"] = 45, ["score"] = 25 },
        };

        var prediction = new PredictionService(ctx)
            .Predict(rows, result.Schema!, result.Model, "regression", "response");

        Assert.Equal(2, prediction.Rows.Count);
        Assert.All(prediction.Rows, r =>
        {
            Assert.NotNull(r.Score);
            Assert.False(double.IsNaN(r.Score!.Value), "Predicted score is NaN");
        });
    }

    [Fact]
    public async Task AnomalyDetection_captured_schema_roundtrips_through_predictionservice()
    {
        var ctx = new MLContext(seed: 42);
        var csvPath = await WriteAnomalyCsvAsync();

        var result = await TrainAsync(ctx, csvPath, "anomaly-detection", "label");
        Assert.NotNull(result.Schema);

        var rows = new[]
        {
            new Dictionary<string, object> { ["v1"] = 11, ["v2"] = 22 },
            new Dictionary<string, object> { ["v1"] = 900, ["v2"] = 800 },
        };

        var prediction = new PredictionService(ctx)
            .Predict(rows, result.Schema!, result.Model, "anomaly-detection", "label");

        Assert.Equal(2, prediction.Rows.Count);
    }
}
