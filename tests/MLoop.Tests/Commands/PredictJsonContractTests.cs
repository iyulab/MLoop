using Microsoft.ML;
using Microsoft.ML.AutoML;
using MLoop.CLI;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;
using MLoop.Core.Storage;
using Spectre.Console;
using System.Text.Json;

namespace MLoop.Tests.Commands;

/// <summary>
/// The CLI half of the contract in <c>docs/PREDICT-RESPONSE.md</c>. Its sibling in
/// <c>MLoop.API.Tests</c> pins the same rows over HTTP; this one exists because the document's first
/// claim is that <b>both surfaces return the same row</b>, and nothing was checking the CLI side of
/// that sentence.
/// </summary>
/// <remarks>
/// The one place the two deliberately differ is null handling — the CLI omits null fields, the server
/// writes them — so the assertions that look inverted between the two files (<c>score</c> absent here,
/// present-and-null there) are the contract, not a disagreement.
/// </remarks>
[Collection("FileSystem")]
public class PredictJsonContractTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;

    public PredictJsonContractTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-predict-json-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testProjectRoot);
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, ".mloop"));

        _fileSystem = new FileSystemManager();
        _projectDiscovery = new ProjectDiscovery(_fileSystem);
        Directory.SetCurrentDirectory(_testProjectRoot);
        _experimentStore = new ExperimentStore(_fileSystem, _projectDiscovery);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { } }
        if (Directory.Exists(_testProjectRoot))
        {
            try { Directory.Delete(_testProjectRoot, recursive: true); } catch { }
        }
    }

    /// <summary>The `--json` payload is the last line of stdout; the command narrates above it.</summary>
    private static JsonElement ParsePayload(string stdout)
    {
        var line = stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Last(l => l.TrimStart().StartsWith('{'));
        return JsonDocument.Parse(line).RootElement.Clone();
    }

    // DataType speaks the SchemaDataTypes vocabulary, not .NET type names: the predict path selects
    // the columns to concatenate into "Features" by `DataType == "Numeric"`, so a fixture saying
    // "Single" builds no feature vector and the model fails with "Could not find feature column
    // 'Features'". The constant is used rather than the literal because this fixture is not the place
    // that decides the vocabulary — it consumes it.
    private async Task SaveExperimentAsync(
        string task, string labelColumn, string trainCsv, string modelPath,
        ColumnSchema[] columns, Dictionary<string, double> metrics)
    {
        var experimentPath = _experimentStore.GetExperimentPath("default", "exp-001");
        Directory.CreateDirectory(experimentPath);
        File.Copy(modelPath, Path.Combine(experimentPath, ExperimentLayout.ModelFileName), overwrite: true);

        await _experimentStore.SaveAsync("default", new ExperimentData
        {
            ModelName = "default",
            ExperimentId = "exp-001",
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = task,
            Config = new ExperimentConfig
            {
                DataFile = trainCsv,
                LabelColumn = labelColumn,
                TimeLimitSeconds = 10,
                Metric = metrics.Keys.First(),
                TestSplit = 0.2,
                InputSchema = new InputSchemaInfo { CapturedAt = DateTime.UtcNow, Columns = [.. columns] },
            },
            Metrics = metrics,
        }, CancellationToken.None);
    }

    /// <summary>
    /// Trains a real multiclass model the way the product loads data (AutoML column inference folds the
    /// numerics into one Features vector), stores it as exp-001, and promotes it — so `predict` runs
    /// against production exactly as a user's would. Classes occur dog, cat, bird: not alphabetical, so
    /// a vocabulary that happened to be sorted cannot pass while paired with the wrong slots.
    /// </summary>
    private async Task<string> SeedMulticlassAsync()
    {
        var trainCsv = Path.Combine(_testProjectRoot, "train.csv");
        var lines = new List<string> { "X1,X2,Species" };
        foreach (var (label, cx, cy) in new[] { ("dog", 9.0, 9.0), ("cat", 5.0, 5.0), ("bird", 1.0, 1.0) })
            for (int i = 0; i < 20; i++)
            {
                double j = (i % 5 - 2) * 0.1;
                lines.Add(string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{cx + j:F4},{cy - j:F4},{label}"));
            }
        File.WriteAllLines(trainCsv, lines);

        var ml = new MLContext(seed: 42);
        var inference = ml.Auto().InferColumns(trainCsv, labelColumnName: "Species", separatorChar: ',');
        var data = ml.Data.CreateTextLoader(inference.TextLoaderOptions).Load(trainCsv);
        var pipeline = ml.Transforms.Conversion.MapValueToKey("Species", "Species")
            .Append(ml.MulticlassClassification.Trainers.SingleThreadSdcaMaximumEntropy("Species", featureColumnName: "Features"))
            .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        var model = pipeline.Fit(data);

        var modelPath = Path.Combine(_testProjectRoot, "mc.zip");
        ml.Model.Save(model, null, modelPath);

        await SaveExperimentAsync("multiclass-classification", "Species", trainCsv, modelPath,
        [
            new ColumnSchema { Name = "X1", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
            new ColumnSchema { Name = "X2", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
            new ColumnSchema { Name = "Species", DataType = SchemaDataTypes.Categorical, Purpose = "Label" },
        ], new Dictionary<string, double> { ["micro_accuracy"] = 1.0 });

        var predictCsv = Path.Combine(_testProjectRoot, "predict.csv");
        File.WriteAllLines(predictCsv, ["X1,X2", "1.0,1.0", "9.0,9.0"]);
        return predictCsv;
    }

    [Fact]
    public async Task Predict_Json_Multiclass_MatchesTheDocumentedRow()
    {
        var predictCsv = await SeedMulticlassAsync();
        var (promoteExit, promoteOut, promoteErr) = await CliRunner.RunAsync("promote", "exp-001");
        Assert.True(promoteExit == 0, promoteOut + promoteErr);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync("predict", predictCsv, "--json");
        Assert.True(exitCode == 0, stdout + stderr);

        var payload = ParsePayload(stdout);
        Assert.Equal("default", payload.GetProperty("model").GetString());
        Assert.Equal("multiclass-classification", payload.GetProperty("task").GetString());
        Assert.Equal(2, payload.GetProperty("count").GetInt32());

        var rows = payload.GetProperty("predictions");
        Assert.Equal(2, rows.GetArrayLength());
        foreach (var row in rows.EnumerateArray())
        {
            var label = row.GetProperty("predictedLabel").GetString();
            Assert.Contains(label, new[] { "cat", "dog", "bird" });

            var probabilities = row.GetProperty("probabilities");
            Assert.Equal(
                new[] { "bird", "cat", "dog" },
                probabilities.EnumerateObject().Select(p => p.Name).OrderBy(n => n, StringComparer.Ordinal));

            // The documented join, asserted the only way that rules out a permuted vocabulary.
            var winner = probabilities.EnumerateObject().OrderByDescending(p => p.Value.GetDouble()).First().Name;
            Assert.Equal(label, winner);

            Assert.InRange(row.GetProperty("confidence").GetDouble(), 0.0, 1.0);

            // The CLI half of the null rule: a field with no value is ABSENT here, while the server
            // emits it as null. A consumer must read "has a value", never "the key exists".
            Assert.False(row.TryGetProperty("score", out _));
            Assert.False(row.TryGetProperty("clusterId", out _));
            Assert.False(row.TryGetProperty("scoreLowerBound", out _));
        }
    }

    /// <summary>
    /// Regression, with the conformal band the document describes. The band is not recomputed at
    /// predict time — it is read from the experiment's stored metrics — so seeding
    /// <c>interval_half_width_90</c> is what a model trained with a band actually leaves behind.
    /// </summary>
    [Fact]
    public async Task Predict_Json_RegressionWithBand_CarriesTheDocumentedBoundsAndConfidence()
    {
        var trainCsv = Path.Combine(_testProjectRoot, "train.csv");
        var random = new Random(42);
        var lines = new List<string> { "X1,X2,Y" };
        for (int i = 0; i < 60; i++)
        {
            double x1 = random.NextDouble() * 10, x2 = random.NextDouble() * 10;
            lines.Add(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"{x1:F4},{x2:F4},{2 * x1 + 3 * x2 + 1:F4}"));
        }
        File.WriteAllLines(trainCsv, lines);

        var ml = new MLContext(seed: 42);
        var inference = ml.Auto().InferColumns(trainCsv, labelColumnName: "Y", separatorChar: ',');
        var data = ml.Data.CreateTextLoader(inference.TextLoaderOptions).Load(trainCsv);
        var model = ml.Regression.Trainers.SingleThreadSdca("Y", featureColumnName: "Features").Fit(data);
        var modelPath = Path.Combine(_testProjectRoot, "reg.zip");
        ml.Model.Save(model, null, modelPath);

        await SaveExperimentAsync("regression", "Y", trainCsv, modelPath,
        [
            new ColumnSchema { Name = "X1", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
            new ColumnSchema { Name = "X2", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
            new ColumnSchema { Name = "Y", DataType = SchemaDataTypes.Numeric, Purpose = "Label" },
        ], new Dictionary<string, double>
        {
            ["r_squared"] = 0.99,
            ["interval_half_width_90"] = 2.0,
            ["residual_std"] = 1.0,
        });

        var predictCsv = Path.Combine(_testProjectRoot, "predict.csv");
        File.WriteAllLines(predictCsv, ["X1,X2", "1.0,1.0", "5.0,5.0"]);

        var (promoteExit, promoteOut, promoteErr) = await CliRunner.RunAsync("promote", "exp-001");
        Assert.True(promoteExit == 0, promoteOut + promoteErr);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync("predict", predictCsv, "--json");
        Assert.True(exitCode == 0, stdout + stderr);

        var payload = ParsePayload(stdout);
        Assert.Equal("regression", payload.GetProperty("task").GetString());

        foreach (var row in payload.GetProperty("predictions").EnumerateArray())
        {
            double score = row.GetProperty("score").GetDouble();
            double lower = row.GetProperty("scoreLowerBound").GetDouble();
            double upper = row.GetProperty("scoreUpperBound").GetDouble();

            // "brackets it" is the document's word — assert the bracket, not just the presence.
            Assert.InRange(score, lower, upper);
            // intervalConfidence is the band's COVERAGE LEVEL, which the document warns not to read as
            // a per-row confidence; confidence is the separate [0,1] the row also carries.
            Assert.Equal(0.90, row.GetProperty("intervalConfidence").GetDouble(), 3);
            Assert.InRange(row.GetProperty("confidence").GetDouble(), 0.0, 1.0);

            // A regression row names no class.
            Assert.False(row.TryGetProperty("predictedLabel", out _));
            Assert.False(row.TryGetProperty("probabilities", out _));
        }
    }

    /// <summary>Writes a two-feature numeric CSV whose rows fall in three well-separated blobs.</summary>
    private string WriteBlobCsv(string name, bool withLabelColumn)
    {
        var path = Path.Combine(_testProjectRoot, name);
        var lines = new List<string> { withLabelColumn ? "X1,X2,Y" : "X1,X2" };
        foreach (var (cx, cy) in new[] { (1.0, 1.0), (5.0, 5.0), (9.0, 9.0) })
            for (int i = 0; i < 20; i++)
            {
                double j = (i % 5 - 2) * 0.1;
                lines.Add(withLabelColumn
                    ? string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{cx + j:F4},{cy - j:F4},0")
                    : string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{cx + j:F4},{cy - j:F4}"));
            }
        File.WriteAllLines(path, lines);
        return path;
    }

    private static ColumnSchema[] TwoNumericFeatures() =>
    [
        new ColumnSchema { Name = "X1", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
        new ColumnSchema { Name = "X2", DataType = SchemaDataTypes.Numeric, Purpose = "Feature" },
    ];

    /// <summary>
    /// Clustering names no class and carries no probability — it reports which cluster and how far the
    /// row sits from each centroid. The document lists exactly those two fields for this task.
    /// </summary>
    [Fact]
    public async Task Predict_Json_Clustering_CarriesClusterIdAndDistances()
    {
        var trainCsv = WriteBlobCsv("train.csv", withLabelColumn: false);

        var ml = new MLContext(seed: 42);
        var data = ml.Data.CreateTextLoader(new Microsoft.ML.Data.TextLoader.Options
        {
            Columns = [
                new Microsoft.ML.Data.TextLoader.Column("X1", Microsoft.ML.Data.DataKind.Single, 0),
                new Microsoft.ML.Data.TextLoader.Column("X2", Microsoft.ML.Data.DataKind.Single, 1)],
            HasHeader = true,
            Separators = [','],
        }).Load(trainCsv);
        var model = ml.Transforms.Concatenate("Features", "X1", "X2")
            .Append(ml.Clustering.Trainers.KMeans("Features", numberOfClusters: 3))
            .Fit(data);

        var modelPath = Path.Combine(_testProjectRoot, "km.zip");
        ml.Model.Save(model, null, modelPath);
        await SaveExperimentAsync("clustering", "", trainCsv, modelPath, TwoNumericFeatures(),
            new Dictionary<string, double> { ["average_distance"] = 0.1 });

        var predictCsv = Path.Combine(_testProjectRoot, "predict.csv");
        File.WriteAllLines(predictCsv, ["X1,X2", "1.0,1.0", "9.0,9.0"]);

        var (promoteExit, promoteOut, promoteErr) = await CliRunner.RunAsync("promote", "exp-001");
        Assert.True(promoteExit == 0, promoteOut + promoteErr);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync("predict", predictCsv, "--json");
        Assert.True(exitCode == 0, stdout + stderr);

        var payload = ParsePayload(stdout);
        Assert.Equal("clustering", payload.GetProperty("task").GetString());
        foreach (var row in payload.GetProperty("predictions").EnumerateArray())
        {
            Assert.True(row.GetProperty("clusterId").GetInt32() >= 0);
            Assert.Equal(3, row.GetProperty("distances").GetArrayLength());
            // No class vocabulary exists for clustering, so neither field the document ties to
            // classification may appear — including confidence, whose classification rule is the
            // winning class probability.
            Assert.False(row.TryGetProperty("predictedLabel", out _));
            Assert.False(row.TryGetProperty("probabilities", out _));
        }
    }

    /// <summary>
    /// Anomaly detection reports a boolean verdict and the raw score behind it, and — unlike the
    /// time-series detector the document singles out — does carry a confidence, derived from distance
    /// to the 0.5 decision boundary.
    /// </summary>
    [Fact]
    public async Task Predict_Json_AnomalyDetection_CarriesVerdictScoreAndConfidence()
    {
        var trainCsv = WriteBlobCsv("train.csv", withLabelColumn: false);

        var ml = new MLContext(seed: 42);
        var data = ml.Data.CreateTextLoader(new Microsoft.ML.Data.TextLoader.Options
        {
            Columns = [
                new Microsoft.ML.Data.TextLoader.Column("X1", Microsoft.ML.Data.DataKind.Single, 0),
                new Microsoft.ML.Data.TextLoader.Column("X2", Microsoft.ML.Data.DataKind.Single, 1)],
            HasHeader = true,
            Separators = [','],
        }).Load(trainCsv);
        var model = ml.Transforms.Concatenate("Features", "X1", "X2")
            .Append(ml.AnomalyDetection.Trainers.RandomizedPca("Features", rank: 1))
            .Fit(data);

        var modelPath = Path.Combine(_testProjectRoot, "pca.zip");
        ml.Model.Save(model, null, modelPath);
        await SaveExperimentAsync("anomaly-detection", "", trainCsv, modelPath, TwoNumericFeatures(),
            new Dictionary<string, double> { ["area_under_roc_curve"] = 0.9 });

        var predictCsv = Path.Combine(_testProjectRoot, "predict.csv");
        File.WriteAllLines(predictCsv, ["X1,X2", "1.0,1.0", "50.0,-30.0"]);

        var (promoteExit, promoteOut, promoteErr) = await CliRunner.RunAsync("promote", "exp-001");
        Assert.True(promoteExit == 0, promoteOut + promoteErr);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync("predict", predictCsv, "--json");
        Assert.True(exitCode == 0, stdout + stderr);

        var payload = ParsePayload(stdout);
        Assert.Equal("anomaly-detection", payload.GetProperty("task").GetString());
        foreach (var row in payload.GetProperty("predictions").EnumerateArray())
        {
            Assert.True(row.GetProperty("isAnomaly").ValueKind is JsonValueKind.True or JsonValueKind.False);
            Assert.True(double.IsFinite(row.GetProperty("anomalyScore").GetDouble()));
            Assert.InRange(row.GetProperty("confidence").GetDouble(), 0.0, 1.0);
            Assert.False(row.TryGetProperty("clusterId", out _));
        }
    }
}
