using System.Text.Json;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Commands;

/// <summary>
/// An explicit <c>mloop promote</c> applies the quality gate training applies before it promotes.
/// </summary>
/// <remarks>
/// Measured on 0.34.0: a binary model trained on 0.6% positives answered "negative" to every row —
/// accuracy 0.993, F1 0 — and training held it in staging, but <c>mloop promote exp-001</c> put it
/// in production without a word. A consumer reviewing promotions saw only the accuracy.
/// </remarks>
[Collection("FileSystem")]
public sealed class PromoteQualityGateTests : IDisposable
{
    private readonly string _root;
    private readonly string _originalDirectory;

    private static readonly Dictionary<string, double> SingleClassModel = new()
    {
        ["accuracy"] = 0.993, ["f1_score"] = 0, ["precision"] = 0, ["recall"] = 0,
        ["negative_recall"] = 1, ["auc"] = 0.95,
    };

    private static readonly Dictionary<string, double> LearnedModel = new()
    {
        ["accuracy"] = 0.91, ["f1_score"] = 0.88, ["precision"] = 0.9, ["recall"] = 0.86,
        ["negative_recall"] = 0.94, ["auc"] = 0.95,
    };

    public PromoteQualityGateTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _root = Path.Combine(Path.GetTempPath(), "mloop-gate-" + Guid.NewGuid());
        Directory.CreateDirectory(Path.Combine(_root, ".mloop"));
        Directory.SetCurrentDirectory(_root);
    }

    public void Dispose()
    {
        Directory.SetCurrentDirectory(_originalDirectory);
        try { Directory.Delete(_root, recursive: true); } catch { }
    }

    private static async Task SaveExperimentAsync(string id, Dictionary<string, double> metrics, string metric = "f1_score")
    {
        var fileSystem = new FileSystemManager();
        var store = new ExperimentStore(fileSystem, new ProjectDiscovery(fileSystem));
        var path = store.GetExperimentPath("default", id);
        Directory.CreateDirectory(path);
        await File.WriteAllTextAsync(Path.Combine(path, "model.zip"), "model");
        await store.SaveAsync("default", new ExperimentData
        {
            ModelName = "default",
            ExperimentId = id,
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = "binary-classification",
            Config = new ExperimentConfig { DataFile = "train.csv", LabelColumn = "y", Metric = metric, TimeLimitSeconds = 60, TestSplit = 0.2 },
            Metrics = metrics,
        }, CancellationToken.None);
    }

    private static string? ProductionExperiment()
    {
        var fileSystem = new FileSystemManager();
        var discovery = new ProjectDiscovery(fileSystem);
        var registry = new ModelRegistry(fileSystem, discovery, new ExperimentStore(fileSystem, discovery));
        return registry.GetProductionAsync("default").GetAwaiter().GetResult()?.ExperimentId;
    }

    [Fact]
    public async Task ExplicitPromote_RefusesASingleClassModel_AndSaysWhy()
    {
        await SaveExperimentAsync("exp-001", SingleClassModel);

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["promote", "exp-001", "--json"]);

        Assert.Equal(1, exitCode);
        var error = JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString()!;
        Assert.Contains("quality gate", error);
        Assert.Contains("f1_score", error);
        Assert.Contains("--force", error);
        Assert.Null(ProductionExperiment());
    }

    [Fact]
    public async Task ExplicitPromote_WithForce_PromotesAndWarns()
    {
        await SaveExperimentAsync("exp-001", SingleClassModel);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync(["promote", "exp-001", "--force", "--json"]);

        Assert.Equal(0, exitCode);
        Assert.Equal("promoted", JsonDocument.Parse(stdout).RootElement.GetProperty("status").GetString());
        Assert.Contains("quality gate", stderr);
        Assert.Equal("exp-001", ProductionExperiment());
    }

    [Fact]
    public async Task ExplicitPromote_ALearnedModel_PassesWithoutForce()
    {
        await SaveExperimentAsync("exp-001", LearnedModel);

        var (exitCode, _, _) = await CliRunner.RunAsync(["promote", "exp-001", "--json"]);

        Assert.Equal(0, exitCode);
        Assert.Equal("exp-001", ProductionExperiment());
    }

    [Fact]
    public async Task ListJson_CarriesEachExperimentsMetricsAndTheGatesVerdict()
    {
        await SaveExperimentAsync("exp-001", SingleClassModel);
        await SaveExperimentAsync("exp-002", LearnedModel);

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["list", "--json"]);

        Assert.Equal(0, exitCode);
        var byId = JsonDocument.Parse(stdout).RootElement.GetProperty("experiments").EnumerateArray()
            .ToDictionary(e => e.GetProperty("experimentId").GetString()!);

        Assert.Equal(0, byId["exp-001"].GetProperty("metrics").GetProperty("f1_score").GetDouble());
        Assert.Contains("f1_score", byId["exp-001"].GetProperty("qualityGate").GetString());
        Assert.Equal(0.88, byId["exp-002"].GetProperty("metrics").GetProperty("f1_score").GetDouble());
        Assert.Equal(JsonValueKind.Null, byId["exp-002"].GetProperty("qualityGate").ValueKind);
    }

    [Fact]
    public async Task ExperimentIndex_NamesTheMetricItsValueCameFrom()
    {
        // A configured spelling the metrics file does not use ("f1" for "f1_score") once let the
        // index record the accuracy under the name "f1".
        await SaveExperimentAsync("exp-001", SingleClassModel, metric: "f1");

        using var index = JsonDocument.Parse(await File.ReadAllTextAsync(
            Path.Combine(_root, "models", "default", "experiment-index.json")));
        var entry = index.RootElement.GetProperty("experiments")[0];

        Assert.Equal("f1_score", entry.GetProperty("metricName").GetString());
        Assert.Equal(0, entry.GetProperty("bestMetric").GetDouble());
    }
}
