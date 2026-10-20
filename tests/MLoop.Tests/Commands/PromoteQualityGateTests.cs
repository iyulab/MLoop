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

    // Answers "negative" for every row and ranks no better than chance: it learned nothing.
    private static readonly Dictionary<string, double> SingleClassModel = new()
    {
        ["accuracy"] = 0.993, ["f1_score"] = 0, ["precision"] = 0, ["recall"] = 0,
        ["negative_recall"] = 1, ["auc"] = 0.5,
    };

    // Also answers "negative" for every row at the default threshold, but ranks the classes well:
    // its scores are usable, its predicted labels are not. Measured shape of a model served for a
    // month by a consumer deciding by score.
    private static readonly Dictionary<string, double> RanksButSitsOnOneClass = new()
    {
        ["accuracy"] = 0.969, ["f1_score"] = 0, ["precision"] = 0, ["recall"] = 0,
        ["negative_recall"] = 1, ["auc"] = 0.968,
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
    public async Task ExplicitPromote_AModelThatRanksButSitsOnOneClass_PassesWithAThresholdWarning()
    {
        await SaveExperimentAsync("exp-001", RanksButSitsOnOneClass, metric: "auc");

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync(["promote", "exp-001", "--json"]);

        Assert.Equal(0, exitCode);
        var warning = JsonDocument.Parse(stdout).RootElement.GetProperty("qualityWarning").GetString()!;
        Assert.Contains("auc 0.9680", warning);
        Assert.Contains("threshold", warning);
        Assert.Contains("threshold", stderr);
        Assert.Equal("exp-001", ProductionExperiment());
    }

    [Fact]
    public async Task ExplicitPromote_AModelThatRanksButScoresZeroOnItsOwnMetric_IsStillRefused()
    {
        // Ranking does not excuse the metric the project asked to optimize: F1 0 is at its floor.
        await SaveExperimentAsync("exp-001", RanksButSitsOnOneClass, metric: "f1_score");

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["promote", "exp-001", "--json"]);

        Assert.Equal(1, exitCode);
        Assert.Contains("f1_score", JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString());
        Assert.Null(ProductionExperiment());
    }

    [Fact]
    public async Task PromoteBest_ChoosesAmongTheModelsTheGateAccepts()
    {
        // On accuracy the single-class model (0.993) outranks the one that learned (0.91).
        await SaveExperimentAsync("exp-001", SingleClassModel, metric: "accuracy");
        await SaveExperimentAsync("exp-002", LearnedModel, metric: "accuracy");

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["promote", "--best", "--json"]);

        Assert.Equal(0, exitCode);
        Assert.Equal("exp-002", JsonDocument.Parse(stdout).RootElement.GetProperty("experimentId").GetString());
    }

    [Fact]
    public async Task PromoteBest_WhenNoModelPasses_RefusesWithEachReason()
    {
        await SaveExperimentAsync("exp-001", SingleClassModel, metric: "accuracy");

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["promote", "--best", "--json"]);

        Assert.Equal(1, exitCode);
        var error = JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString()!;
        Assert.Contains("exp-001:", error);
        Assert.Contains("quality gate", error);
        Assert.Null(ProductionExperiment());
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
        Assert.Equal(JsonValueKind.Null, byId["exp-002"].GetProperty("qualityWarning").ValueKind);
    }

    [Fact]
    public async Task ListJson_TellsTheModelThatLearnedNothingFromTheOneThatSitsOnItsThreshold()
    {
        await SaveExperimentAsync("exp-001", SingleClassModel, metric: "auc");
        await SaveExperimentAsync("exp-002", RanksButSitsOnOneClass, metric: "auc");

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["list", "--json"]);

        Assert.Equal(0, exitCode);
        var byId = JsonDocument.Parse(stdout).RootElement.GetProperty("experiments").EnumerateArray()
            .ToDictionary(e => e.GetProperty("experimentId").GetString()!);

        Assert.NotEqual(JsonValueKind.Null, byId["exp-001"].GetProperty("qualityGate").ValueKind);
        Assert.Equal(JsonValueKind.Null, byId["exp-001"].GetProperty("qualityWarning").ValueKind);
        Assert.Equal(JsonValueKind.Null, byId["exp-002"].GetProperty("qualityGate").ValueKind);
        Assert.Contains("auc 0.9680", byId["exp-002"].GetProperty("qualityWarning").GetString());
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

    [Fact]
    public async Task Rollback_ReturnsProductionToTheExperimentItReplaced()
    {
        await SaveExperimentAsync("exp-001", LearnedModel);
        await SaveExperimentAsync("exp-002", LearnedModel);
        Assert.Equal(0, (await CliRunner.RunAsync(["promote", "exp-001", "--json"])).ExitCode);
        Assert.Equal(0, (await CliRunner.RunAsync(["promote", "exp-002", "--json"])).ExitCode);

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["rollback", "--json"]);

        Assert.Equal(0, exitCode);
        using var result = JsonDocument.Parse(stdout);
        Assert.Equal("exp-001", result.RootElement.GetProperty("experimentId").GetString());
        Assert.Equal("exp-002", result.RootElement.GetProperty("previousProduction").GetString());
        Assert.Equal("exp-001", ProductionExperiment());
    }

    [Fact]
    public async Task Rollback_RefusesAnExperimentThatNeverServed()
    {
        await SaveExperimentAsync("exp-001", LearnedModel);
        await SaveExperimentAsync("exp-002", LearnedModel);
        Assert.Equal(0, (await CliRunner.RunAsync(["promote", "exp-001", "--json"])).ExitCode);

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["rollback", "--to", "exp-002", "--json"]);

        Assert.Equal(1, exitCode);
        Assert.Contains("never been in production", JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString());
        Assert.Equal("exp-001", ProductionExperiment());
    }
}
