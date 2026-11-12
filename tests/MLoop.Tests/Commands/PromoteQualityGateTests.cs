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
        ["accuracy"] = 0.993,
        ["f1_score"] = 0,
        ["precision"] = 0,
        ["recall"] = 0,
        ["negative_recall"] = 1,
        ["auc"] = 0.5,
    };

    // Also answers "negative" for every row at the default threshold, but ranks the classes well:
    // its scores are usable, its predicted labels are not. Measured shape of a model served for a
    // month by a consumer deciding by score.
    private static readonly Dictionary<string, double> RanksButSitsOnOneClass = new()
    {
        ["accuracy"] = 0.969,
        ["f1_score"] = 0,
        ["precision"] = 0,
        ["recall"] = 0,
        ["negative_recall"] = 1,
        ["auc"] = 0.968,
    };

    private static readonly Dictionary<string, double> LearnedModel = new()
    {
        ["accuracy"] = 0.91,
        ["f1_score"] = 0.88,
        ["precision"] = 0.9,
        ["recall"] = 0.86,
        ["negative_recall"] = 0.94,
        ["auc"] = 0.95,
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

    private static async Task SaveExperimentAsync(string id, Dictionary<string, double> metrics, string metric = "f1_score", long? testRows = null, double? majority = null)
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
            Config = new ExperimentConfig
            {
                DataFile = "train.csv",
                LabelColumn = "y",
                Metric = metric,
                TimeLimitSeconds = 60,
                TestSplit = 0.2,
                InputSchema = majority is null ? null : new MLoop.Core.Models.InputSchemaInfo
                {
                    Columns = [new MLoop.Core.Models.ColumnSchema { Name = "y", DataType = "Categorical", Purpose = "Label", UniqueValueCount = 2, MajorityClassRatio = majority }],
                    CapturedAt = DateTime.UtcNow,
                },
            },
            Metrics = metrics,
            Result = testRows is null ? null : new ExperimentResult { BestTrainer = "FastTree", TrainingTimeSeconds = 1, TestRows = testRows },
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

    /// <summary>
    /// Promoting the experiment that is already in production changed nothing yet backed the model up onto itself and
    /// printed "Promotion complete" — run twice it left two backup directories. It is now a stated no-op; --force still
    /// promotes again.
    /// </summary>
    [Fact]
    public async Task PromotingWhatIsAlreadyInProduction_ChangesNothing_AndSaysSo()
    {
        await SaveExperimentAsync("exp-001", LearnedModel);
        await SaveExperimentAsync("exp-002", LearnedModel);
        Assert.Equal(0, (await CliRunner.RunAsync(["promote", "exp-001", "--json"])).ExitCode);
        Assert.Equal(0, (await CliRunner.RunAsync(["promote", "exp-002", "--json"])).ExitCode);   // exp-001 is backed up once
        var backups = Path.Combine(_root, "models", "default", "backups");
        var backedUp = Directory.GetDirectories(backups).Length;

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["promote", "exp-002", "--json"]);

        Assert.Equal(0, exitCode);
        using var doc = JsonDocument.Parse(stdout);
        Assert.Equal("unchanged", doc.RootElement.GetProperty("status").GetString());
        Assert.Equal(backedUp, Directory.GetDirectories(backups).Length);
        Assert.Equal("exp-002", ProductionExperiment());

        var (forced, forcedOut, _) = await CliRunner.RunAsync(["promote", "exp-002", "--force", "--json"]);
        Assert.Equal(0, forced);
        Assert.Equal("promoted", JsonDocument.Parse(forcedOut).RootElement.GetProperty("status").GetString());
    }

    // Passes the gate (F1 > 0, AUC > 0.5) yet scores far below LearnedModel on every metric.
    private static readonly Dictionary<string, double> WeakButPassing = new()
    {
        ["accuracy"] = 0.575,
        ["f1_score"] = 0.61,
        ["precision"] = 0.59,
        ["recall"] = 0.64,
        ["negative_recall"] = 0.5,
        ["auc"] = 0.57,
    };

    /// <summary>
    /// An explicit promote replaced a production model scoring accuracy 0.80 with one at 0.575 (AUC 0.57), the gate
    /// "passed", and the output said only "This will replace the current production model" — training would have left the
    /// weaker model in staging and said why; the explicit path made the same comparison silently.
    /// </summary>
    [Fact]
    public async Task ExplicitPromote_OfAModelThatDoesNotBeatProduction_SaysSo_AndStillPromotes()
    {
        await SaveExperimentAsync("exp-001", LearnedModel, metric: "f1_score");
        await SaveExperimentAsync("exp-002", WeakButPassing, metric: "f1_score");
        Assert.Equal(0, (await CliRunner.RunAsync(["promote", "exp-001", "--json"])).ExitCode);

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["promote", "exp-002", "--force"]);

        Assert.Equal(0, exitCode);
        var said = string.Join(' ', stdout.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));   // the console wraps lines
        Assert.Contains("would have left it in staging", said, StringComparison.Ordinal);
        Assert.Contains("does not beat production (exp-001) on f1_score: 0.61 vs 0.88", said, StringComparison.Ordinal);
        Assert.Equal("exp-002", ProductionExperiment());   // the explicit act is still the owner's
    }

    [Fact]
    public async Task ExplicitPromote_Json_CarriesTheSameReason_AndNullWhenThereIsNone()
    {
        await SaveExperimentAsync("exp-001", LearnedModel, metric: "f1_score");
        await SaveExperimentAsync("exp-002", WeakButPassing, metric: "f1_score");
        await SaveExperimentAsync("exp-003", LearnedModel, metric: "f1_score");

        var (_, first, _) = await CliRunner.RunAsync(["promote", "exp-001", "--json"]);
        Assert.Equal(JsonValueKind.Null, JsonDocument.Parse(first).RootElement.GetProperty("notBetterThanProduction").ValueKind);

        var (weakerExit, weaker, _) = await CliRunner.RunAsync(["promote", "exp-002", "--json"]);
        Assert.Equal(0, weakerExit);
        Assert.StartsWith("it does not beat production (exp-001) on f1_score: 0.61 vs 0.88",
            JsonDocument.Parse(weaker).RootElement.GetProperty("notBetterThanProduction").GetString(), StringComparison.Ordinal);

        // exp-003 equals exp-001 on every metric and production is exp-002 (0.61): it beats production, no reason.
        var (_, stronger, _) = await CliRunner.RunAsync(["promote", "exp-003", "--json"]);
        Assert.Equal(JsonValueKind.Null, JsonDocument.Parse(stronger).RootElement.GetProperty("notBetterThanProduction").ValueKind);
    }

    /// <summary>
    /// The held-out row count is recorded with the experiment, so a promote that runs long after the training can still say a score
    /// cannot be told from chance — in the human output, as <c>qualityWarning</c> in <c>--json</c>, and in the promotion history.
    /// </summary>
    [Fact]
    public async Task ExplicitPromote_SaysAScoreWithinTheNoiseOfTheFloorCannotBeToldFromChance_AndStillPromotes()
    {
        await SaveExperimentAsync("exp-001", new() { ["accuracy"] = 0.52, ["f1_score"] = 0.5, ["auc"] = 0.55 }, metric: "accuracy", testRows: 240, majority: 0.51);

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["promote", "exp-001", "--json"]);

        Assert.Equal(0, exitCode);
        var warning = JsonDocument.Parse(stdout).RootElement.GetProperty("qualityWarning").GetString()!;
        Assert.Contains("on 240 held-out rows", warning, StringComparison.Ordinal);
        Assert.Contains("cannot be told from chance", warning, StringComparison.Ordinal);
        Assert.Equal("exp-001", ProductionExperiment());   // said, not refused
    }

    [Fact]
    public async Task ExplicitPromote_SaysNothingOfChance_ForAnExperimentWithoutARowCount_OrAClearlyBetterScore()
    {
        // recorded before the field: no n to judge by
        await SaveExperimentAsync("exp-001", new() { ["accuracy"] = 0.52, ["f1_score"] = 0.5, ["auc"] = 0.55 }, metric: "accuracy", majority: 0.51);
        var (_, old, _) = await CliRunner.RunAsync(["promote", "exp-001", "--json"]);
        Assert.False(JsonDocument.Parse(old).RootElement.TryGetProperty("qualityWarning", out var w) && w.ValueKind == JsonValueKind.String);

        // a real margin on the same row count
        await SaveExperimentAsync("exp-002", new() { ["accuracy"] = 0.85, ["f1_score"] = 0.8, ["auc"] = 0.9 }, metric: "accuracy", testRows: 240, majority: 0.51);
        var (_, clear, _) = await CliRunner.RunAsync(["promote", "exp-002", "--json"]);
        Assert.False(JsonDocument.Parse(clear).RootElement.TryGetProperty("qualityWarning", out var w2) && w2.ValueKind == JsonValueKind.String);
    }

    [Fact]
    public async Task TheHeldOutRowCount_SurvivesTheRecord_AndIsAbsentWhenNotKept()
    {
        await SaveExperimentAsync("exp-001", LearnedModel, testRows: 240);
        await SaveExperimentAsync("exp-002", LearnedModel);

        var fileSystem = new FileSystemManager();
        var store = new ExperimentStore(fileSystem, new ProjectDiscovery(fileSystem));
        Assert.Equal(240L, (await store.LoadAsync("default", "exp-001", CancellationToken.None)).Result?.TestRows);
        Assert.Null((await store.LoadAsync("default", "exp-002", CancellationToken.None)).Result?.TestRows);

        var written = await File.ReadAllTextAsync(Path.Combine(store.GetExperimentPath("default", "exp-002"), "metadata.json"));
        Assert.DoesNotContain("testRows", written, StringComparison.Ordinal);   // a field never kept is not written
    }

    [Fact]
    public async Task ExplicitPromote_OfAModelThatBeatsProduction_AddsNoNote()
    {
        await SaveExperimentAsync("exp-001", WeakButPassing, metric: "f1_score");
        await SaveExperimentAsync("exp-002", LearnedModel, metric: "f1_score");
        Assert.Equal(0, (await CliRunner.RunAsync(["promote", "exp-001", "--json"])).ExitCode);

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["promote", "exp-002", "--force"]);

        Assert.Equal(0, exitCode);
        Assert.DoesNotContain("would have left it in staging", stdout, StringComparison.Ordinal);
        Assert.Equal("exp-002", ProductionExperiment());
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
