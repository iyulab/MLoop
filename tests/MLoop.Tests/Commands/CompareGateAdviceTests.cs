using System.Text.Json;
using System.Text.RegularExpressions;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>compare</c> recommends what <c>promote</c> would accept. Its ranking is by metric, and on accuracy a model that
/// answers one class for every row outranks one that learned — so recommending the first of the ranking sent the
/// reader to a <c>promote</c> that refused it.
/// </summary>
[Collection("FileSystem")]
public sealed class CompareGateAdviceTests : IDisposable
{
    private readonly string _root;
    private readonly string _originalDirectory;

    private static readonly Dictionary<string, double> SingleClassModel = new()
    {
        ["accuracy"] = 0.993,
        ["f1_score"] = 0,
        ["precision"] = 0,
        ["recall"] = 0,
        ["negative_recall"] = 1,
        ["auc"] = 0.5,
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

    private static readonly Dictionary<string, double> BetterLearnedModel = new()
    {
        ["accuracy"] = 0.95,
        ["f1_score"] = 0.92,
        ["precision"] = 0.93,
        ["recall"] = 0.9,
        ["negative_recall"] = 0.96,
        ["auc"] = 0.97,
    };

    public CompareGateAdviceTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _root = Path.Combine(Path.GetTempPath(), "mloop-compare-gate-" + Guid.NewGuid());
        Directory.CreateDirectory(Path.Combine(_root, ".mloop"));
        Directory.SetCurrentDirectory(_root);
    }

    public void Dispose()
    {
        Directory.SetCurrentDirectory(_originalDirectory);
        try { Directory.Delete(_root, recursive: true); } catch { }
    }

    private static async Task SaveExperimentAsync(string id, Dictionary<string, double> metrics)
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
            Config = new ExperimentConfig { DataFile = "train.csv", LabelColumn = "y", Metric = "accuracy", TimeLimitSeconds = 60, TestSplit = 0.2 },
            Metrics = metrics,
        }, CancellationToken.None);
    }

    /// <summary>What a terminal shows: colour codes dropped, and a line the table wrapped joined back.</summary>
    private static string Plain(string text) => Regex.Replace(Regex.Replace(text, "\u001b\\[[0-9;]*m", ""), @"\s+", " ");

    [Fact]
    public async Task TheRecommendation_SkipsAnExperimentThePromotionGateRefuses_AndNamesIt()
    {
        await SaveExperimentAsync("exp-001", SingleClassModel);   // first on accuracy, learned nothing
        await SaveExperimentAsync("exp-002", LearnedModel);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync(["compare", "exp-001", "exp-002"]);

        Assert.Equal(0, exitCode);
        var said = Plain(stdout + stderr);
        Assert.Contains("Consider promoting exp-002", said);
        Assert.DoesNotContain("Consider promoting exp-001", said);
        Assert.Contains("exp-001 ranks higher on accuracy but would not be promoted", said);
    }

    [Fact]
    public async Task TheJson_KeepsTheMetricsBest_AndAddsWhatMayBePromoted()
    {
        await SaveExperimentAsync("exp-001", SingleClassModel);
        await SaveExperimentAsync("exp-002", LearnedModel);

        var (exitCode, stdout, _) = await CliRunner.RunAsync(["compare", "exp-001", "exp-002", "--json"]);

        Assert.Equal(0, exitCode);
        var root = JsonDocument.Parse(stdout).RootElement;
        Assert.Equal("exp-001", root.GetProperty("best").GetString());          // the metric's first, as before
        Assert.Equal("exp-002", root.GetProperty("recommended").GetString());   // what promote accepts
        var held = Assert.Single(root.GetProperty("held").EnumerateArray());
        Assert.Equal("exp-001", held.GetProperty("id").GetString());
        Assert.Contains("one class", held.GetProperty("reason").GetString());
    }

    [Fact]
    public async Task WhenNothingIsRefused_TheRecommendationIsTheMetricsBest()
    {
        // The partner of the tests above: with no refusal in the way the advice is the ranking's first, held is empty.
        await SaveExperimentAsync("exp-001", LearnedModel);
        await SaveExperimentAsync("exp-002", BetterLearnedModel);

        var (_, stdout, stderr) = await CliRunner.RunAsync(["compare", "exp-001", "exp-002"]);
        Assert.Contains("Consider promoting exp-002", Plain(stdout + stderr));
        Assert.DoesNotContain("would not be promoted", Plain(stdout + stderr));

        var (_, json, _) = await CliRunner.RunAsync(["compare", "exp-001", "exp-002", "--json"]);
        var root = JsonDocument.Parse(json).RootElement;
        Assert.Equal("exp-002", root.GetProperty("recommended").GetString());
        Assert.Empty(root.GetProperty("held").EnumerateArray());
    }

    [Fact]
    public async Task WhenProductionIsTheBestTheGatesAccept_NothingIsRecommended()
    {
        await SaveExperimentAsync("exp-001", SingleClassModel);
        await SaveExperimentAsync("exp-002", LearnedModel);
        Assert.Equal(0, (await CliRunner.RunAsync(["promote", "exp-002", "--json"])).ExitCode);

        var (_, json, _) = await CliRunner.RunAsync(["compare", "exp-001", "exp-002", "--json"]);

        var root = JsonDocument.Parse(json).RootElement;
        Assert.Equal(JsonValueKind.Null, root.GetProperty("recommended").ValueKind);
        Assert.Equal("exp-001", Assert.Single(root.GetProperty("held").EnumerateArray()).GetProperty("id").GetString());
    }
}
