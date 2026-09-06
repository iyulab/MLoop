using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;

namespace MLoop.Tests.Infrastructure.FileSystem;

/// <summary>
/// Pins the trial history an experiment leaves behind. ML.NET hands back every trial's trainer,
/// metrics and runtime on <c>ExperimentResult.RunDetails</c>; MLoop used to take <c>BestRun</c> off
/// it and drop the rest, so each run destroyed the only record of what else the search tried.
/// </summary>
[Collection("FileSystem")]
public class ExperimentTrialHistoryTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly ExperimentStore _experimentStore;
    private const string ModelName = "default";

    public ExperimentTrialHistoryTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-trials-" + Guid.NewGuid());
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, ".mloop"));
        Directory.SetCurrentDirectory(_testProjectRoot);

        var fileSystem = new FileSystemManager();
        _experimentStore = new ExperimentStore(fileSystem, new ProjectDiscovery(fileSystem));
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { /* best effort */ }
        try { Directory.Delete(_testProjectRoot, recursive: true); } catch { /* best effort */ }
    }

    private static TrialRecord Trial(int n, string trainer, string metric, double value, double runtime = 1.0) =>
        new()
        {
            TrialNumber = n,
            Trainer = TrainerDescriptor.Of(trainer),
            Metrics = new Dictionary<string, double> { [metric] = value },
            RuntimeSeconds = runtime
        };

    private static ExperimentData Experiment(
        string experimentId, IReadOnlyList<TrialRecord> trials, string? rankingMetric, string task = "regression") =>
        new()
        {
            ModelName = ModelName,
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = "completed",
            Task = task,
            Config = new ExperimentConfig
            {
                DataFile = "test.csv",
                LabelColumn = "label",
                TimeLimitSeconds = 60,
                Metric = rankingMetric ?? "r_squared",
                TestSplit = 0.2
            },
            Trials = trials,
            RankingMetric = rankingMetric
        };

    private async Task<string> SaveAsync(IReadOnlyList<TrialRecord> trials, string? rankingMetric, string task = "regression")
    {
        var id = await _experimentStore.GenerateIdAsync(ModelName, CancellationToken.None);
        await _experimentStore.SaveAsync(ModelName, Experiment(id, trials, rankingMetric, task), CancellationToken.None);
        return _experimentStore.GetExperimentPath(ModelName, id);
    }

    private static JsonElement ReadLeaderboard(string experimentPath) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(experimentPath, "leaderboard.json"))).RootElement;

    [Fact]
    public async Task Trials_are_written_one_json_object_per_line_in_completion_order()
    {
        var path = await SaveAsync([
            Trial(1, "FastTreeRegression", "r_squared", 0.71, runtime: 0.9),
            Trial(2, "LightGbmRegression", "r_squared", 0.88, runtime: 1.4)
        ], rankingMetric: "r_squared");

        var lines = await File.ReadAllLinesAsync(Path.Combine(path, "trials.ndjson"));

        Assert.Equal(2, lines.Length);
        var first = JsonDocument.Parse(lines[0]).RootElement;
        Assert.Equal(1, first.GetProperty("trialNumber").GetInt32());
        Assert.Equal("FastTreeRegression", first.GetProperty("trainerName").GetString());
        Assert.Equal(0.9, first.GetProperty("runtimeSeconds").GetDouble());
        Assert.Equal(0.71, first.GetProperty("metrics").GetProperty("r_squared").GetDouble());
        Assert.Equal(2, JsonDocument.Parse(lines[1]).RootElement.GetProperty("trialNumber").GetInt32());
    }

    /// <summary>
    /// A leaderboard whose rows are the same trainer at different hyperparameters is where the
    /// folded name hurts most: <c>KMeans (k=2)</c> … <c>KMeans (k=10)</c> reads as ten trainers. The
    /// display form stays in <c>trainerName</c> for readers that have it; the parts sit beside it.
    /// </summary>
    [Fact]
    public async Task A_trial_row_carries_the_trainer_in_parts_beside_its_display_name()
    {
        var path = await SaveAsync([
            new TrialRecord
            {
                TrialNumber = 1,
                Trainer = TrainerDescriptor.Of("KMeans", ("k", 3)),
                Metrics = new Dictionary<string, double> { ["davies_bouldin_index"] = 0.19 },
                RuntimeSeconds = 0.2
            }
        ], rankingMetric: "davies_bouldin_index");

        var row = JsonDocument.Parse(
            (await File.ReadAllLinesAsync(Path.Combine(path, "trials.ndjson")))[0]).RootElement;

        Assert.Equal("KMeans (k=3)", row.GetProperty("trainerName").GetString());
        Assert.Equal("KMeans", row.GetProperty("trainer").GetProperty("name").GetString());
        Assert.Equal("3", row.GetProperty("trainer").GetProperty("params").GetProperty("k").GetString());
    }

    [Fact]
    public async Task Leaderboard_puts_the_best_trial_first_for_a_higher_is_better_metric()
    {
        var path = await SaveAsync([
            Trial(1, "A", "r_squared", 0.71),
            Trial(2, "B", "r_squared", 0.88),
            Trial(3, "C", "r_squared", 0.55)
        ], rankingMetric: "r_squared");

        var leaderboard = ReadLeaderboard(path);

        Assert.Equal("higher_is_better", leaderboard.GetProperty("direction").GetString());
        Assert.Equal(3, leaderboard.GetProperty("trialCount").GetInt32());
        Assert.Equal(
            ["B", "A", "C"],
            leaderboard.GetProperty("trials").EnumerateArray().Select(t => t.GetProperty("trainerName").GetString()));
    }

    [Fact]
    public async Task Leaderboard_inverts_the_order_for_a_lower_is_better_metric()
    {
        // The direction failure was exactly this: a local lower-is-better test that missed a metric ranked
        // the worst model first. The direction comes from the MetricDirection authority instead.
        var path = await SaveAsync([
            Trial(1, "A", "rmse", 12.0),
            Trial(2, "B", "rmse", 3.5),
            Trial(3, "C", "rmse", 7.25)
        ], rankingMetric: "rmse");

        var leaderboard = ReadLeaderboard(path);

        Assert.Equal("lower_is_better", leaderboard.GetProperty("direction").GetString());
        Assert.Equal(
            ["B", "C", "A"],
            leaderboard.GetProperty("trials").EnumerateArray().Select(t => t.GetProperty("trainerName").GetString()));
    }

    [Fact]
    public async Task An_unrecognized_metric_leaves_the_trials_unranked_rather_than_guessing()
    {
        var path = await SaveAsync([
            Trial(1, "A", "custom_score", 1.0),
            Trial(2, "B", "custom_score", 9.0)
        ], rankingMetric: "custom_score");

        var leaderboard = ReadLeaderboard(path);

        Assert.Equal("unknown", leaderboard.GetProperty("direction").GetString());
        Assert.Equal(
            ["A", "B"],
            leaderboard.GetProperty("trials").EnumerateArray().Select(t => t.GetProperty("trainerName").GetString()));
    }

    [Fact]
    public async Task A_trial_missing_the_ranking_metric_sorts_last_instead_of_disappearing()
    {
        var withoutMetric = new TrialRecord
        {
            TrialNumber = 2,
            Trainer = TrainerDescriptor.Of("NoMetric"),
            Metrics = new Dictionary<string, double>(),
            RuntimeSeconds = 0.2
        };

        var path = await SaveAsync([Trial(1, "A", "r_squared", 0.4), withoutMetric], rankingMetric: "r_squared");

        var names = ReadLeaderboard(path).GetProperty("trials").EnumerateArray()
            .Select(t => t.GetProperty("trainerName").GetString());

        Assert.Equal(["A", "NoMetric"], names);
    }

    [Fact]
    public async Task A_run_that_reported_no_trial_writes_no_trial_files_at_all()
    {
        // The paths with no metric to report (object detection, QA) record no trials, so there is
        // nothing to write. An empty leaderboard would claim a search happened and turned up
        // nothing. Note this is no longer "single pipeline": a task that fits one pipeline and
        // measures it does record that one candidate.
        var path = await SaveAsync([], rankingMetric: null, task: "anomaly-detection");

        Assert.False(File.Exists(Path.Combine(path, "trials.ndjson")));
        Assert.False(File.Exists(Path.Combine(path, "leaderboard.json")));
    }
}
