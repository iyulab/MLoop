using Microsoft.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// The leaderboard side of the same data the progress channel streams live: what the search tried
/// and how each trial scored, taken off <c>ExperimentResult.RunDetails</c> after the run rather than
/// from the progress channel — RunDetails carries each trial's full metric set and its own runtime,
/// and is there whether or not anyone was listening to progress.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class TrialHistoryCollectionTests : IDisposable
{
    private readonly string _tmpDir;

    public TrialHistoryCollectionTests()
    {
        _tmpDir = Path.Combine(Path.GetTempPath(), $"mloop_trialhistory_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tmpDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tmpDir))
            Directory.Delete(_tmpDir, recursive: true);
    }

    private async Task<AutoMLResult> TrainAsync(string name, IEnumerable<string> lines, string task, string label,
        string? metric = null, int timeLimitSeconds = 20)
    {
        var path = Path.Combine(_tmpDir, name);
        await File.WriteAllLinesAsync(path, lines);

        var ctx = new MLContext(seed: 42);
        var runner = new AutoMLRunner(ctx, new CsvDataLoader(ctx), _tmpDir);
        return await runner.RunAsync(new TrainingConfig
        {
            ModelName = "trialhistory",
            DataFile = path,
            LabelColumn = label,
            Task = task,
            TimeLimitSeconds = timeLimitSeconds,
            Metric = metric ?? string.Empty
        }, progress: null, CancellationToken.None);
    }

    /// <summary>
    /// Deliberately not a noiseless linear target: with an exactly-solvable response some trainers
    /// score a non-finite metric on the validation fold and AutoML then ends the whole run with
    /// "the metric for all completed trials are NaN or Infinity" (measured). A small deterministic
    /// wobble keeps the fixture learnable while leaving real residuals to score.
    /// </summary>
    private static List<string> RegressionFixture(int rows)
    {
        var lines = new List<string> { "age,score,response" };
        for (int i = 0; i < rows; i++)
        {
            var age = 20 + (i % 50);
            var score = i % 30;
            var wobble = ((i * 37) % 11) - 5;
            lines.Add($"{age},{score},{age * 2.0 + score * 0.5 + wobble}");
        }
        return lines;
    }

    [Fact]
    public async Task A_search_keeps_every_completed_trial_not_just_the_best_one()
    {
        // Collected with no progress listener attached on purpose: the history must not depend on
        // anyone watching the run.
        var result = await TrainAsync("regression.csv", RegressionFixture(300), "regression", "response", metric: "r_squared");

        Assert.NotEmpty(result.Trials);
        Assert.Equal(Enumerable.Range(1, result.Trials.Count), result.Trials.Select(t => t.TrialNumber));
        Assert.All(result.Trials, t =>
        {
            Assert.False(string.IsNullOrWhiteSpace(t.TrainerName));
            Assert.True(t.RuntimeSeconds >= 0);
            Assert.Contains("r_squared", t.Metrics.Keys);
            Assert.All(t.Metrics.Values, v => Assert.True(double.IsFinite(v), "a non-finite metric was recorded"));
        });
    }

    [Fact]
    public async Task The_history_names_the_metric_the_search_ranked_by()
    {
        var result = await TrainAsync("regression_rmse.csv", RegressionFixture(300), "regression", "response", metric: "rmse");

        Assert.Equal("rmse", result.RankingMetric);
        Assert.All(result.Trials, t => Assert.Contains("rmse", t.Metrics.Keys));
    }

    [Fact]
    public async Task Each_trial_carries_the_full_metric_set_not_only_the_ranked_one()
    {
        // The leaderboard is worth keeping because it answers "how did the runner-up do on the other
        // metrics" — a single value per trial would not.
        var result = await TrainAsync("regression_full.csv", RegressionFixture(300), "regression", "response", metric: "r_squared");

        var trial = result.Trials[0];
        Assert.Contains("r_squared", trial.Metrics.Keys);
        Assert.Contains("rmse", trial.Metrics.Keys);
        Assert.Contains("mae", trial.Metrics.Keys);
    }

    [Fact]
    public async Task A_single_pipeline_task_reports_no_trial_history()
    {
        // Anomaly detection fits one explicit pipeline; there is no search and therefore no
        // leaderboard. Its one result is already the experiment's metrics.
        var lines = new List<string> { "v1,v2" };
        for (int i = 0; i < 200; i++)
            lines.Add($"{i % 23},{(i % 47) * 0.5}");

        var result = await TrainAsync("anomaly.csv", lines, "anomaly-detection", label: string.Empty);

        Assert.Empty(result.Trials);
        Assert.Null(result.RankingMetric);
    }
}
