using Microsoft.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// The regression these exist for: binary / multiclass / regression all accepted an
/// <see cref="IProgress{TrainingProgress}"/>, threaded it down, and never called
/// <c>Report</c> on it — while the six non-tabular tasks did. Downstream that showed up as a
/// progress bar frozen at 0%, an auto-time probe summary that always said "0 trials", and a
/// consumer process with nothing to display for the entire training run.
///
/// Measured against ML.NET AutoML directly: a healthy run reports every completed trial
/// (10 trials → 10 reports) and reports none that threw. So "at least one report on a run that
/// produced a model" is the honest assertion — the exact trial count is AutoML's to decide.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class TabularTrialProgressTests : IDisposable
{
    private readonly string _tmpDir;

    public TabularTrialProgressTests()
    {
        _tmpDir = Path.Combine(Path.GetTempPath(), $"mloop_trialprogress_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tmpDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tmpDir))
            Directory.Delete(_tmpDir, recursive: true);
    }

    /// <summary>Collects inline — Progress&lt;T&gt; would hand callbacks to the thread pool and race the asserts.</summary>
    private sealed class CollectingProgress : IProgress<TrainingProgress>
    {
        private readonly List<TrainingProgress> _items = [];
        public void Report(TrainingProgress value) { lock (_items) _items.Add(value); }
        public IReadOnlyList<TrainingProgress> Trials
        {
            get { lock (_items) return _items.Where(p => p.Phase is null).ToList(); }
        }
    }

    private async Task<string> WriteCsvAsync(string name, IEnumerable<string> lines)
    {
        var path = Path.Combine(_tmpDir, name);
        await File.WriteAllLinesAsync(path, lines);
        return path;
    }

    private async Task<(AutoMLResult Result, CollectingProgress Progress)> TrainAsync(
        string csvPath, string task, string label, string? metric = null, int timeLimitSeconds = 10)
    {
        var ctx = new MLContext(seed: 42);
        var runner = new AutoMLRunner(ctx, new CsvDataLoader(ctx), _tmpDir);
        var config = new TrainingConfig
        {
            ModelName = "trialprogress",
            DataFile = csvPath,
            LabelColumn = label,
            Task = task,
            TimeLimitSeconds = timeLimitSeconds,
            Metric = metric ?? string.Empty
        };

        var progress = new CollectingProgress();
        var result = await runner.RunAsync(config, progress, CancellationToken.None);
        return (result, progress);
    }

    private static void AssertTrialsWereReported(CollectingProgress progress, string expectedMetricName)
    {
        var trials = progress.Trials;

        Assert.NotEmpty(trials);
        Assert.Equal(Enumerable.Range(1, trials.Count), trials.Select(t => t.TrialNumber));
        Assert.All(trials, t =>
        {
            Assert.False(string.IsNullOrWhiteSpace(t.TrainerName));
            Assert.Equal(expectedMetricName, t.MetricName);
            Assert.True(t.ElapsedSeconds > 0, "a trial reported zero elapsed time");
        });
    }

    [Fact]
    public async Task Binary_classification_reports_every_completed_trial()
    {
        var lines = new List<string> { "f1,f2,label" };
        for (int i = 0; i < 150; i++)
        {
            var f1 = i % 37;
            var f2 = i % 17;
            lines.Add($"{f1},{f2},{(f1 + f2 > 25 ? 1 : 0)}");
        }
        var csv = await WriteCsvAsync("binary.csv", lines);

        var (result, progress) = await TrainAsync(csv, "binary-classification", "label", metric: "accuracy");

        Assert.NotNull(result.Model);
        AssertTrialsWereReported(progress, expectedMetricName: "accuracy");
    }

    [Fact]
    public async Task Multiclass_classification_reports_every_completed_trial()
    {
        var lines = new List<string> { "f1,f2,label" };
        for (int i = 0; i < 150; i++)
        {
            var f1 = i % 37;
            var f2 = i % 17;
            lines.Add($"{f1},{f2},class{(f1 + f2) % 3}");
        }
        var csv = await WriteCsvAsync("multiclass.csv", lines);

        // 30s, not the 10s the other two use: measured on this fixture, multiclass completes no
        // trial at all inside 10s and the run ends in NoSuccessfulTrialException. Verified against
        // the pre-change code, so it is a property of multiclass AutoML on a small budget rather
        // than anything this change introduced.
        var (result, progress) = await TrainAsync(
            csv, "multiclass-classification", "label", metric: "micro_accuracy", timeLimitSeconds: 30);

        Assert.NotNull(result.Model);
        AssertTrialsWereReported(progress, expectedMetricName: "micro_accuracy");
    }

    [Fact]
    public async Task Regression_reports_every_completed_trial()
    {
        var lines = new List<string> { "age,score,response" };
        for (int i = 0; i < 150; i++)
        {
            var age = 20 + (i % 50);
            var score = i % 30;
            lines.Add($"{age},{score},{age * 2.0 + score * 0.5}");
        }
        var csv = await WriteCsvAsync("regression.csv", lines);

        var (result, progress) = await TrainAsync(csv, "regression", "response", metric: "r_squared");

        Assert.NotNull(result.Model);
        AssertTrialsWereReported(progress, expectedMetricName: "r_squared");
    }

    [Fact]
    public async Task Trial_metric_name_follows_the_metric_the_experiment_optimizes()
    {
        // Not the task's default: the reported name must track --metric, because that is what
        // AutoML ranks trials by and therefore what the number on screen means.
        var lines = new List<string> { "age,score,response" };
        for (int i = 0; i < 150; i++)
        {
            var age = 20 + (i % 50);
            var score = i % 30;
            lines.Add($"{age},{score},{age * 2.0 + score * 0.5}");
        }
        var csv = await WriteCsvAsync("regression_rmse.csv", lines);

        var (_, progress) = await TrainAsync(csv, "regression", "response", metric: "rmse");

        Assert.NotEmpty(progress.Trials);
        Assert.All(progress.Trials, t => Assert.Equal("rmse", t.MetricName));
    }

    [Fact]
    public async Task Training_without_a_progress_listener_still_succeeds()
    {
        // The three paths now always pass a progressHandler; when nobody listens it is null, and
        // ML.NET has to accept that for the no-progress call shape to keep working.
        var lines = new List<string> { "age,score,response" };
        for (int i = 0; i < 200; i++)
        {
            var age = 20 + (i % 50);
            var score = i % 30;
            lines.Add($"{age},{score},{age * 2.0 + score * 0.5}");
        }
        var csv = await WriteCsvAsync("regression_noprogress.csv", lines);

        var ctx = new MLContext(seed: 42);
        var runner = new AutoMLRunner(ctx, new CsvDataLoader(ctx), _tmpDir);
        var config = new TrainingConfig
        {
            ModelName = "trialprogress",
            DataFile = csv,
            LabelColumn = "response",
            Task = "regression",
            TimeLimitSeconds = 10
        };

        var result = await runner.RunAsync(config, progress: null, CancellationToken.None);

        Assert.NotNull(result.Model);
    }
}
