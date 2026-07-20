using MLoop.Core.AutoML;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// Pins the RunDetail → TrainingProgress adaptation. The three tabular AutoML paths accepted an
/// <see cref="IProgress{TrainingProgress}"/> and never reported through it, so every consumer of
/// the progress channel (the CLI progress bar, the probe trial counter) sat at zero on the most
/// common tasks. These tests fix the contract that replaced that silence.
/// </summary>
public class TrialProgressReporterTests
{
    private sealed class FakeMetrics
    {
        public required double Value { get; init; }
    }

    private static (TrialProgressReporter<FakeMetrics> Reporter, List<TrainingProgress> Reported) Build(
        string metricName = "auc")
    {
        var reported = new List<TrainingProgress>();
        var sink = new SynchronousProgress(reported.Add);
        return (new TrialProgressReporter<FakeMetrics>(sink, metricName, m => m.Value), reported);
    }

    /// <summary>Reports inline, so assertions do not race the thread pool as Progress&lt;T&gt; would.</summary>
    private sealed class SynchronousProgress(Action<TrainingProgress> onReport) : IProgress<TrainingProgress>
    {
        public void Report(TrainingProgress value) => onReport(value);
    }

    [Fact]
    public void ReportTrial_forwards_trainer_metric_name_and_value()
    {
        var (reporter, reported) = Build(metricName: "f1_score");

        reporter.ReportTrial("ReplaceMissingValues=>Concatenate=>FastTreeBinary", new FakeMetrics { Value = 0.8123 });

        var progress = Assert.Single(reported);
        Assert.Equal("ReplaceMissingValues=>Concatenate=>FastTreeBinary", progress.TrainerName);
        Assert.Equal("f1_score", progress.MetricName);
        Assert.Equal(0.8123, progress.Metric);
    }

    [Fact]
    public void ReportTrial_numbers_trials_from_one_and_upward()
    {
        var (reporter, reported) = Build();

        reporter.ReportTrial("A", new FakeMetrics { Value = 0.1 });
        reporter.ReportTrial("B", new FakeMetrics { Value = 0.2 });
        reporter.ReportTrial("C", new FakeMetrics { Value = 0.3 });

        Assert.Equal([1, 2, 3], reported.Select(p => p.TrialNumber));
        Assert.Equal(3, reporter.CompletedTrials);
    }

    [Fact]
    public void ReportTrial_ignores_a_trial_that_produced_no_metrics()
    {
        var (reporter, reported) = Build();

        reporter.ReportTrial("Threw", metrics: null);

        Assert.Empty(reported);
        Assert.Equal(0, reporter.CompletedTrials);
    }

    [Fact]
    public void ReportTrial_elapsed_is_cumulative_not_per_trial()
    {
        // The CLI turns ElapsedSeconds into a percentage of the time budget. If it carried a single
        // trial's runtime the bar would jump backwards whenever a fast trial followed a slow one.
        var (reporter, reported) = Build();

        reporter.ReportTrial("A", new FakeMetrics { Value = 0.1 });
        Thread.Sleep(30);
        reporter.ReportTrial("B", new FakeMetrics { Value = 0.2 });

        Assert.True(reported[1].ElapsedSeconds >= reported[0].ElapsedSeconds,
            $"elapsed went backwards: {reported[0].ElapsedSeconds} → {reported[1].ElapsedSeconds}");
        Assert.True(reported[1].ElapsedSeconds > 0);
    }

    [Fact]
    public void ReportTrial_leaves_phase_unset_so_the_CLI_reads_it_as_a_trial_update()
    {
        // TrainCommand branches on Phase.HasValue: a non-null Phase is an auto-time marker and is
        // returned early, never reaching the trial display.
        var (reporter, reported) = Build();

        reporter.ReportTrial("A", new FakeMetrics { Value = 0.5 });

        Assert.Null(Assert.Single(reported).Phase);
    }

    [Fact]
    public void ReportTrial_counts_correctly_when_trials_arrive_concurrently()
    {
        // AutoML reports from its own worker threads; the trial ordinal must not lose increments.
        var reported = new System.Collections.Concurrent.ConcurrentBag<TrainingProgress>();
        var sink = new SynchronousProgress(reported.Add);
        var reporter = new TrialProgressReporter<FakeMetrics>(sink, "auc", m => m.Value);

        // Bounded degree on purpose: a race needs only a handful of threads to surface, while an
        // unbounded Parallel.For saturates every core — and xUnit runs other collections in this
        // assembly concurrently, including ones that train real AutoML models against a 10s budget.
        Parallel.For(0, 200, new ParallelOptions { MaxDegreeOfParallelism = 4 },
            i => reporter.ReportTrial($"T{i}", new FakeMetrics { Value = i }));

        Assert.Equal(200, reporter.CompletedTrials);
        Assert.Equal(Enumerable.Range(1, 200), reported.Select(p => p.TrialNumber).OrderBy(n => n));
    }

    [Fact]
    public void ReportTrial_names_an_unnamed_trial_rather_than_reporting_a_blank()
    {
        var (reporter, reported) = Build();

        reporter.ReportTrial(trainerName: null, new FakeMetrics { Value = 0.5 });

        Assert.Equal("(unknown)", Assert.Single(reported).TrainerName);
    }
}
