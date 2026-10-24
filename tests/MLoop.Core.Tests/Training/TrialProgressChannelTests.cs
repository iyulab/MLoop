using MLoop.Core.Models;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// Pins the two fields the channel owns so that no call site can supply them: the trial ordinal and
/// the elapsed-since-start clock. Every non-AutoML task path used to build its own
/// <see cref="TrainingProgress"/> with both hardcoded to zero — which froze the CLI progress bar,
/// because its percentage is computed from <see cref="TrainingProgress.ElapsedSeconds"/>.
/// </summary>
public class TrialProgressChannelTests
{
    /// <summary>Reports inline, so assertions do not race the thread pool as Progress&lt;T&gt; would.</summary>
    private sealed class SynchronousProgress(Action<TrainingProgress> onReport) : IProgress<TrainingProgress>
    {
        public void Report(TrainingProgress value) => onReport(value);
    }

    private static (TrialProgressChannel Channel, List<TrainingProgress> Reported) Build()
    {
        var reported = new List<TrainingProgress>();
        return (new TrialProgressChannel(new SynchronousProgress(reported.Add)), reported);
    }

    [Fact]
    public void ReportCompleted_forwards_trainer_metric_name_and_value()
    {
        var (channel, reported) = Build();

        channel.ReportCompleted(TrainerDescriptor.Of("KMeans (k=3)"), "davies_bouldin_index", 0.7421);

        var progress = Assert.Single(reported);
        Assert.Equal("KMeans (k=3)", progress.TrainerName);
        Assert.Equal("davies_bouldin_index", progress.MetricName);
        Assert.Equal(0.7421, progress.Metric);
        Assert.Null(progress.Phase);
    }

    [Fact]
    public void ReportCompleted_numbers_trials_from_one_and_upward()
    {
        var (channel, reported) = Build();

        channel.ReportCompleted(TrainerDescriptor.Of("A"), "ndcg", 0.1);
        channel.ReportCompleted(TrainerDescriptor.Of("B"), "ndcg", 0.2);

        Assert.Equal([1, 2], reported.Select(p => p.TrialNumber));
        Assert.Equal(2, channel.CompletedTrials);
    }

    [Fact]
    public void ReportCompleted_carries_real_elapsed_time_never_zero()
    {
        var (channel, reported) = Build();
        // Let the clock move: reported straight after construction, a real clock can still read zero (Apple
        // silicon ticks at ~42 ns) — what this guards against is a hardcoded zero, not a fast machine.
        Thread.Sleep(5);

        channel.ReportCompleted(TrainerDescriptor.Of("A"), "mae", 12.5);

        Assert.True(Assert.Single(reported).ElapsedSeconds > 0, "the channel reported zero elapsed time");
    }

    [Fact]
    public void ReportCompleted_elapsed_is_cumulative_not_per_trial()
    {
        var (channel, reported) = Build();

        channel.ReportCompleted(TrainerDescriptor.Of("A"), "rmse", 1.0);
        Thread.Sleep(30);
        channel.ReportCompleted(TrainerDescriptor.Of("B"), "rmse", 0.9);

        Assert.True(reported[1].ElapsedSeconds > reported[0].ElapsedSeconds,
            $"elapsed did not advance: {reported[0].ElapsedSeconds} → {reported[1].ElapsedSeconds}");
    }

    /// <summary>
    /// The blank-name guard lives on <see cref="TrainerDescriptor"/> now, so it holds for every
    /// path into the channel rather than for the one that remembered to check.
    /// </summary>
    [Fact]
    public void ReportCompleted_names_an_unnamed_trial_rather_than_reporting_a_blank()
    {
        var (channel, reported) = Build();

        channel.ReportCompleted(TrainerDescriptor.Of(null!), "accuracy", 0.5);

        Assert.Equal("(unknown)", Assert.Single(reported).TrainerName);
    }

    /// <summary>
    /// The record the channel keeps and the progress it forwards describe one trial — the record
    /// carries the trainer in parts, the progress line its rendering.
    /// </summary>
    [Fact]
    public void ReportCompleted_records_the_trainer_in_parts()
    {
        var (channel, reported) = Build();

        channel.ReportCompleted(
            TrainerDescriptor.Of("KMeans", ("k", 3)), "davies_bouldin_index", 0.7421);

        var record = Assert.Single(channel.Records);
        Assert.Equal("KMeans", record.Trainer.Name);
        Assert.Equal("3", record.Trainer.Params!["k"]);
        Assert.Equal("KMeans (k=3)", record.TrainerName);
        Assert.Equal("KMeans (k=3)", Assert.Single(reported).TrainerName);
    }

    /// <summary>
    /// A search reports its failed trials. They are trials like any other — counted, numbered,
    /// recorded — with the error and no metric, so the trials file and the event stream agree.
    /// </summary>
    [Fact]
    public void A_search_trial_that_failed_is_recorded_and_reported_with_its_error()
    {
        var (channel, reported) = Build();
        MLoop.Search.ITrialSink sink = channel;

        sink.Completed(new MLoop.Search.TrialResult
        {
            Trainer = "LightGbm",
            RuntimeSeconds = 0.25,
            Error = "lib_lightgbm not found",
        }, "auc");

        var record = Assert.Single(channel.Records);
        Assert.Equal("lib_lightgbm not found", record.Error);
        Assert.Empty(record.Metrics);
        Assert.Equal(0.25, record.RuntimeSeconds);
        var progress = Assert.Single(reported);
        Assert.Equal("lib_lightgbm not found", progress.Error);
        Assert.True(double.IsNaN(progress.Metric));
        Assert.Equal(1, channel.CompletedTrials);
    }

    /// <summary>A search trial's record keeps its own runtime and its spread over folds.</summary>
    [Fact]
    public void A_search_trial_keeps_its_runtime_and_validation()
    {
        var (channel, _) = Build();
        MLoop.Search.ITrialSink sink = channel;

        sink.Completed(new MLoop.Search.TrialResult
        {
            Trainer = "FastTree",
            Metrics = new Dictionary<string, double> { ["auc"] = 0.9 },
            Folds = 5,
            StandardDeviation = new Dictionary<string, double> { ["auc"] = 0.02, ["auprc"] = double.NaN },
            RuntimeSeconds = 1.5,
        }, "auc");

        var record = Assert.Single(channel.Records);
        Assert.Null(record.Error);
        Assert.Equal(1.5, record.RuntimeSeconds);
        Assert.Equal(5, record.Validation!.Folds);
        Assert.Equal(new Dictionary<string, double> { ["auc"] = 0.02 }, record.Validation.StandardDeviation);
    }
}
