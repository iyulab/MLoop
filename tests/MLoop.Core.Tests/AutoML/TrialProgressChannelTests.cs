using MLoop.Core.AutoML;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.AutoML;

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

        channel.ReportCompleted("KMeans (k=3)", "davies_bouldin_index", 0.7421);

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

        channel.ReportCompleted("A", "ndcg", 0.1);
        channel.ReportCompleted("B", "ndcg", 0.2);

        Assert.Equal([1, 2], reported.Select(p => p.TrialNumber));
        Assert.Equal(2, channel.CompletedTrials);
    }

    [Fact]
    public void ReportCompleted_carries_real_elapsed_time_never_zero()
    {
        var (channel, reported) = Build();

        channel.ReportCompleted("A", "mae", 12.5);

        Assert.True(Assert.Single(reported).ElapsedSeconds > 0, "the channel reported zero elapsed time");
    }

    [Fact]
    public void ReportCompleted_elapsed_is_cumulative_not_per_trial()
    {
        var (channel, reported) = Build();

        channel.ReportCompleted("A", "rmse", 1.0);
        Thread.Sleep(30);
        channel.ReportCompleted("B", "rmse", 0.9);

        Assert.True(reported[1].ElapsedSeconds > reported[0].ElapsedSeconds,
            $"elapsed did not advance: {reported[0].ElapsedSeconds} → {reported[1].ElapsedSeconds}");
    }

    [Fact]
    public void ReportCompleted_names_an_unnamed_trial_rather_than_reporting_a_blank()
    {
        var (channel, reported) = Build();

        channel.ReportCompleted(trainerName: null, "accuracy", 0.5);

        Assert.Equal("(unknown)", Assert.Single(reported).TrainerName);
    }
}
