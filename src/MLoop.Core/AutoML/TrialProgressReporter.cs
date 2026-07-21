using Microsoft.ML.AutoML;
using MLoop.Core.Models;

namespace MLoop.Core.AutoML;

/// <summary>
/// Adapts ML.NET AutoML's per-trial <see cref="RunDetail{TMetrics}"/> stream onto MLoop's
/// <see cref="TrainingProgress"/> channel.
/// </summary>
/// <remarks>
/// <para>
/// AutoML reports a trial the moment it finishes; it does <em>not</em> report trials that threw
/// (measured: a run where every trial failed produced zero reports). So a silent progress bar means
/// "nothing has completed yet", not "the handler is broken" — that distinction is exactly what
/// <see cref="NoSuccessfulTrialException"/> reports at the end.
/// </para>
/// <para>
/// Two fields cannot be taken from <see cref="RunDetail{TMetrics}"/> directly, and both are owned by
/// <see cref="TrialProgressChannel"/> instead: it carries no trial number, so the ordinal is counted
/// there; and its <c>RuntimeInSeconds</c> is that one trial's duration, whereas
/// <see cref="TrainingProgress.ElapsedSeconds"/> is read as elapsed-since-start by the
/// percentage calculation in the CLI's progress bar. Reporting per-trial runtime there would make
/// the bar jump backwards on every fast trial.
/// </para>
/// </remarks>
public sealed class TrialProgressReporter<TMetrics> : IProgress<RunDetail<TMetrics>>
    where TMetrics : class
{
    private readonly TrialProgressChannel _channel;
    private readonly string _metricName;
    private readonly Func<TMetrics, double> _selectMetric;

    /// <param name="sink">MLoop's progress channel; every completed trial is forwarded here.</param>
    /// <param name="metricName">
    /// Canonical name of the metric AutoML is optimizing — see <c>AutoMLRunner.Describe*Metric</c>.
    /// </param>
    /// <param name="selectMetric">Reads that metric off a trial's validation metrics.</param>
    public TrialProgressReporter(
        IProgress<TrainingProgress> sink,
        string metricName,
        Func<TMetrics, double> selectMetric)
    {
        _channel = new TrialProgressChannel(sink);
        _metricName = metricName ?? throw new ArgumentNullException(nameof(metricName));
        _selectMetric = selectMetric ?? throw new ArgumentNullException(nameof(selectMetric));
    }

    /// <summary>Number of trials forwarded so far.</summary>
    public int CompletedTrials => _channel.CompletedTrials;

    public void Report(RunDetail<TMetrics> value)
        => ReportTrial(value?.TrainerName, value?.ValidationMetrics);

    /// <summary>
    /// The reporter's actual operation, split out from <see cref="Report"/> because
    /// <see cref="RunDetail{TMetrics}"/> has only an internal constructor — it is a transport, not
    /// something a test can build.
    /// </summary>
    /// <param name="trainerName">AutoML's name for the trial's pipeline.</param>
    /// <param name="metrics">
    /// The trial's validation metrics, or <c>null</c> if it did not produce any (a trial that threw).
    /// Nothing is reported in that case: there is no result to show, and the terminal exception
    /// already carries the failure.
    /// </param>
    public void ReportTrial(string? trainerName, TMetrics? metrics)
    {
        if (metrics is null)
            return;

        _channel.ReportCompleted(trainerName, _metricName, _selectMetric(metrics));
    }
}
