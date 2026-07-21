using System.Diagnostics;
using MLoop.Core.Models;

namespace MLoop.Core.AutoML;

/// <summary>
/// The one place a trial becomes a <see cref="TrainingProgress"/>. Owns the two fields no call site
/// can supply honestly on its own — the trial ordinal and the elapsed-since-experiment-start clock —
/// so neither can be fabricated.
/// </summary>
/// <remarks>
/// <para>
/// Both fields were being hardcoded to zero before this type existed: the six non-AutoML task paths
/// and the BUG-36 manual fallback each built a <see cref="TrainingProgress"/> inline with
/// <c>ElapsedSeconds = 0</c>, and four of them also reported <em>before</em> fitting with
/// <c>Metric = 0</c>. Downstream that renders as <c>Trial 1: RandomizedPca - detection_rate=0.0000</c>
/// on a bar frozen at 0% — the percentage in the CLI is computed from
/// <see cref="TrainingProgress.ElapsedSeconds"/> — which is the same symptom cycle-177 removed from
/// the tabular paths, reintroduced as a fabricated value instead of a silence.
/// </para>
/// <para>
/// Hence the contract: <b>a trial is reported when it completes, carrying the metric it actually
/// produced.</b> A trial with no metric is not reported at all — there is nothing truthful to show,
/// and the terminal exception already carries the failure.
/// </para>
/// </remarks>
public sealed class TrialProgressChannel
{
    private readonly IProgress<TrainingProgress> _sink;
    private readonly Stopwatch _clock;
    private int _completedTrials;

    /// <param name="sink">MLoop's progress channel; every completed trial is forwarded here.</param>
    public TrialProgressChannel(IProgress<TrainingProgress> sink)
    {
        _sink = sink ?? throw new ArgumentNullException(nameof(sink));
        _clock = Stopwatch.StartNew();
    }

    /// <summary>Number of trials forwarded so far.</summary>
    public int CompletedTrials => Volatile.Read(ref _completedTrials);

    /// <summary>
    /// Forwards one completed trial. The ordinal is assigned here (AutoML reports from its own
    /// worker threads, so the increment is interlocked) and the elapsed time is read from this
    /// channel's clock — a caller cannot pass either one in.
    /// </summary>
    /// <param name="trainerName">Name of the trial's pipeline; blank names are labelled rather than shown empty.</param>
    /// <param name="metricName">Canonical MLoop metric name — the vocabulary of <c>AutoMLResult.Metrics</c>.</param>
    /// <param name="metric">The value that trial actually produced for <paramref name="metricName"/>.</param>
    public void ReportCompleted(string? trainerName, string metricName, double metric)
    {
        _sink.Report(new TrainingProgress
        {
            TrialNumber = Interlocked.Increment(ref _completedTrials),
            TrainerName = string.IsNullOrWhiteSpace(trainerName) ? "(unknown)" : trainerName,
            MetricName = metricName,
            Metric = metric,
            ElapsedSeconds = _clock.Elapsed.TotalSeconds
        });
    }
}
