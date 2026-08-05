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
    private readonly IProgress<TrainingProgress>? _sink;
    private readonly Stopwatch _clock;
    private readonly Lock _gate = new();
    private readonly List<TrialRecord> _records = [];
    private TimeSpan _previousCompletion;
    private string? _rankingMetric;
    private int _completedTrials;

    /// <param name="sink">
    /// MLoop's progress channel; every completed trial is forwarded here. Optional — a run with
    /// nobody watching still has to record what it tried, so the channel keeps its ledger either
    /// way and only the forwarding is skipped.
    /// </param>
    public TrialProgressChannel(IProgress<TrainingProgress>? sink)
    {
        _sink = sink;
        _clock = Stopwatch.StartNew();
    }

    /// <summary>Number of trials forwarded so far.</summary>
    public int CompletedTrials => Volatile.Read(ref _completedTrials);

    /// <summary>
    /// The trials this channel reported, as records — the persistable form of the same stream.
    /// </summary>
    /// <remarks>
    /// Task paths that run their own search (clustering over K, anomaly over rank, ts-anomaly over
    /// detectors, ranking over trainers) and the manual fallback have no AutoML <c>RunDetails</c> to
    /// harvest afterwards, so their leaderboard exists only here. Taking it from the channel rather
    /// than rebuilding it at each call site is what keeps "reported" and "recorded" the same set:
    /// one <c>ReportCompleted</c> call produces both, and neither can be added without the other.
    /// The AutoML paths ignore this list and read <c>RunDetails</c> instead — that source carries
    /// each trial's full metric set and its own runtime, which the live stream does not see — and
    /// the two productions are held together by <see cref="TrialLedger.IsReportable"/>.
    /// </remarks>
    public IReadOnlyList<TrialRecord> Records
    {
        get { lock (_gate) return [.. _records]; }
    }

    /// <summary>
    /// The metric these trials were reported against — what a leaderboard over
    /// <see cref="Records"/> ranks by. <c>null</c> when nothing was reported.
    /// </summary>
    /// <remarks>
    /// Read from the reports rather than restated by the call site. Naming it twice per path — once
    /// in <see cref="ReportCompleted"/>, once beside <c>Trials</c> — is two literals that have to
    /// agree, and when they do not the leaderboard ranks by a metric its own rows do not carry.
    /// </remarks>
    public string? RankingMetric
    {
        get { lock (_gate) return _rankingMetric; }
    }

    /// <summary>
    /// Forwards one completed trial and records it. The ordinal is assigned here (AutoML reports
    /// from its own worker threads, so the increment is interlocked) and both clocks are read from
    /// this channel — a caller cannot pass either one in.
    /// </summary>
    /// <param name="trainer">
    /// What the trial trained, in parts. A descriptor rather than a name so the record can be read
    /// by trainer and hyperparameter without parsing — the searches here identify candidates by
    /// hyperparameter (<c>k</c>, <c>rank</c>), which is exactly what a folded name loses.
    /// </param>
    /// <param name="metricName">Canonical MLoop metric name — the vocabulary of <c>AutoMLResult.Metrics</c>.</param>
    /// <param name="metric">The value that trial actually produced for <paramref name="metricName"/>.</param>
    /// <param name="allMetrics">
    /// Everything that trial measured, when the call site has it. Omitted, the record carries the
    /// one reported metric — which is the truth available, not a padded-out metric set.
    /// </param>
    public void ReportCompleted(
        TrainerDescriptor trainer,
        string metricName,
        double metric,
        IReadOnlyDictionary<string, double>? allMetrics = null)
    {
        var ordinal = Interlocked.Increment(ref _completedTrials);
        var elapsed = _clock.Elapsed;

        lock (_gate)
        {
            // A hand-rolled search runs its candidates one after another, so the gap since the
            // previous completion is that candidate's duration. AutoML's own runtime is more exact
            // but only its RunDetails knows it, and those paths do not read this list.
            var runtime = elapsed - _previousCompletion;
            _previousCompletion = elapsed;
            _rankingMetric = metricName;

            _records.Add(new TrialRecord
            {
                TrialNumber = ordinal,
                Trainer = trainer,
                Metrics = TrialLedger.OnlyFinite(allMetrics ?? new Dictionary<string, double> { [metricName] = metric }),
                RuntimeSeconds = runtime.TotalSeconds
            });
        }

        _sink?.Report(new TrainingProgress
        {
            TrialNumber = ordinal,
            TrainerName = trainer.Display,
            MetricName = metricName,
            Metric = metric,
            ElapsedSeconds = elapsed.TotalSeconds
        });
    }
}
