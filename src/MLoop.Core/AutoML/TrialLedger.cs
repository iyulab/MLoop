namespace MLoop.Core.AutoML;

/// <summary>
/// The rules that decide whether a trial counts — shared by the two places that produce trial
/// records, so they agree by construction rather than by coincidence.
/// </summary>
/// <remarks>
/// A run reports trials twice: live, through <see cref="TrialProgressChannel"/>, and after the fact,
/// from AutoML's <c>RunDetails</c>. Both were applying the same rule ("a trial with validation
/// metrics") written out separately, and a consumer noticed the one place where two counters
/// disagreed. The counts must be one fact; where the two productions cannot be merged (RunDetails
/// carries each trial's full metric set and its own runtime, which the live channel does not), the
/// rule that decides membership is single-sourced here instead.
/// </remarks>
public static class TrialLedger
{
    /// <summary>
    /// Whether a trial produced something worth recording. A trial that threw has no validation
    /// metrics: there is nothing to rank it by, and the terminal exception already reports the
    /// failure. Such a trial is neither reported live nor persisted — that is what keeps the event
    /// stream, <c>phase(complete).trials</c> and <c>trials.ndjson</c> counting the same things.
    /// </summary>
    public static bool IsReportable(object? validationMetrics) => validationMetrics is not null;

    /// <summary>
    /// Drops non-finite metric values and copies the rest. A trial that failed to produce a usable
    /// AUC reports <c>NaN</c>, which has no JSON representation — writing it would make the whole
    /// trials file unparseable, and coercing it to 0 would rank a broken trial as a perfect one.
    /// </summary>
    /// <remarks>
    /// The copy is load-bearing, not incidental: several task paths keep filling their metric
    /// dictionary after the trial is reported (forecasting appends <c>horizon</c> and
    /// <c>window_size</c>), so a stored reference would let a later mutation rewrite history.
    /// </remarks>
    public static Dictionary<string, double> OnlyFinite(IReadOnlyDictionary<string, double> metrics)
    {
        var finite = new Dictionary<string, double>(metrics.Count);
        foreach (var (key, value) in metrics)
        {
            if (double.IsFinite(value))
                finite[key] = value;
        }
        return finite;
    }
}
