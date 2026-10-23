using MLoop.Core.Models;
using MLoop.Search;

namespace MLoop.Core.Training;

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
    /// Whether a trial is recorded: it produced metrics, or it failed and says why. A trial with
    /// neither — AutoML's live stream never shows one that threw — has nothing to record. One rule for
    /// the live channel and the after-the-fact records is what keeps the event stream,
    /// <c>phase(complete).trials</c> and <c>trials.ndjson</c> counting the same things.
    /// </summary>
    public static bool IsReportable(TrialResult trial) => trial.Succeeded || trial.Error is not null;

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

    /// <summary>The trainer of a search's trial, as MLoop records trainers.</summary>
    public static TrainerDescriptor Descriptor(TrialResult trial) => new()
    {
        Name = trial.Trainer,
        Parameters = trial.Parameters
    };

    /// <summary>
    /// A search's trial as the record kept with the experiment, numbered <paramref name="number"/>.
    /// A scored trial keeps its finite metrics (<see cref="OnlyFinite"/>) and, when they are a mean over
    /// folds, their spread; a failed one keeps its error and no metrics.
    /// </summary>
    public static TrialRecord ToRecord(TrialResult trial, int number)
    {
        if (!IsReportable(trial))
            throw new ArgumentException("A trial with neither metrics nor an error has no record.", nameof(trial));
        return new TrialRecord
        {
            TrialNumber = number,
            Trainer = Descriptor(trial),
            Metrics = trial.Metrics is { } metrics ? OnlyFinite(metrics) : new Dictionary<string, double>(),
            RuntimeSeconds = trial.RuntimeSeconds ?? 0,
            Validation = trial is { Folds: { } folds, StandardDeviation: { } spread }
                ? new TrialValidation(folds, OnlyFinite(spread))
                : null,
            Error = trial.Metrics is null ? trial.Error : null,
        };
    }
}
