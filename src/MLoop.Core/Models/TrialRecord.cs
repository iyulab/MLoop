namespace MLoop.Core.Models;

/// <summary>
/// One completed AutoML trial, kept after the experiment ends so the leaderboard the search produced
/// survives it.
/// </summary>
/// <remarks>
/// ML.NET's <c>ExperimentResult.RunDetails</c> holds every trial's trainer, validation metrics and
/// runtime — and MLoop used to read <c>BestRun</c> off it and drop the rest, so each run threw away
/// the only record of what else was tried and how it scored. Trials that threw carry no metrics and
/// are not recorded: there is nothing to rank them by, and the terminal exception already reports
/// the failure.
/// </remarks>
public sealed class TrialRecord
{
    /// <summary>1-based position in the order AutoML completed the trials.</summary>
    public required int TrialNumber { get; init; }

    /// <summary>AutoML's name for the trial's pipeline, e.g. <c>…=&gt;FastTreeBinary</c>.</summary>
    public required string TrainerName { get; init; }

    /// <summary>
    /// The trial's validation metrics in MLoop's own vocabulary (the keys of
    /// <c>AutoMLResult.Metrics</c>), so a trial row and the final results table name the same things.
    /// Non-finite values are dropped rather than written — <c>NaN</c> has no JSON representation and
    /// would make the whole file unreadable.
    /// </summary>
    public required IReadOnlyDictionary<string, double> Metrics { get; init; }

    /// <summary>Seconds this one trial took (AutoML's <c>RunDetail.RuntimeInSeconds</c>).</summary>
    public required double RuntimeSeconds { get; init; }
}
