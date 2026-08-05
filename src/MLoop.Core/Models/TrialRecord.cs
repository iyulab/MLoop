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

    /// <summary>
    /// What this trial trained, in parts — so a reader can group or compare trials by trainer
    /// without parsing <see cref="TrainerName"/>.
    /// </summary>
    /// <remarks>
    /// The hand-rolled searches identify their candidates by hyperparameter — <c>KMeans (k=3)</c>,
    /// <c>RandomizedPca (rank=20)</c> — so on those paths the trial's name is exactly the folded
    /// string that made <c>bestTrainer</c> unusable as an identifier. A leaderboard whose rows are
    /// "the same trainer at different k" is the case where telling them apart matters most.
    /// </remarks>
    public required TrainerDescriptor Trainer { get; init; }

    /// <summary>
    /// The trial's trainer as displayed, e.g. <c>…=&gt;FastTreeBinary</c> or <c>KMeans (k=3)</c>.
    /// Derived from <see cref="Trainer"/>, and still written to <c>trials.ndjson</c> — readers of
    /// that file keep the field they have.
    /// </summary>
    public string TrainerName => Trainer.Display;

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
