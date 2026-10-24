using System.Text.Json.Serialization;

namespace MLoop.Core.Models;

/// <summary>
/// One completed AutoML trial, kept after the experiment ends so the leaderboard the search produced
/// survives it.
/// </summary>
/// <remarks>
/// ML.NET's <c>ExperimentResult.RunDetails</c> holds every trial's trainer, validation metrics and
/// runtime — and MLoop used to read <c>BestRun</c> off it and drop the rest, so each run threw away
/// the only record of what else was tried and how it scored. A trial that failed is recorded when the
/// search reports it (MLoop's own search does; AutoML does not), with <see cref="Error"/> and no
/// metrics — it ranks below every trial that scored.
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
    /// <c>TrainedModel.Metrics</c>), so a trial row and the final results table name the same things.
    /// Non-finite values are dropped rather than written — <c>NaN</c> has no JSON representation and
    /// would make the whole file unreadable.
    /// </summary>
    public required IReadOnlyDictionary<string, double> Metrics { get; init; }

    /// <summary>Seconds this one trial took (AutoML's <c>RunDetail.RuntimeInSeconds</c>).</summary>
    public required double RuntimeSeconds { get; init; }

    /// <summary>
    /// How the trial's metrics were validated, when they are a mean over folds: the fold count and each
    /// metric's spread. Absent on paths that score one split.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public TrialValidation? Validation { get; init; }

    /// <summary>Why the trial failed; absent when it scored. A failed trial has no <see cref="Metrics"/>.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Error { get; init; }

    /// <summary>
    /// True on the trial the experiment's model was refit from, when that is not simply the highest-scoring one
    /// — MLoop's search keeps the smallest model within one standard error of the top score. Absent elsewhere.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public bool Selected { get; init; }

    /// <summary>This record, marked <see cref="Selected"/>.</summary>
    public TrialRecord AsSelected() => new()
    {
        TrialNumber = TrialNumber,
        Trainer = Trainer,
        Metrics = Metrics,
        RuntimeSeconds = RuntimeSeconds,
        Validation = Validation,
        Error = Error,
        Selected = true,
    };
}

/// <summary>
/// A trial's validation: how many folds, each metric's standard deviation across them, and each metric's standard
/// error — the folds' spread over √folds, or for a holdout (one fold, no spread) the spread of its slices. The
/// standard error is what tells two trials' scores apart: the search keeps a smaller model within one of it.
/// </summary>
public sealed record TrialValidation(
    int Folds,
    IReadOnlyDictionary<string, double> StandardDeviation,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] IReadOnlyDictionary<string, double>? StandardError = null);
