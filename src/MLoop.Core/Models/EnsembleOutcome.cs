namespace MLoop.Core.Models;

/// <summary>One member of a blend the run answered with: the trainer of its trial, how often the blend counts it, and its weight.</summary>
/// <param name="Trainer">The member's trainer, as the trial records name it.</param>
/// <param name="Count">How many times forward selection added the member.</param>
/// <param name="Weight">The member's share of the blend: its count over the sum of all counts.</param>
public sealed record EnsembleMemberRecord(string Trainer, int Count, double Weight);

/// <summary>
/// What a run that was asked for a blend (<see cref="TrainingConfig.Ensemble"/>) came to: the blend it answers with and how it
/// scored out of fold against the best single trial — or why the single best model stands instead.
/// </summary>
public sealed record EnsembleOutcome
{
    /// <summary>The blend's members, best trial first; <see langword="null"/> when the single model stands.</summary>
    public IReadOnlyList<EnsembleMemberRecord>? Members { get; init; }

    /// <summary>The blend's metric on the out-of-fold answers; <see langword="null"/> when none was chosen.</summary>
    public double? OutOfFoldScore { get; init; }

    /// <summary>The best retained trial's metric on the same answers; <see langword="null"/> when no blend was weighed.</summary>
    public double? BestSingleOutOfFoldScore { get; init; }

    /// <summary>Why the single best model stands, or <see langword="null"/> when the run answers with the blend.</summary>
    public string? SkippedBecause { get; init; }

    /// <summary>Whether the run answers with the blend.</summary>
    public bool IsBlend => Members is not null;
}
