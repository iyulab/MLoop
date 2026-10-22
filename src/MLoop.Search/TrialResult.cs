namespace MLoop.Search;

/// <summary>
/// One trial of a search: the trainer and settings it tried and what came of it.
/// </summary>
public sealed record TrialResult
{
    /// <summary>The trainer's name, as a person reads it (<c>FastTreeBinary</c>).</summary>
    public required string Trainer { get; init; }

    /// <summary>The settings the trial chose, in the order they were set; null when it chose none.</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Parameters { get; init; }

    /// <summary>The trial's validation metrics by name; null when it produced none.</summary>
    public IReadOnlyDictionary<string, double>? Metrics { get; init; }

    /// <summary>How many folds <see cref="Metrics"/> is the mean of; null when the trial was not validated over folds.</summary>
    public int? Folds { get; init; }

    /// <summary>Each metric's standard deviation over the <see cref="Folds"/>; null with them.</summary>
    public IReadOnlyDictionary<string, double>? StandardDeviation { get; init; }

    /// <summary>How long the trial ran, when that is known.</summary>
    public double? RuntimeSeconds { get; init; }

    /// <summary>Why the trial failed, when it did.</summary>
    public string? Error { get; init; }

    /// <summary>The trial produced metrics to rank it by.</summary>
    public bool Succeeded => Metrics is not null;
}

/// <summary>Where a search reports each trial as it finishes.</summary>
public interface ITrialSink
{
    /// <param name="trial">The finished trial.</param>
    /// <param name="rankingMetric">The metric the search ranks trials by — a key of the trial's metrics.</param>
    void Completed(TrialResult trial, string rankingMetric);
}
