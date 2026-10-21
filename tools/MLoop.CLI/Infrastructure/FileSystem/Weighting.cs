namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// How an experiment's rows were counted: by a weight column, or each once. Two experiments' metrics compare
/// only under the same weighting — metrics of one name over different weightings describe different
/// distributions. The one place that sameness is judged: ranking (<see cref="ExperimentRanking"/>),
/// <c>compare</c>, and automatic promotion all ask it here, so they cannot disagree on case or on
/// "absent" versus "empty".
/// </summary>
public sealed record Weighting
{
    private Weighting(string? column) => Column = column;

    /// <summary>The weight column as recorded; <c>null</c> when the rows were not weighted.</summary>
    public string? Column { get; }

    /// <summary>The weighting of an experiment that recorded <paramref name="column"/> (empty means none).</summary>
    public static Weighting Of(string? column) => new(string.IsNullOrWhiteSpace(column) ? null : column.Trim());

    /// <summary>Same weighting: both unweighted, or weighted by one column — compared without case, as columns are matched.</summary>
    public bool Equals(Weighting? other)
        => other is not null && string.Equals(Column, other.Column, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc />
    public override int GetHashCode() => StringComparer.OrdinalIgnoreCase.GetHashCode(Column ?? string.Empty);

    /// <summary>"unweighted" or "weighted by 'w'".</summary>
    public override string ToString() => Column is null ? "unweighted" : $"weighted by '{Column}'";

    /// <summary>True when <paramref name="weightings"/> are not all the same.</summary>
    public static bool Differ(IEnumerable<Weighting> weightings) => weightings.Distinct().Skip(1).Any();
}
