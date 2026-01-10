namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

/// <summary>
/// Represents a data quality pattern detected in a sample.
/// </summary>
public sealed class DetectedPattern
{
    /// <summary>
    /// Type of pattern detected.
    /// </summary>
    public required PatternType Type { get; init; }

    /// <summary>
    /// Column name where pattern was detected.
    /// </summary>
    public required string ColumnName { get; init; }

    /// <summary>
    /// Human-readable description of the pattern.
    /// Example: "15.2% missing values (NULL, N/A, empty strings)"
    /// </summary>
    public required string Description { get; init; }

    /// <summary>
    /// Severity level of the pattern.
    /// </summary>
    public required Severity Severity { get; init; }

    /// <summary>
    /// Number of occurrences of this pattern in the sample.
    /// </summary>
    public required int Occurrences { get; init; }

    /// <summary>
    /// Total number of rows analyzed.
    /// </summary>
    public required int TotalRows { get; init; }

    /// <summary>
    /// Confidence that this pattern is real (0-1).
    /// 1.0 = Deterministic detection
    /// </summary>
    public required double Confidence { get; init; }

    /// <summary>
    /// Percentage of rows affected (0-1).
    /// </summary>
    public double AffectedPercentage => (double)Occurrences / TotalRows;

    /// <summary>
    /// Example values demonstrating the pattern (optional).
    /// Useful for HITL context.
    /// </summary>
    public IReadOnlyList<string>? Examples { get; init; }

    /// <summary>
    /// Suggested fix or action (optional).
    /// Example: "Convert to ISO-8601 format"
    /// </summary>
    public string? SuggestedFix { get; init; }
}
