using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;

/// <summary>
/// Interface for pattern detection algorithms.
/// Each detector focuses on a specific type of data quality pattern.
/// </summary>
public interface IPatternDetector
{
    /// <summary>
    /// Type of pattern this detector focuses on.
    /// </summary>
    PatternType PatternType { get; }

    /// <summary>
    /// Detect patterns in a single column.
    /// </summary>
    /// <param name="column">DataFrame column to analyze.</param>
    /// <param name="columnName">Name of the column.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of detected patterns (can be empty if none found).</returns>
    Task<IReadOnlyList<DetectedPattern>> DetectAsync(
        DataFrameColumn column,
        string columnName,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Whether this detector should run on the given column.
    /// Example: OutlierDetector only runs on numeric columns.
    /// </summary>
    /// <param name="column">DataFrame column to check.</param>
    /// <returns>True if detector should run on this column.</returns>
    bool IsApplicable(DataFrameColumn column);
}
