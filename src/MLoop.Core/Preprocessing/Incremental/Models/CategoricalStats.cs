namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Statistical metrics for categorical columns.
/// </summary>
public sealed class CategoricalStats
{
    /// <summary>
    /// Gets the count of non-null values.
    /// </summary>
    public required long Count { get; init; }

    /// <summary>
    /// Gets the number of unique values.
    /// </summary>
    public required int UniqueCount { get; init; }

    /// <summary>
    /// Gets the cardinality ratio (unique count / total count).
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    /// <item><description>Ratio ≈ 1.0: High cardinality (many unique values, e.g., IDs)</description></item>
    /// <item><description>Ratio ≈ 0.0: Low cardinality (few unique values, e.g., categories)</description></item>
    /// </list>
    /// </remarks>
    public double CardinalityRatio => Count > 0 ? (double)UniqueCount / Count : 0.0;

    /// <summary>
    /// Gets the top N most frequent values with their counts.
    /// </summary>
    /// <remarks>
    /// Sorted by frequency descending. Limited to MaxCategoricalValues from configuration.
    /// </remarks>
    public required List<(string Value, long Frequency)> TopValues { get; init; }

    /// <summary>
    /// Gets the most frequent value (mode).
    /// </summary>
    public string? Mode => TopValues.Count > 0 ? TopValues[0].Value : null;

    /// <summary>
    /// Gets the mode frequency.
    /// </summary>
    public long ModeFrequency => TopValues.Count > 0 ? TopValues[0].Frequency : 0;

    /// <summary>
    /// Gets the mode percentage.
    /// </summary>
    public double ModePercentage => Count > 0 ? (double)ModeFrequency / Count * 100.0 : 0.0;

    /// <summary>
    /// Gets the Shannon entropy (measure of information/randomness).
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    /// <item><description>Entropy = 0: All values are the same (no information)</description></item>
    /// <item><description>Entropy = log2(n): Uniform distribution (maximum information)</description></item>
    /// </list>
    /// <para>Formula: H = -Σ(p(x) * log2(p(x)))</para>
    /// </remarks>
    public required double Entropy { get; init; }

    /// <summary>
    /// Gets whether this column appears to be a unique identifier.
    /// </summary>
    /// <remarks>
    /// Heuristic: UniqueCount ≈ Count (cardinality ratio > 0.95)
    /// </remarks>
    public bool IsLikelyIdentifier => CardinalityRatio > 0.95;

    /// <summary>
    /// Gets whether this column has low cardinality (good for one-hot encoding).
    /// </summary>
    /// <remarks>
    /// Heuristic: UniqueCount &lt; 20 and cardinality ratio &lt; 0.1
    /// </remarks>
    public bool IsLowCardinality => UniqueCount < 20 && CardinalityRatio < 0.1;

    /// <summary>
    /// Gets whether this column has high cardinality (consider target encoding or embedding).
    /// </summary>
    /// <remarks>
    /// Heuristic: UniqueCount > 100 or cardinality ratio > 0.5
    /// </remarks>
    public bool IsHighCardinality => UniqueCount > 100 || CardinalityRatio > 0.5;
}
