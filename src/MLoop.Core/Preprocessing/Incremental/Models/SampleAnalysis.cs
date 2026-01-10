namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Comprehensive analysis of a DataFrame sample.
/// </summary>
public sealed class SampleAnalysis
{
    /// <summary>
    /// Gets the sampling stage number (1-based).
    /// </summary>
    public required int StageNumber { get; init; }

    /// <summary>
    /// Gets the sample ratio used (0.0 to 1.0).
    /// </summary>
    public required double SampleRatio { get; init; }

    /// <summary>
    /// Gets the timestamp when this analysis was performed.
    /// </summary>
    public required DateTime Timestamp { get; init; }

    /// <summary>
    /// Gets the total number of rows in the sample.
    /// </summary>
    public required long RowCount { get; init; }

    /// <summary>
    /// Gets the total number of columns in the sample.
    /// </summary>
    public required int ColumnCount { get; init; }

    /// <summary>
    /// Gets the per-column analyses.
    /// </summary>
    public required List<ColumnAnalysis> Columns { get; init; }

    /// <summary>
    /// Gets the overall data quality score (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// Composite score based on:
    /// <list type="bullet">
    /// <item><description>Missing value percentage</description></item>
    /// <item><description>Number of quality issues</description></item>
    /// <item><description>Data type consistency</description></item>
    /// </list>
    /// <para>1.0 = Perfect quality, 0.0 = Severe issues</para>
    /// </remarks>
    public required double QualityScore { get; init; }

    /// <summary>
    /// Gets all data quality issues across all columns.
    /// </summary>
    public List<DataQualityIssue> AllQualityIssues
    {
        get
        {
            return Columns
                .SelectMany(c => c.QualityIssues)
                .OrderByDescending(i => i.Severity)
                .ToList();
        }
    }

    /// <summary>
    /// Gets the count of numeric columns.
    /// </summary>
    public int NumericColumnCount => Columns.Count(c => c.IsNumeric);

    /// <summary>
    /// Gets the count of categorical columns.
    /// </summary>
    public int CategoricalColumnCount => Columns.Count(c => c.IsCategorical);

    /// <summary>
    /// Gets the count of temporal columns.
    /// </summary>
    public int TemporalColumnCount => Columns.Count(c => c.IsTemporal);

    /// <summary>
    /// Gets the count of columns with missing values.
    /// </summary>
    public int ColumnsWithMissingCount => Columns.Count(c => c.NullCount > 0);

    /// <summary>
    /// Gets the overall missing value percentage across all columns.
    /// </summary>
    public double OverallMissingPercentage
    {
        get
        {
            if (ColumnCount == 0) return 0.0;
            long totalCells = RowCount * ColumnCount;
            long totalMissing = Columns.Sum(c => c.NullCount);
            return totalCells > 0 ? (double)totalMissing / totalCells * 100.0 : 0.0;
        }
    }

    /// <summary>
    /// Gets the memory footprint estimate in bytes.
    /// </summary>
    /// <remarks>
    /// Rough estimate based on row/column count and data types.
    /// </remarks>
    public long EstimatedMemoryBytes { get; init; }

    /// <summary>
    /// Gets the memory footprint estimate in megabytes.
    /// </summary>
    public double EstimatedMemoryMB => EstimatedMemoryBytes / (1024.0 * 1024.0);

    /// <summary>
    /// Gets additional analysis metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; init; } = new();

    /// <summary>
    /// Gets a summary string for logging.
    /// </summary>
    /// <returns>Human-readable summary.</returns>
    public string GetSummary()
    {
        return $"Stage {StageNumber} ({SampleRatio:P}): " +
               $"{RowCount:N0} rows, {ColumnCount} columns, " +
               $"Quality {QualityScore:P0}, " +
               $"{ColumnsWithMissingCount} cols with missing values, " +
               $"{AllQualityIssues.Count} issues";
    }
}
