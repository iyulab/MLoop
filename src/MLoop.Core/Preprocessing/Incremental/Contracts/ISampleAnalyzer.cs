using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;

namespace MLoop.Core.Preprocessing.Incremental.Contracts;

/// <summary>
/// Represents an analyzer for calculating comprehensive statistics on DataFrame samples.
/// </summary>
/// <remarks>
/// The sample analyzer provides detailed statistical analysis including:
/// <list type="bullet">
/// <item><description>Numeric statistics (mean, median, std dev, quartiles, skewness, kurtosis)</description></item>
/// <item><description>Categorical analysis (unique counts, frequencies, entropy)</description></item>
/// <item><description>Missing value assessment (counts, percentages, patterns)</description></item>
/// <item><description>Data quality metrics (outliers, distribution shape)</description></item>
/// </list>
/// </remarks>
public interface ISampleAnalyzer
{
    /// <summary>
    /// Analyzes a DataFrame sample and returns comprehensive statistics.
    /// </summary>
    /// <param name="sample">The DataFrame sample to analyze.</param>
    /// <param name="stageNumber">The sampling stage number (1-based) for tracking.</param>
    /// <param name="config">Optional analysis configuration.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>A comprehensive analysis of the sample.</returns>
    /// <remarks>
    /// <para>Analysis includes:</para>
    /// <list type="bullet">
    /// <item><description>Column-level statistics (per data type)</description></item>
    /// <item><description>Missing value analysis</description></item>
    /// <item><description>Distribution characteristics</description></item>
    /// <item><description>Data quality assessment</description></item>
    /// </list>
    /// <para>Performance: Target &lt;2s for 100K rows.</para>
    /// </remarks>
    Task<SampleAnalysis> AnalyzeAsync(
        DataFrame sample,
        int stageNumber,
        AnalysisConfiguration? config = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Analyzes a specific column in detail.
    /// </summary>
    /// <param name="column">The DataFrameColumn to analyze.</param>
    /// <param name="columnName">The name of the column.</param>
    /// <param name="columnIndex">The index of the column in the DataFrame.</param>
    /// <returns>Detailed column analysis.</returns>
    ColumnAnalysis AnalyzeColumn(DataFrameColumn column, string columnName, int columnIndex);

    /// <summary>
    /// Compares two sample analyses to detect convergence.
    /// </summary>
    /// <param name="previous">The previous stage analysis.</param>
    /// <param name="current">The current stage analysis.</param>
    /// <param name="threshold">Variance threshold for convergence (default 0.01 = 1%).</param>
    /// <returns>True if analyses have converged, false otherwise.</returns>
    /// <remarks>
    /// Convergence is detected when key statistics (mean, median, distribution)
    /// have variance less than the threshold between consecutive stages.
    /// This indicates that additional sampling is unlikely to change the results significantly.
    /// </remarks>
    bool HasConverged(SampleAnalysis previous, SampleAnalysis current, double threshold = 0.01);
}
