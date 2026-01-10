using Microsoft.Data.Analysis;

namespace MLoop.Core.Preprocessing.Incremental.Contracts;

/// <summary>
/// Represents a sampling engine for progressive dataset sampling in incremental preprocessing.
/// </summary>
/// <remarks>
/// The sampling engine supports multiple strategies (stratified, random, adaptive) and enables
/// efficient processing of large datasets through progressive sampling stages.
/// </remarks>
public interface ISamplingEngine
{
    /// <summary>
    /// Samples a DataFrame at the specified ratio using the configured strategy.
    /// </summary>
    /// <param name="data">The source DataFrame to sample from.</param>
    /// <param name="sampleRatio">The ratio of data to sample (0.0 to 1.0).</param>
    /// <param name="config">Optional sampling configuration. If null, uses default configuration.</param>
    /// <param name="progress">Optional progress reporter for long-running operations.</param>
    /// <param name="cancellationToken">Cancellation token for async operation.</param>
    /// <returns>A sampled DataFrame.</returns>
    /// <remarks>
    /// <para>Sample ratios:</para>
    /// <list type="bullet">
    /// <item><description>0.001 = 0.1% (initial quick sample)</description></item>
    /// <item><description>0.01 = 1% (small sample)</description></item>
    /// <item><description>0.1 = 10% (medium sample)</description></item>
    /// <item><description>1.0 = 100% (full dataset)</description></item>
    /// </list>
    /// <para>For stratified sampling, the class distribution is preserved within ±2% tolerance.</para>
    /// </remarks>
    Task<DataFrame> SampleAsync(
        DataFrame data,
        double sampleRatio,
        Models.SamplingConfiguration? config = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the current sampling strategy being used.
    /// </summary>
    ISamplingStrategy Strategy { get; }

    /// <summary>
    /// Validates that the sample preserves key properties of the source data.
    /// </summary>
    /// <param name="source">The original DataFrame.</param>
    /// <param name="sample">The sampled DataFrame.</param>
    /// <param name="tolerance">Tolerance for distribution differences (default 0.02 = 2%).</param>
    /// <returns>True if sample is valid, false otherwise.</returns>
    /// <remarks>
    /// Validates:
    /// <list type="bullet">
    /// <item><description>Sample size matches expected ratio</description></item>
    /// <item><description>Class distribution preserved (for stratified sampling)</description></item>
    /// <item><description>No duplicate rows introduced</description></item>
    /// </list>
    /// </remarks>
    bool ValidateSample(DataFrame source, DataFrame sample, double tolerance = 0.02);
}
