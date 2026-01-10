using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;

namespace MLoop.Core.Preprocessing.Incremental.Contracts;

/// <summary>
/// Represents a strategy for sampling data from a DataFrame.
/// </summary>
/// <remarks>
/// <para>Available strategies:</para>
/// <list type="bullet">
/// <item><description><b>Stratified</b>: Preserves class distribution for classification tasks</description></item>
/// <item><description><b>Random</b>: Simple random sampling without stratification</description></item>
/// <item><description><b>Adaptive</b>: Dynamically adjusts strategy based on dataset characteristics</description></item>
/// </list>
/// <para>Custom strategies can be implemented by creating a class that implements this interface.</para>
/// </remarks>
public interface ISamplingStrategy
{
    /// <summary>
    /// Gets the name of the sampling strategy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Determines if this strategy is applicable to the given DataFrame and configuration.
    /// </summary>
    /// <param name="data">The DataFrame to check.</param>
    /// <param name="config">The sampling configuration.</param>
    /// <returns>True if this strategy is applicable, false otherwise.</returns>
    /// <remarks>
    /// <para>Applicability examples:</para>
    /// <list type="bullet">
    /// <item><description>Stratified: Requires a label column with categorical values</description></item>
    /// <item><description>Random: Always applicable</description></item>
    /// <item><description>Adaptive: Checks dataset size and label distribution</description></item>
    /// </list>
    /// </remarks>
    bool IsApplicable(DataFrame data, Models.SamplingConfiguration config);

    /// <summary>
    /// Samples the DataFrame at the specified ratio.
    /// </summary>
    /// <param name="data">The source DataFrame to sample.</param>
    /// <param name="sampleRatio">The ratio of data to sample (0.0 to 1.0).</param>
    /// <param name="config">The sampling configuration.</param>
    /// <param name="randomSeed">Random seed for reproducibility (default 42).</param>
    /// <returns>A sampled DataFrame.</returns>
    /// <remarks>
    /// <para>Implementation requirements:</para>
    /// <list type="bullet">
    /// <item><description>Must be deterministic with the same random seed</description></item>
    /// <item><description>Must preserve original row indices (for traceability)</description></item>
    /// <item><description>Must handle edge cases (empty data, single row, all same value)</description></item>
    /// <item><description>Must be memory efficient (&lt;500MB for 1M rows)</description></item>
    /// </list>
    /// </remarks>
    DataFrame Sample(DataFrame data, double sampleRatio, Models.SamplingConfiguration config, int randomSeed = 42);

    /// <summary>
    /// Validates the sampled DataFrame against quality criteria.
    /// </summary>
    /// <param name="source">The original DataFrame.</param>
    /// <param name="sample">The sampled DataFrame.</param>
    /// <param name="config">The sampling configuration.</param>
    /// <returns>Validation result with details.</returns>
    /// <remarks>
    /// Validates strategy-specific criteria. For example:
    /// <list type="bullet">
    /// <item><description>Stratified: Class distribution preserved within tolerance</description></item>
    /// <item><description>Random: Sample size correct, no duplicates</description></item>
    /// </list>
    /// </remarks>
    ValidationResult Validate(DataFrame source, DataFrame sample, Models.SamplingConfiguration config);
}
