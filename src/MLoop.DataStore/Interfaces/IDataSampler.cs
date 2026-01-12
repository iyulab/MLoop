namespace MLoop.DataStore.Interfaces;

/// <summary>
/// Samples production data for retraining dataset creation.
/// </summary>
public interface IDataSampler
{
    /// <summary>
    /// Creates a sample dataset from logged predictions and feedback.
    /// </summary>
    /// <param name="modelName">Model to sample data for</param>
    /// <param name="sampleSize">Number of samples to collect</param>
    /// <param name="strategy">Sampling strategy</param>
    /// <param name="outputPath">Path to save the sampled dataset</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task<SamplingResult> SampleAsync(
        string modelName,
        int sampleSize,
        SamplingStrategy strategy,
        string outputPath,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets statistics about available data for sampling.
    /// </summary>
    Task<SamplingStatistics> GetStatisticsAsync(
        string modelName,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Sampling strategy for dataset creation.
/// </summary>
public enum SamplingStrategy
{
    /// <summary>Random sampling from all available data.</summary>
    Random,

    /// <summary>Stratified sampling to maintain class distribution.</summary>
    Stratified,

    /// <summary>Sample from recent data only.</summary>
    Recent,

    /// <summary>Prioritize samples with feedback.</summary>
    FeedbackPriority,

    /// <summary>Sample from low-confidence predictions.</summary>
    LowConfidence
}

/// <summary>
/// Result of a sampling operation.
/// </summary>
public record SamplingResult(
    string OutputPath,
    int SampledCount,
    int TotalAvailable,
    SamplingStrategy StrategyUsed,
    DateTimeOffset CreatedAt);

/// <summary>
/// Statistics about available data for sampling.
/// </summary>
public record SamplingStatistics(
    string ModelName,
    int TotalPredictions,
    int PredictionsWithFeedback,
    int LowConfidenceCount,
    DateTimeOffset OldestEntry,
    DateTimeOffset NewestEntry);
