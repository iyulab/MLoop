namespace MLoop.DataStore.Interfaces;

/// <summary>
/// Samples production data for retraining dataset creation.
/// </summary>
public interface IDataSampler
{
    /// <summary>
    /// Creates a training dataset from logged predictions and their feedback: each row is the logged
    /// input with the feedback's actual value under the model's label column (empty when the prediction
    /// has none), plus a weight column when the model trains weighted or any feedback carries a weight.
    /// The model's own answer and the feedback source are not written — a column in the file is a
    /// feature to the next training, and the prediction would teach the model to copy itself.
    /// </summary>
    /// <param name="modelName">Model to sample data for</param>
    /// <param name="sampleSize">Number of samples to collect</param>
    /// <param name="strategy">Sampling strategy</param>
    /// <param name="outputPath">Path to save the sampled dataset</param>
    /// <param name="columns">The model's label column, and its weight column if it trains weighted</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task<SamplingResult> SampleAsync(
        string modelName,
        int sampleSize,
        SamplingStrategy strategy,
        string outputPath,
        RetrainingColumns columns,
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
/// The columns a retraining dataset is written with: the model's label, and the weight column it trains
/// with (<c>null</c> when it trains unweighted — a weight column is then written only if feedback carries
/// weights, under <see cref="DefaultWeight"/>).
/// </summary>
public sealed record RetrainingColumns(string Label, string? Weight = null)
{
    /// <summary>
    /// The model's input columns as it was trained on them. A logged input is written under the spelling found
    /// here when the two differ only in case — a client that sent <c>x1</c> to a model trained on <c>X1</c> was
    /// answered, and its rows must merge with the original training data. Empty when unknown.
    /// </summary>
    public IReadOnlyList<string> Inputs { get; init; } = [];

    /// <summary>The weight column's name when the model declares none.</summary>
    public const string DefaultWeight = "weight";
}

/// <summary>
/// Result of a sampling operation. <see cref="LabeledCount"/> rows carry an actual value — the rest
/// have an empty label, which training drops. <see cref="WeightColumn"/> is the column the rows were
/// weighted under, or <c>null</c> when none was written.
/// </summary>
public record SamplingResult(
    string OutputPath,
    int SampledCount,
    int TotalAvailable,
    SamplingStrategy StrategyUsed,
    DateTimeOffset CreatedAt,
    int LabeledCount = 0,
    string? LabelColumn = null,
    string? WeightColumn = null);

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
