namespace MLoop.AIAgent.Core.Memory.Models;

/// <summary>
/// Records the outcome of dataset processing for pattern learning.
/// </summary>
public sealed class ProcessingOutcome
{
    /// <summary>
    /// Whether the processing was successful.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// List of preprocessing steps applied in order.
    /// </summary>
    public List<PreprocessingStep> Steps { get; set; } = [];

    /// <summary>
    /// Time taken for preprocessing (milliseconds).
    /// </summary>
    public long ProcessingTimeMs { get; set; }

    /// <summary>
    /// Model performance metrics if training was performed.
    /// </summary>
    public Dictionary<string, double>? PerformanceMetrics { get; set; }

    /// <summary>
    /// Best performing trainer name.
    /// </summary>
    public string? BestTrainer { get; set; }

    /// <summary>
    /// Error message if processing failed.
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Timestamp of processing.
    /// </summary>
    public DateTime ProcessedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Generates a description for semantic search.
    /// </summary>
    public string Describe()
    {
        if (!Success)
            return $"Failed processing: {ErrorMessage}";

        var stepList = string.Join(" -> ", Steps.Select(s => s.Type));
        var result = $"Successful processing: {stepList}. ";

        if (PerformanceMetrics?.Count > 0)
        {
            var metrics = string.Join(", ", PerformanceMetrics.Select(m => $"{m.Key}={m.Value:F3}"));
            result += $"Metrics: {metrics}. ";
        }

        if (!string.IsNullOrEmpty(BestTrainer))
            result += $"Best trainer: {BestTrainer}.";

        return result;
    }
}

/// <summary>
/// Represents a single preprocessing step.
/// </summary>
public sealed class PreprocessingStep
{
    /// <summary>
    /// Type of preprocessing (e.g., "MissingValueHandler", "OneHotEncoder").
    /// </summary>
    public string Type { get; set; } = default!;

    /// <summary>
    /// Target columns for this step.
    /// </summary>
    public List<string>? TargetColumns { get; set; }

    /// <summary>
    /// Configuration parameters for this step.
    /// </summary>
    public Dictionary<string, object>? Parameters { get; set; }

    /// <summary>
    /// Order in the preprocessing pipeline.
    /// </summary>
    public int Order { get; set; }
}

/// <summary>
/// Recommendation based on similar dataset patterns.
/// </summary>
public sealed class ProcessingRecommendation
{
    /// <summary>
    /// Fingerprint of the similar dataset.
    /// </summary>
    public DatasetFingerprint? SimilarFingerprint { get; set; }

    /// <summary>
    /// Outcome from the similar dataset processing.
    /// </summary>
    public ProcessingOutcome? Outcome { get; set; }

    /// <summary>
    /// Similarity score (0.0 to 1.0).
    /// </summary>
    public float SimilarityScore { get; set; }

    /// <summary>
    /// Recommended preprocessing steps.
    /// </summary>
    public List<PreprocessingStep> RecommendedSteps => Outcome?.Steps ?? [];

    /// <summary>
    /// Confidence in the recommendation.
    /// </summary>
    public float Confidence => SimilarityScore;
}
