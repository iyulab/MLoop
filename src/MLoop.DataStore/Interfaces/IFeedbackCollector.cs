namespace MLoop.DataStore.Interfaces;

/// <summary>
/// Collects ground truth feedback for model performance monitoring.
/// </summary>
public interface IFeedbackCollector
{
    /// <summary>
    /// Records feedback (ground truth) for a previous prediction.
    /// </summary>
    /// <param name="predictionId">ID of the original prediction</param>
    /// <param name="actualValue">The ground truth value</param>
    /// <param name="source">Feedback source (user, system, etc.)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task RecordFeedbackAsync(
        string predictionId,
        object actualValue,
        string? source = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets feedback entries for a model within a time range.
    /// </summary>
    Task<IReadOnlyList<FeedbackEntry>> GetFeedbackAsync(
        string modelName,
        DateTimeOffset? from = null,
        DateTimeOffset? to = null,
        int limit = 1000,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Calculates accuracy metrics based on predictions vs feedback.
    /// </summary>
    Task<FeedbackMetrics> CalculateMetricsAsync(
        string modelName,
        DateTimeOffset? from = null,
        DateTimeOffset? to = null,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Represents a feedback entry linking prediction to ground truth.
/// </summary>
public record FeedbackEntry(
    string PredictionId,
    string ModelName,
    object PredictedValue,
    object ActualValue,
    string? Source,
    DateTimeOffset Timestamp);

/// <summary>
/// Aggregated feedback metrics for model monitoring.
/// </summary>
public record FeedbackMetrics(
    string ModelName,
    int TotalPredictions,
    int TotalFeedback,
    double? Accuracy,
    double? Precision,
    double? Recall,
    DateTimeOffset CalculatedAt);
