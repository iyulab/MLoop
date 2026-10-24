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
    /// <param name="source">Where the value came from (a person, a reviewing model, a system) — free text, the caller's vocabulary</param>
    /// <param name="weight">
    /// How much this row counts when the feedback becomes training or evaluation data: the relative
    /// frequency it stands for (for a row reviewed with probability π, 1/π). Finite and not negative;
    /// <c>null</c> counts as 1.
    /// </param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="weight"/> is negative, NaN or infinite.</exception>
    Task RecordFeedbackAsync(
        string predictionId,
        object actualValue,
        string? source = null,
        double? weight = null,
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
        CancellationToken cancellationToken = default,
        string? experimentId = null);
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
    DateTimeOffset Timestamp,
    double? Weight = null,
    string? ExperimentId = null)
{
    /// <summary>The weight this entry counts with — 1 when none was recorded.</summary>
    public double EffectiveWeight => Weight ?? 1.0;

    /// <summary>
    /// The entry that stands for each prediction. Feedback is append-only, so recording it again for a
    /// prediction corrects it: the most recent entry wins, and each prediction is counted once.
    /// </summary>
    public static IReadOnlyList<FeedbackEntry> LatestPerPrediction(IEnumerable<FeedbackEntry> entries) =>
        entries
            .GroupBy(e => e.PredictionId)
            .Select(g => g.OrderByDescending(e => e.Timestamp).First())
            .ToList();

    /// <summary>Throws when <paramref name="weight"/> cannot weight a row: negative, NaN or infinite.</summary>
    public static void ValidateWeight(double? weight)
    {
        if (weight is { } w && (double.IsNaN(w) || double.IsInfinity(w) || w < 0))
            throw new ArgumentOutOfRangeException(nameof(weight), w,
                "A feedback weight is a finite number that is not negative (for a row reviewed with probability π, 1/π).");
    }
}

/// <summary>
/// Aggregated feedback metrics for model monitoring. When any entry carries a weight, <see cref="Accuracy"/>
/// counts each prediction by its weight (<see cref="Weighted"/>) — the estimate a sample drawn with unequal
/// probabilities needs; counts stay row counts.
/// </summary>
/// <param name="ModelName">The model name the metrics are for.</param>
/// <param name="TotalFeedback">Predictions with feedback counted — each once, its latest entry standing (rows, not
/// weights).</param>
/// <param name="Accuracy">Share of counted predictions whose answer matched the feedback; null without feedback.</param>
/// <param name="CalculatedAt">When these metrics were computed.</param>
/// <param name="Weighted">Whether any entry carried a weight, so <see cref="Accuracy"/> is a weighted estimate.</param>
/// <param name="EffectiveSampleSize">How many rows <see cref="Accuracy"/> is worth as evidence: the row count
/// unweighted, Kish's (Σw)² ÷ Σw² weighted — a few heavy rows carry less certainty than as many equal ones. Null
/// without feedback.</param>
/// <param name="ExperimentId">The experiment these metrics are scoped to, or null for every answer under the model
/// name.</param>
/// <param name="UnattributedFeedback">Scoped metrics only: feedback left out because neither it nor a prediction log
/// line names the experiment that answered (written before feedback recorded it, its log since pruned).</param>
public record FeedbackMetrics(
    string ModelName,
    int TotalFeedback,
    double? Accuracy,
    DateTimeOffset CalculatedAt,
    bool Weighted = false,
    double? EffectiveSampleSize = null,
    string? ExperimentId = null,
    int UnattributedFeedback = 0);
