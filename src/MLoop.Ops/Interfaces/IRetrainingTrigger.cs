namespace MLoop.Ops.Interfaces;

/// <summary>
/// Evaluates conditions and triggers model retraining.
/// </summary>
public interface IRetrainingTrigger
{
    /// <summary>
    /// Evaluates whether retraining should be triggered based on conditions.
    /// </summary>
    /// <param name="modelName">Model to evaluate</param>
    /// <param name="conditions">Conditions to check</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Evaluation result with decision and reasoning</returns>
    Task<TriggerEvaluation> EvaluateAsync(
        string modelName,
        IEnumerable<RetrainingCondition> conditions,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the default retraining conditions for a model.
    /// </summary>
    Task<IReadOnlyList<RetrainingCondition>> GetDefaultConditionsAsync(
        string modelName,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Condition that triggers retraining when met.
/// </summary>
public record RetrainingCondition(
    ConditionType Type,
    string Name,
    double Threshold,
    string? Description = null);

/// <summary>
/// Types of retraining conditions.
/// </summary>
public enum ConditionType
{
    /// <summary>Accuracy dropped below threshold.</summary>
    AccuracyDrop,

    /// <summary>Data drift detected.</summary>
    DataDrift,

    /// <summary>Time since last training exceeded threshold (days).</summary>
    TimeBased,

    /// <summary>New feedback count exceeded threshold.</summary>
    FeedbackVolume,

    /// <summary>Model performance degraded on recent data.</summary>
    PerformanceDegradation
}

/// <summary>
/// Result of trigger evaluation.
/// </summary>
public record TriggerEvaluation(
    bool ShouldRetrain,
    IReadOnlyList<ConditionResult> ConditionResults,
    string? RecommendedAction,
    DateTimeOffset EvaluatedAt);

/// <summary>
/// Result of evaluating a single condition.
/// </summary>
public record ConditionResult(
    RetrainingCondition Condition,
    bool IsMet,
    double CurrentValue,
    string? Details);
