using MLoop.DataStore.Interfaces;
using MLoop.Ops.Interfaces;

namespace MLoop.Ops.Services;

/// <summary>
/// Feedback-based retraining trigger that evaluates accuracy and feedback volume.
/// Supports AccuracyDrop and FeedbackVolume conditions only.
/// </summary>
public sealed class FeedbackBasedTrigger : IRetrainingTrigger
{
    private readonly IFeedbackCollector _feedbackCollector;

    public FeedbackBasedTrigger(IFeedbackCollector feedbackCollector)
    {
        _feedbackCollector = feedbackCollector ?? throw new ArgumentNullException(nameof(feedbackCollector));
    }

    /// <inheritdoc />
    public async Task<TriggerEvaluation> EvaluateAsync(
        string modelName,
        IEnumerable<RetrainingCondition> conditions,
        CancellationToken cancellationToken = default)
    {
        var results = new List<ConditionResult>();
        var metrics = await _feedbackCollector.CalculateMetricsAsync(modelName, cancellationToken: cancellationToken);

        foreach (var condition in conditions)
        {
            var result = condition.Type switch
            {
                ConditionType.AccuracyDrop => EvaluateAccuracyDrop(condition, metrics),
                ConditionType.FeedbackVolume => EvaluateFeedbackVolume(condition, metrics),
                _ => CreateUnsupportedResult(condition)
            };
            results.Add(result);
        }

        var shouldRetrain = results.Any(r => r.IsMet);
        var triggeredConditions = results.Where(r => r.IsMet).ToList();

        string? recommendation = null;
        if (shouldRetrain)
        {
            var reasons = triggeredConditions.Select(r => r.Condition.Name);
            recommendation = $"Retraining recommended due to: {string.Join(", ", reasons)}";
        }

        return new TriggerEvaluation(
            ShouldRetrain: shouldRetrain,
            ConditionResults: results,
            RecommendedAction: recommendation,
            EvaluatedAt: DateTimeOffset.UtcNow);
    }

    /// <inheritdoc />
    public Task<IReadOnlyList<RetrainingCondition>> GetDefaultConditionsAsync(
        string modelName,
        CancellationToken cancellationToken = default)
    {
        var defaults = new List<RetrainingCondition>
        {
            new(ConditionType.AccuracyDrop, "accuracy_threshold", 0.7,
                "Trigger retraining when accuracy drops below 70%"),
            new(ConditionType.FeedbackVolume, "feedback_count", 100,
                "Trigger retraining when 100+ new feedback entries are available")
        };

        return Task.FromResult<IReadOnlyList<RetrainingCondition>>(defaults);
    }

    private static ConditionResult EvaluateAccuracyDrop(
        RetrainingCondition condition,
        FeedbackMetrics metrics)
    {
        if (!metrics.Accuracy.HasValue)
        {
            return new ConditionResult(
                condition,
                IsMet: false,
                CurrentValue: 0,
                Details: "No accuracy data available (no feedback collected)");
        }

        var currentAccuracy = metrics.Accuracy.Value;
        var isMet = currentAccuracy < condition.Threshold;

        return new ConditionResult(
            condition,
            IsMet: isMet,
            CurrentValue: currentAccuracy,
            Details: isMet
                ? $"Accuracy {currentAccuracy:P1} is below threshold {condition.Threshold:P1}"
                : $"Accuracy {currentAccuracy:P1} is at or above threshold {condition.Threshold:P1}");
    }

    private static ConditionResult EvaluateFeedbackVolume(
        RetrainingCondition condition,
        FeedbackMetrics metrics)
    {
        var feedbackCount = metrics.TotalFeedback;
        var isMet = feedbackCount >= condition.Threshold;

        return new ConditionResult(
            condition,
            IsMet: isMet,
            CurrentValue: feedbackCount,
            Details: isMet
                ? $"Feedback count {feedbackCount} meets threshold {condition.Threshold:N0}"
                : $"Feedback count {feedbackCount} below threshold {condition.Threshold:N0}");
    }

    private static ConditionResult CreateUnsupportedResult(RetrainingCondition condition)
    {
        return new ConditionResult(
            condition,
            IsMet: false,
            CurrentValue: 0,
            Details: $"Condition type '{condition.Type}' is not supported by FeedbackBasedTrigger");
    }
}
