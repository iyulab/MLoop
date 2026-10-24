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
        var metrics = await _feedbackCollector.CalculateMetricsAsync(modelName, cancellationToken: cancellationToken).ConfigureAwait(false);

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
        var defaults = new List<RetrainingCondition> { RetrainingDefaults.Accuracy(), RetrainingDefaults.Feedback() };

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
                CurrentValue: null,
                Details: "No accuracy data available (no feedback collected)");
        }

        // An established drop, not a point estimate: enough feedback, and even the interval's best case below the
        // threshold. Three answers with two wrong read as 33% accuracy and used to recommend retraining.
        var currentAccuracy = metrics.Accuracy.Value;
        var n = metrics.EffectiveSampleSize ?? metrics.TotalFeedback;
        var evidence = metrics.Weighted ? $"{n:0.#} effective rows of feedback" : $"{n:0} feedback";
        if (n < RetrainingDefaults.MinimumAccuracyFeedback)
        {
            return new ConditionResult(
                condition,
                IsMet: false,
                CurrentValue: currentAccuracy,
                Details: $"Accuracy {currentAccuracy:P1} on {evidence} — too few to judge (needs {RetrainingDefaults.MinimumAccuracyFeedback})");
        }

        var upper = RetrainingDefaults.AccuracyUpperBound(currentAccuracy, n);
        var isMet = upper < condition.Threshold;
        string details;
        if (isMet)
            details = $"Accuracy {currentAccuracy:P1} (95% interval up to {upper:P1}, {evidence}) is below threshold {condition.Threshold:P1}";
        else if (currentAccuracy < condition.Threshold)
            details = $"Accuracy {currentAccuracy:P1} is below threshold {condition.Threshold:P1}, but its 95% interval reaches {upper:P1} on {evidence} — not established";
        else
            details = $"Accuracy {currentAccuracy:P1} is at or above threshold {condition.Threshold:P1} ({evidence})";

        return new ConditionResult(condition, IsMet: isMet, CurrentValue: currentAccuracy, Details: details);
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
            CurrentValue: null,
            Details: $"Condition type '{condition.Type}' is not supported by FeedbackBasedTrigger");
    }
}
