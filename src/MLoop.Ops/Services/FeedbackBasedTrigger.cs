using MLoop.Core.Evaluation;
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
    private readonly Func<string, CancellationToken, Task<string?>>? _productionExperiment;
    private readonly Func<string, CancellationToken, Task<string?>>? _productionTask;
    private readonly Func<string, CancellationToken, Task<double?>>? _productionHeldOutError;

    /// <param name="feedbackCollector">Where feedback is read.</param>
    /// <param name="productionExperiment">The experiment in production for a model name, or null when none is —
    /// accuracy is then judged on that experiment's answers alone. Without it, on every answer under the name.</param>
    /// <param name="productionTask">The task of the model in production (<c>similarity</c>, <c>regression</c>, …), or null when
    /// unknown. A task whose answer is a value (<see cref="TaskMetadata.AnswersAreValues"/>) has no accuracy for an answer that
    /// stated no interval — "the same value" is not a measure of it — and the condition says so instead of asking for retraining.</param>
    /// <param name="productionHeldOutError">The RMSE the model in production made on the rows it held out at training, or null when it
    /// recorded none (a ranking model's score is not on the scale of its labels). What <see cref="ConditionType.ErrorRise"/> compares the
    /// feedback's error with; without it the condition says it has nothing to compare with.</param>
    public FeedbackBasedTrigger(
        IFeedbackCollector feedbackCollector,
        Func<string, CancellationToken, Task<string?>>? productionExperiment = null,
        Func<string, CancellationToken, Task<string?>>? productionTask = null,
        Func<string, CancellationToken, Task<double?>>? productionHeldOutError = null)
    {
        _feedbackCollector = feedbackCollector ?? throw new ArgumentNullException(nameof(feedbackCollector));
        _productionExperiment = productionExperiment;
        _productionTask = productionTask;
        _productionHeldOutError = productionHeldOutError;
    }

    /// <inheritdoc />
    public async Task<TriggerEvaluation> EvaluateAsync(
        string modelName,
        IEnumerable<RetrainingCondition> conditions,
        CancellationToken cancellationToken = default)
    {
        var results = new List<ConditionResult>();
        // Production's own answers: after a promotion the replaced model's feedback is not the new one's accuracy,
        // and a shadow candidate's (predict --experiment) is not production's.
        var experimentId = _productionExperiment is null
            ? null
            : await _productionExperiment(modelName, cancellationToken).ConfigureAwait(false);
        var task = _productionTask is null
            ? null
            : await _productionTask(modelName, cancellationToken).ConfigureAwait(false);
        var valueAnswers = TaskMetadata.AnswersAreValues(task);
        var heldOutError = _productionHeldOutError is null
            ? null
            : await _productionHeldOutError(modelName, cancellationToken).ConfigureAwait(false);
        var metrics = await _feedbackCollector.CalculateMetricsAsync(
            modelName, cancellationToken: cancellationToken, experimentId: experimentId, valueAnswers: valueAnswers).ConfigureAwait(false);

        foreach (var condition in conditions)
        {
            // A task whose answers carry no prediction id (a forecast) can never receive feedback: every condition that reads feedback
            // has nothing to read, and "no data" would promise data that cannot arrive.
            if (!TaskMetadata.TakesFeedback(task) && condition.Type is ConditionType.AccuracyDrop or ConditionType.FeedbackVolume
                    or ConditionType.IntervalCoverage or ConditionType.ErrorRise)
            {
                results.Add(new ConditionResult(
                    condition,
                    IsMet: false,
                    CurrentValue: null,
                    Details: $"Does not apply: a {task} model's forecasts are not logged and carry no prediction id, so no feedback can name one" +
                             " — `mloop evaluate` on the series that followed is how to judge it")
                {
                    Applies = false,
                });
                continue;
            }

            var result = condition.Type switch
            {
                ConditionType.AccuracyDrop => EvaluateAccuracyDrop(condition, metrics, valueAnswers ? task : null),
                ConditionType.FeedbackVolume => EvaluateFeedbackVolume(condition, metrics),
                ConditionType.IntervalCoverage => EvaluateIntervalCoverage(condition, metrics, answersClass: task is not null && !valueAnswers ? task : null),
                ConditionType.ErrorRise => EvaluateErrorRise(condition, metrics, valueAnswers ? task : null, heldOutError),
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
            RetrainingDefaults.Accuracy(), RetrainingDefaults.Coverage(), RetrainingDefaults.ErrorRise(), RetrainingDefaults.Feedback(),
        };

        return Task.FromResult<IReadOnlyList<RetrainingCondition>>(defaults);
    }

    private static ConditionResult EvaluateAccuracyDrop(
        RetrainingCondition condition,
        FeedbackMetrics metrics,
        string? valueTask)
    {
        if (!metrics.Accuracy.HasValue)
        {
            // A value answer that stated no interval has no accuracy to drop — say why, so an empty reading is not taken for
            // "no feedback" and the user is pointed at what does judge a value: an interval, or `evaluate` on labeled rows.
            if (valueTask is not null && metrics.TotalFeedback > 0 && metrics.IntervalCoverage is null)
                return new ConditionResult(
                    condition,
                    IsMet: false,
                    CurrentValue: null,
                    Details: $"Accuracy does not apply: this {valueTask} model answers a value and its predictions stated no interval, so \"the same value\" judges none of the {metrics.TotalFeedback} feedback entries" + Scope(metrics) +
                             " — `mloop evaluate` on labeled rows is how to judge it")
                {
                    Applies = false,
                };
            // Feedback that is all read as interval coverage has no accuracy to drop: say so, not that none was collected.
            return new ConditionResult(
                condition,
                IsMet: false,
                CurrentValue: null,
                Details: metrics.IntervalCoverage is { } coverage
                    ? $"Accuracy does not apply: the {metrics.TotalFeedback} feedback entries answer predictions that stated an interval, read as interval coverage ({coverage.Observed:P1} inside the {coverage.Nominal:P0} stated)" + Scope(metrics)
                    : "No accuracy data available (no feedback collected" + Scope(metrics) + ")")
            {
                Applies = metrics.IntervalCoverage is null,
            };
        }

        // An established drop, not a point estimate: enough feedback, and even the interval's best case below the
        // threshold. Three answers with two wrong read as 33% accuracy and used to recommend retraining.
        var currentAccuracy = metrics.Accuracy.Value;
        var n = metrics.EffectiveSampleSize ?? metrics.TotalFeedback;
        var evidence = (metrics.Weighted ? $"{n:0.#} effective rows of feedback" : $"{n:0} feedback") + Scope(metrics);
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

    /// <summary>Which answers the accuracy is of, when scoped, and what was left out.</summary>
    private static string Scope(FeedbackMetrics metrics) =>
        metrics.ExperimentId is not { } experiment
            ? ""
            : $" on {experiment}'s answers"
              + (metrics.UnattributedFeedback > 0
                  ? $"; {metrics.UnattributedFeedback} earlier feedback names no experiment its log still records and is not counted"
                  : "");

    /// <summary>
    /// The coverage of the intervals the answers stated, judged as accuracy is: not on the point estimate but on whether even
    /// its best case (the 95 % Wilson upper bound over the effective rows) is under the level they stated less the tolerance
    /// (<see cref="RetrainingCondition.Threshold"/>), and only with enough rows to say.
    /// </summary>
    private static ConditionResult EvaluateIntervalCoverage(
        RetrainingCondition condition,
        FeedbackMetrics metrics,
        string? answersClass)
    {
        // A model whose task is known and answers a class or a text states no interval: not «no data yet» but nothing to check.
        if (answersClass is not null && metrics.IntervalCoverage is null)
        {
            return new ConditionResult(
                condition,
                IsMet: false,
                CurrentValue: null,
                Details: $"Interval coverage does not apply: this {answersClass} model's answer is a class or a text, which states no interval")
            {
                Applies = false,
            };
        }

        if (metrics.IntervalCoverage is not { } coverage)
        {
            return new ConditionResult(
                condition,
                IsMet: false,
                CurrentValue: null,
                Details: "No interval coverage data available (no feedback for a prediction that stated an interval" + Scope(metrics) + ")");
        }

        var n = coverage.EffectiveN;
        var evidence = (metrics.Weighted ? $"{n:0.#} effective rows of feedback" : $"{coverage.N} feedback") + Scope(metrics);
        if (n < RetrainingDefaults.MinimumAccuracyFeedback)
        {
            return new ConditionResult(
                condition,
                IsMet: false,
                CurrentValue: coverage.Observed,
                Details: $"Interval coverage {coverage.Observed:P1} of the {coverage.Nominal:P0} stated, on {evidence} — too few to judge (needs {RetrainingDefaults.MinimumAccuracyFeedback})");
        }

        var floor = coverage.Nominal - condition.Threshold;
        var upper = RetrainingDefaults.AccuracyUpperBound(coverage.Observed, n);
        var isMet = upper < floor;
        var stated = $"the {coverage.Nominal:P0} stated";
        string details;
        if (isMet)
            details = $"Interval coverage {coverage.Observed:P1} of {stated} (95% interval up to {upper:P1}, {evidence}) is below {floor:P1}";
        else if (coverage.Observed < floor)
            details = $"Interval coverage {coverage.Observed:P1} of {stated} is below {floor:P1}, but its 95% interval reaches {upper:P1} on {evidence} — not established";
        else
            details = $"Interval coverage {coverage.Observed:P1} of {stated} is at or above {floor:P1} ({evidence})";

        return new ConditionResult(condition, IsMet: isMet, CurrentValue: coverage.Observed, Details: details);
    }

    /// <summary>
    /// A model that answers values, judged by how far its answers were from the actual values: not on the point estimate of the feedback's
    /// RMSE but on whether even its 95 % lower end is at the ratio of the error the model made on its held-out rows
    /// (<see cref="RetrainingCondition.Threshold"/>), and only with enough rows to say. A class or text answer has no distance and a model
    /// that recorded no held-out RMSE has nothing to compare with — each says so rather than reading as «no problem».
    /// </summary>
    private static ConditionResult EvaluateErrorRise(
        RetrainingCondition condition,
        FeedbackMetrics metrics,
        string? valueTask,
        double? heldOutError)
    {
        if (valueTask is null)
        {
            return new ConditionResult(
                condition,
                IsMet: false,
                CurrentValue: null,
                Details: "Error rise does not apply: this model's answer is a class or a text, not a value")
            {
                Applies = false,
            };
        }

        if (heldOutError is not { } baseline)
        {
            return new ConditionResult(
                condition,
                IsMet: false,
                CurrentValue: null,
                Details: $"Error rise does not apply: the {valueTask} model in production recorded no RMSE on its held-out rows" +
                         (string.Equals(valueTask, "ranking", StringComparison.OrdinalIgnoreCase)
                             ? " — a ranking score is not on the scale of its labels, so its feedback has no error to read"
                             : " to compare the feedback with"))
            {
                Applies = false,
            };
        }

        if (metrics.ValueError is not { } error)
        {
            return new ConditionResult(
                condition,
                IsMet: false,
                CurrentValue: null,
                Details: "No error data available (no feedback whose answer and actual value are both numbers" + Scope(metrics) + ")");
        }

        var ratio = error.Rmse / baseline;
        var evidence = (metrics.Weighted ? $"{error.EffectiveN:0.#} effective rows of feedback" : $"{error.N} feedback") + Scope(metrics);
        if (error.EffectiveN < RetrainingDefaults.MinimumAccuracyFeedback)
        {
            return new ConditionResult(
                condition,
                IsMet: false,
                CurrentValue: ratio,
                Details: $"Error {ratio:0.00}× the held-out error ({Formatted(error.Rmse)} against {Formatted(baseline)}) on {evidence} — too few to judge (needs {RetrainingDefaults.MinimumAccuracyFeedback})");
        }

        var lower = RetrainingDefaults.ErrorLowerBound(error) / baseline;
        var isMet = lower >= condition.Threshold;
        string details;
        if (isMet)
            details = $"Error {ratio:0.00}× the held-out error (95% interval from {lower:0.00}×, {Formatted(error.Rmse)} against {Formatted(baseline)}, {evidence}) is at least {condition.Threshold:0.##}×";
        else if (ratio >= condition.Threshold)
            details = $"Error {ratio:0.00}× the held-out error is at least {condition.Threshold:0.##}×, but its 95% interval starts at {lower:0.00}× on {evidence} — not established";
        else
            details = $"Error {ratio:0.00}× the held-out error ({Formatted(error.Rmse)} against {Formatted(baseline)}, {evidence}) is under {condition.Threshold:0.##}×";

        return new ConditionResult(condition, IsMet: isMet, CurrentValue: ratio, Details: details);
    }

    private static string Formatted(double value) => value.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);

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
