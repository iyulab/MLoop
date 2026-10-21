using MLoop.DataStore.Interfaces;
using MLoop.Ops.Interfaces;

namespace MLoop.Ops.Services;

/// <summary>
/// The retraining conditions a model is checked against when nobody names others — one list, read by
/// <c>mloop trigger check</c> and <c>GET /trigger</c> alike. Before it existed the CLI checked accuracy
/// and feedback volume while the API checked only the time since training, so the same model got two
/// different answers depending on which surface asked.
/// </summary>
public static class RetrainingDefaults
{
    /// <summary>Retrain when feedback accuracy falls below this.</summary>
    public const double AccuracyThreshold = 0.7;

    /// <summary>Retrain once this many feedback entries are available.</summary>
    public const int FeedbackVolume = 100;

    /// <summary>Retrain when the latest training is older than this many days.</summary>
    public const int IntervalDays = TimeBasedTrigger.DefaultRetrainingIntervalDays;

    /// <summary>Condition names — also what a caller matches to override one default.</summary>
    public const string AccuracyName = "accuracy_threshold";
    public const string FeedbackVolumeName = "feedback_count";
    public const string IntervalName = "Scheduled Retraining";
    public const string DriftName = "input_drift";

    /// <summary>Retrain when any input column's PSI against the training profile reaches this (significant drift).</summary>
    public const double DriftThreshold = MLoop.Core.Monitoring.DriftMeasure.SignificantThreshold;

    public static RetrainingCondition Accuracy(double threshold = AccuracyThreshold) =>
        new(ConditionType.AccuracyDrop, AccuracyName, threshold,
            $"Trigger retraining when accuracy drops below {threshold:P0}");

    public static RetrainingCondition Feedback(int count = FeedbackVolume) =>
        new(ConditionType.FeedbackVolume, FeedbackVolumeName, count,
            $"Trigger retraining when {count}+ new feedback entries are available");

    public static RetrainingCondition Interval(int days = IntervalDays) =>
        new(ConditionType.TimeBased, IntervalName, days,
            $"Retrain if more than {days} days since last training");

    public static RetrainingCondition Drift(double threshold = DriftThreshold) =>
        new(ConditionType.DataDrift, DriftName, threshold,
            $"Trigger retraining when an input column's PSI against the training data reaches {threshold:0.##}");

    /// <summary>Every default condition, in the order they are reported.</summary>
    public static IReadOnlyList<RetrainingCondition> All { get; } = [Accuracy(), Feedback(), Interval(), Drift()];
}

/// <summary>
/// Evaluates every retraining condition type the project can measure by handing each condition to the
/// trigger that measures it — feedback accuracy and volume to <see cref="FeedbackBasedTrigger"/>, time
/// since training to <see cref="TimeBasedTrigger"/>, input drift to <see cref="DriftEvaluator"/> over the
/// prediction logs — and reports the results in the order given.
/// </summary>
public sealed class CompositeRetrainingTrigger : IRetrainingTrigger
{
    private readonly FeedbackBasedTrigger _feedback;
    private readonly TimeBasedTrigger _time;
    private readonly string _projectRoot;
    private readonly IPredictionLogger _predictionLogs;

    public CompositeRetrainingTrigger(string projectRoot, IFeedbackCollector feedbackCollector, IPredictionLogger predictionLogs)
    {
        _feedback = new FeedbackBasedTrigger(feedbackCollector);
        _time = new TimeBasedTrigger(projectRoot);
        _projectRoot = projectRoot;
        _predictionLogs = predictionLogs ?? throw new ArgumentNullException(nameof(predictionLogs));
    }

    /// <inheritdoc />
    public async Task<TriggerEvaluation> EvaluateAsync(
        string modelName,
        IEnumerable<RetrainingCondition> conditions,
        CancellationToken cancellationToken = default)
    {
        var list = conditions.ToList();
        var timeConditions = list.Where(c => c.Type == ConditionType.TimeBased).ToList();
        var driftConditions = list.Where(c => c.Type == ConditionType.DataDrift).ToList();
        // Everything else goes to the feedback trigger, which also reports the types nothing measures
        // yet (performance degradation) as unsupported rather than dropping them.
        var otherConditions = list.Where(c => c.Type is not (ConditionType.TimeBased or ConditionType.DataDrift)).ToList();

        var results = new List<ConditionResult>();
        if (otherConditions.Count > 0)
            results.AddRange((await _feedback.EvaluateAsync(modelName, otherConditions, cancellationToken)
                .ConfigureAwait(false)).ConditionResults);
        if (timeConditions.Count > 0)
            results.AddRange((await _time.EvaluateAsync(modelName, timeConditions, cancellationToken)
                .ConfigureAwait(false)).ConditionResults);
        if (driftConditions.Count > 0)
        {
            var check = await DriftEvaluator.CheckAsync(_projectRoot, _predictionLogs, modelName, cancellationToken: cancellationToken)
                .ConfigureAwait(false);
            foreach (var condition in driftConditions)
            {
                var psi = check.Status == DriftCheckStatus.Measured ? check.Report?.MaxPsi : null;
                results.Add(new ConditionResult(condition, psi >= condition.Threshold, psi, check.Message)
                {
                    Drift = check.Report,
                });
            }
        }

        var ordered = results.OrderBy(r => list.IndexOf(r.Condition)).ToList();
        var met = ordered.Where(r => r.IsMet).ToList();

        return new TriggerEvaluation(
            ShouldRetrain: met.Count > 0,
            ConditionResults: ordered,
            RecommendedAction: met.Count > 0
                ? $"Retraining recommended due to: {string.Join(", ", met.Select(r => r.Condition.Name))}"
                : null,
            EvaluatedAt: DateTimeOffset.UtcNow);
    }

    /// <inheritdoc />
    public Task<IReadOnlyList<RetrainingCondition>> GetDefaultConditionsAsync(
        string modelName,
        CancellationToken cancellationToken = default) =>
        Task.FromResult(RetrainingDefaults.All);
}
