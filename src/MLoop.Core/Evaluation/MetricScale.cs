namespace MLoop.Core.Evaluation;

/// <summary>
/// Whether a metric lives on a fixed scale, so that a difference between two of its values can be
/// read as an absolute number. The third half of metric knowledge, beside the metric's canonical
/// <see cref="MetricNames">name</see> and its optimization <see cref="MetricDirection">direction</see>.
/// </summary>
/// <remarks>
/// <para>Nothing owned this, and the gap was load-bearing. Overfitting detection compares a
/// training score with a test score and calls a gap larger than <c>0.1</c> suspicious. On accuracy
/// that is a sentence about the model. On <c>rmse</c> it is a sentence about the unit the label
/// happens to be in: predicting a price in won, a gap of 0.1 is beneath notice and the check never
/// fires; predicting a probability, almost any two runs differ by more and it always fires. The
/// same threshold means two different things, and neither is visible at the call site.</para>
/// <para>This is why widening a bounded-metric check to every task is not a substitution. A task's
/// primary metric can be <c>mae</c> (forecasting), <c>rmse</c> (recommendation) or
/// <c>average_distance</c> (clustering) — all three carry the label's own units.</para>
/// </remarks>
public static class MetricScale
{
    /// <summary>
    /// Metrics whose values sit on a fixed scale of roughly zero to one, where an absolute
    /// difference is comparable across datasets.
    /// </summary>
    /// <remarks>
    /// <c>r_squared</c> is included although it is unbounded below: it is capped at 1, it is a
    /// proportion of variance rather than a quantity in the label's units, and a difference of 0.1
    /// means the same thing whatever the label measures. The rest are proportions or rates by
    /// construction. <c>log_loss_reduction</c> is likewise capped at 1 and unitless.
    /// </remarks>
    private static readonly HashSet<string> UnitScaled = new(StringComparer.OrdinalIgnoreCase)
    {
        "accuracy",
        "macro_accuracy",
        "micro_accuracy",
        "top_k_accuracy",
        "auc",
        "auprc",
        "f1_score",
        "precision",
        "recall",
        "negative_precision",
        "negative_recall",
        "ndcg",
        "detection_rate",
        "r_squared",
        "log_loss_reduction",
    };

    /// <summary>
    /// True when a difference between two values of <paramref name="metricName"/> can be judged as
    /// an absolute number. False both for metrics carrying the label's units (<c>rmse</c>,
    /// <c>mae</c>, <c>mse</c>, <c>log_loss</c>, <c>average_distance</c>,
    /// <c>davies_bouldin_index</c>, <c>mape</c>) and for a metric this authority does not know —
    /// the two are deliberately the same answer here, because both mean "do not read a raw
    /// difference as a verdict", and a caller that needs to tell them apart should ask
    /// <see cref="MetricDirection.IsKnown"/>.
    /// </summary>
    public static bool IsUnitScaled(string? metricName)
        => !string.IsNullOrWhiteSpace(metricName) && UnitScaled.Contains(metricName.Trim());
}
