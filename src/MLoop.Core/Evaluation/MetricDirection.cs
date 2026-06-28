namespace MLoop.Core.Evaluation;

/// <summary>
/// Single source of truth for a metric's optimization direction (lower-is-better vs
/// higher-is-better). This knowledge was previously duplicated and had drifted across four sites —
/// <c>ModelRegistry.IsErrorMetric</c> (the only complete copy), <c>EvaluateCommand.IsLowerBetterMetric</c>,
/// <c>CompareCommand</c> (three inline copies), and <c>FileModelComparer.IsLowerBetterMetric</c> — with
/// the latter three all missing clustering's <c>average_distance</c>/<c>davies_bouldin_index</c> and
/// <c>mape</c>. The consequence (F-27): <c>mloop compare</c> sorting clustering experiments by their
/// canonical metric (<c>average_distance</c>) ranked the <i>worst</i> model first, and evaluate's diff
/// coloring was inverted. Lives in <c>MLoop.Core</c> so both <c>MLoop.CLI</c> and <c>MLoop.Ops</c>
/// converge here — the same TD-06 convergence done for task→metric.
/// </summary>
public static class MetricDirection
{
    /// <summary>
    /// True when a lower value of the metric is better — error/distance metrics (rmse, mae, mse,
    /// mape, log loss) and clustering's separation metrics (average_distance, davies_bouldin_index).
    /// False for higher-is-better metrics (accuracy, auc, r_squared, ndcg, …). Matched
    /// case-insensitively on the canonical metric key.
    /// </summary>
    public static bool IsLowerBetter(string metricName)
    {
        var lower = metricName.ToLowerInvariant();
        return lower.Contains("error")
            || lower.Contains("mae")
            || lower.Contains("mse")
            || lower.Contains("rmse")
            || lower.Contains("loss")
            || lower.Contains("mape")
            || lower == "average_distance"
            || lower == "davies_bouldin_index";
    }
}
