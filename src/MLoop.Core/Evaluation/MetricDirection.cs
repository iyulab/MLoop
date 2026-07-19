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

    /// <summary>
    /// True when this authority actually recognizes the metric's direction — either as a known
    /// lower-is-better metric (<see cref="IsLowerBetter"/>) or a known higher-is-better one.
    /// False means <see cref="IsLowerBetter"/> returned its catch-all <c>false</c> (i.e.
    /// "maximize" is a *default*, not a determination). A judgment-delegation consumer needs this
    /// distinction: a lower-is-better custom metric arriving unrecognized would otherwise be
    /// silently ranked as maximize, making the worst model "best". Incompleteness here is safe by
    /// construction — an unrecognized higher-is-better metric reports <c>default</c>, which a
    /// conservative consumer treats as ambiguous (over-cautious), never as a wrong direction.
    /// </summary>
    public static bool IsKnown(string metricName)
    {
        if (string.IsNullOrWhiteSpace(metricName))
            return false;
        return IsLowerBetter(metricName) || IsKnownHigherBetter(metricName.ToLowerInvariant());
    }

    /// <summary>
    /// Recognized higher-is-better metrics (canonical primaries from <see cref="TaskMetadata"/> —
    /// accuracy/macro_accuracy/micro_accuracy, r_squared, auc, ndcg, detection_rate — plus common
    /// classification/ranking variants). Kept as an explicit recognizer here, the direction
    /// authority's home, rather than inferring "not lower-better ⇒ higher-better" so that
    /// <see cref="IsKnown"/> can tell a determination from the default.
    /// </summary>
    private static bool IsKnownHigherBetter(string lower) =>
        lower.Contains("accuracy")
        || lower.Contains("auc")
        || lower.Contains("auprc")
        || lower.Contains("r_squared")
        || lower.Contains("rsquared")
        || lower == "r2"
        || lower.Contains("f1")
        || lower.Contains("precision")
        || lower.Contains("recall")
        || lower.Contains("ndcg")
        || lower.Contains("dcg")
        || lower == "detection_rate";
}
