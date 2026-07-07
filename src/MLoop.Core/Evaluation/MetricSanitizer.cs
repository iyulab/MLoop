namespace MLoop.Core.Evaluation;

/// <summary>
/// Single authority for replacing non-finite metric values (NaN, +∞, −∞) with a
/// direction-aware worst-case sentinel, before metrics are compared, quality-gated, or
/// serialized. A non-finite metric is not a number the rest of the system can reason about:
/// <list type="bullet">
///   <item><description><c>System.Text.Json</c> rejects NaN/Infinity outright, so persisting
///   such a value crashes the whole training pipeline at save time (the symptom that motivated
///   this authority: R² on a near-zero-variance test split degenerates to −∞, and
///   <c>metrics.json</c> serialization throws "positive/negative infinity cannot be written as
///   valid JSON").</description></item>
///   <item><description>A non-finite measurement is meaningless, so it must never win
///   best-model selection nor pass the promotion gate. Mapping it to the <b>worst</b>
///   representable value for the metric's direction guarantees that.</description></item>
/// </list>
/// The direction is resolved through <see cref="MetricDirection"/> so this can never drift from
/// the compare/gate direction elsewhere (the F-27 class of cross-assembly drift). Both the
/// training path (<c>AutoMLRunner.RunAsync</c>) and the evaluate path (<c>EvaluationEngine</c>)
/// funnel their metric dictionaries through here.
///
/// <para>Why worst-representable and not 0: a naïve <c>NaN → 0</c> guard (which this authority
/// replaces) is actively wrong for lower-is-better metrics — 0 is the <i>best</i> possible RMSE
/// or log-loss, so it launders a broken model into a perfect-looking one that then gets
/// promoted. Direction-aware sentinels avoid that:</para>
/// <list type="bullet">
///   <item><description>lower-is-better (rmse, mae, mse, log_loss, …) → <see cref="double.MaxValue"/></description></item>
///   <item><description>higher-is-better (r_squared, accuracy, auc, f1, …) → <see cref="double.MinValue"/></description></item>
/// </list>
/// </summary>
public static class MetricSanitizer
{
    /// <summary>
    /// Replaces every non-finite value in <paramref name="metrics"/> (in place) with the worst
    /// representable value for that metric's direction. Returns the keys that were replaced —
    /// empty when everything was already finite — so callers with a logger can surface a
    /// "metric undefined (too few samples?)" warning. Null/empty input is a no-op.
    /// </summary>
    public static IReadOnlyList<string> SanitizeInPlace(Dictionary<string, double>? metrics)
    {
        if (metrics is null || metrics.Count == 0)
            return Array.Empty<string>();

        List<string>? replaced = null;
        // Snapshot the keys so the dictionary can be mutated during iteration.
        foreach (var key in metrics.Keys.ToArray())
        {
            var value = metrics[key];
            if (double.IsFinite(value))
                continue;

            metrics[key] = MetricDirection.IsLowerBetter(key) ? double.MaxValue : double.MinValue;
            (replaced ??= new List<string>()).Add(key);
        }

        return replaced ?? (IReadOnlyList<string>)Array.Empty<string>();
    }

    /// <summary>
    /// Fluent variant of <see cref="SanitizeInPlace"/> that sanitizes <paramref name="metrics"/>
    /// in place and returns the same dictionary, for use at <c>return</c> sites that don't need
    /// the replaced-key list.
    /// </summary>
    public static Dictionary<string, double> SanitizeAndReturn(Dictionary<string, double> metrics)
    {
        SanitizeInPlace(metrics);
        return metrics;
    }
}
