using MLoop.Core.Evaluation;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// CLI promotion-gate metric policy: metric-name/alias resolution, minimum quality-gate
/// thresholds, and degenerate-model detection.
/// <para>
/// This knowledge is intentionally CLI-side (not <c>MLoop.Core</c>): it encodes the CLI's
/// promotion-gate policy — alias spellings the CLI accepts, the 1/N-baseline floors the gate
/// applies, and the degenerate-classifier heuristic — none of which the Core inference/evaluation
/// engines need. The task→canonical-metric facts and metric direction (lower-vs-higher-better)
/// are the shared domain truth and remain in <see cref="TaskMetadata"/> / <see cref="MetricDirection"/>;
/// this type composes them with promotion policy.
/// </para>
/// Extracted from <c>ModelRegistry</c> so deployment (<c>ModelRegistry</c>) and metric policy are
/// separate responsibilities, and so <c>CompareCommand</c>/<c>TrainCommand</c> depend on the policy
/// they actually need rather than reaching into the production registry (SRP).
/// </summary>
public static class MetricPolicy
{
    /// <summary>
    /// Resolves a user-facing metric name/alias (e.g. "f1", "r2", "log-loss") to the
    /// canonical key actually present among <paramref name="availableKeys"/> (e.g.
    /// "f1_score", "r_squared", "log_loss"). The EvaluationEngine stores canonical keys,
    /// while the CLI accepts aliases — without this mapping a raw lookup silently misses
    /// (the root cause of BUG-45's blocked auto-promotion and Compare's ignored --sort).
    /// Returns the matching canonical key, or null if no known variant is present.
    /// </summary>
    public static string? ResolveMetricKey(string metricName, IEnumerable<string> availableKeys)
    {
        if (string.IsNullOrWhiteSpace(metricName))
        {
            return null;
        }

        var keys = availableKeys as ICollection<string> ?? availableKeys.ToList();

        // Exact match wins.
        if (keys.Contains(metricName))
        {
            return metricName;
        }

        var normalized = metricName.Trim().ToLowerInvariant().Replace("-", "_");

        // Map common aliases to their canonical stored keys (most-specific first).
        var candidates = normalized switch
        {
            "f1" or "f1score" => new[] { "f1_score", "macro_f1" },
            "r2" or "rsquared" => new[] { "r_squared" },
            "logloss" => new[] { "log_loss" },
            "accuracy" => new[] { "accuracy", "micro_accuracy", "macro_accuracy" },
            "area_under_roc_curve" => new[] { "auc" },
            _ => new[] { normalized }
        };

        foreach (var candidate in candidates)
        {
            if (keys.Contains(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    /// <summary>
    /// Resolves the canonical metric key for the quality gate, falling back to the task's
    /// default metric when <paramref name="metricName"/> is the deferred "auto" (or otherwise
    /// unresolvable). <see cref="ResolveMetricKey"/> handles explicit metrics and aliases;
    /// this adds task-awareness so directory-based tasks — which init leaves as "auto" — still
    /// engage the gate instead of silently skipping it (BUG-46). Returns null when neither the
    /// requested metric nor the task default is present (e.g. object detection's mAP has no
    /// universal threshold).
    /// </summary>
    public static string? ResolveCanonicalMetricKey(string metricName, string? task, IEnumerable<string> availableKeys)
    {
        var keys = availableKeys as ICollection<string> ?? availableKeys.ToList();

        var resolved = ResolveMetricKey(metricName, keys);
        if (resolved != null)
        {
            return resolved;
        }

        // Task→primary-metric mapping lives in the shared TaskMetadata source of truth so the
        // promotion gate evaluates the same metric init writes and AutoML optimizes (TD-06).
        var taskDefault = TaskMetadata.PrimaryMetric(task);
        return taskDefault != null ? ResolveMetricKey(taskDefault, keys) : null;
    }

    /// <summary>
    /// Returns the minimum viable metric threshold for the quality gate.
    /// Models scoring below this threshold are not promoted to production.
    /// Returns null for metrics without a universal minimum (e.g., error metrics).
    /// </summary>
    public static double? GetMinimumMetricThreshold(string metricName, int? classCount = null)
    {
        return metricName.ToLowerInvariant() switch
        {
            "r_squared" or "r2" => 0.0,                // Must be better than mean prediction
            "auc" or "area_under_roc_curve" => 0.5,     // Must be better than random
            "accuracy" or "micro_accuracy" => classCount.HasValue && classCount.Value > 1
                ? 1.0 / classCount.Value                 // Must be better than random (1/N)
                : 0.0,
            "macro_accuracy" => classCount.HasValue && classCount.Value > 1
                ? 1.0 / classCount.Value                 // Must be better than random (1/N)
                : 0.0,
            "f1" or "f1_score" => classCount.HasValue && classCount.Value > 1
                ? 1.0 / classCount.Value                 // Must be better than random (1/N)
                : 0.0,
            "macro_f1" => classCount.HasValue && classCount.Value > 1
                ? 1.0 / classCount.Value                 // Must be better than random (1/N)
                : 0.0,
            _ => null                                    // No threshold for unknown/error metrics
        };
    }

    /// <summary>
    /// True when the metric is error-direction (lower is better). Delegates to the shared
    /// <see cref="MetricDirection"/> authority so the promotion gate's compare direction cannot
    /// drift from the rest of the system.
    /// </summary>
    public static bool IsErrorMetric(string metricName)
        => MetricDirection.IsLowerBetter(metricName);

    /// <summary>
    /// True when a metric value is the direction-aware worst-case sentinel that
    /// <see cref="MetricSanitizer"/> records for an undefined (NaN/±∞) measurement —
    /// <see cref="double.MaxValue"/> for lower-is-better metrics, <see cref="double.MinValue"/>
    /// for higher-is-better. The sanitizer's contract says such a value "must never … pass the
    /// promotion gate", but the gate only enforced floors for higher-is-better metrics — a
    /// regression optimized on rmse had NO floor at all, so an all-NaN-scoring degenerate model
    /// (recorded as rmse = MaxValue) still auto-promoted as the first production model. Exact
    /// equality is correct here: the sentinels are exact assignments, and no real evaluator
    /// metric lands on ±double.Max/MinValue.
    /// </summary>
    public static bool IsUndefinedMetricSentinel(string metricName, double value)
        => MetricDirection.IsLowerBetter(metricName)
            ? value == double.MaxValue
            : value == double.MinValue;

    /// <summary>
    /// Detects degenerate classification models that achieve high accuracy by only
    /// predicting one class. Returns true if accuracy > 0.5 but F1 ≈ 0 (always-negative
    /// degenerate), or, symmetrically, if the model never predicts the negative class at
    /// all (always-positive degenerate — D16). The F1 check alone only catches the first
    /// case: F1 is computed against the *positive* class, so it is ≈0 when the model
    /// always predicts negative, but stays high (F1 ≈ 2·prevalence/(1+prevalence)) when
    /// the model always predicts positive — which is the common real-world convention for
    /// binary QC/pass-fail data (majority "OK"/pass mapped to the positive label). That
    /// combination (recall ≈ 1, negative_recall ≈ 0) is undetectable from accuracy/F1/AUC
    /// alone and slipped past this gate in a live KAMP SEQ006 run (accuracy 0.747, F1
    /// 0.855, AUC 0.772 — all "healthy" — while negative_recall was exactly 0, i.e. the
    /// promoted model never once predicted the minority "NG" class).
    /// </summary>
    public static bool IsClassificationDegenerateModel(Dictionary<string, double> metrics)
    {
        // Check binary: accuracy > 0.5 but f1_score == 0 (always predicts negative)
        if (metrics.TryGetValue("f1_score", out var f1) &&
            metrics.TryGetValue("accuracy", out var acc))
        {
            if (acc > 0.5 && f1 < 0.001)
                return true;
        }

        // Check binary: accuracy > 0.5 but negative_recall == 0 (always predicts positive — D16)
        if (metrics.TryGetValue("negative_recall", out var negRecall) &&
            metrics.TryGetValue("accuracy", out var accForNegRecall))
        {
            if (accForNegRecall > 0.5 && negRecall < 0.001)
                return true;
        }

        // Check multiclass: macro_accuracy > random but macro_f1 == 0
        if (metrics.TryGetValue("macro_f1", out var macroF1) &&
            metrics.TryGetValue("macro_accuracy", out var macroAcc))
        {
            if (macroAcc > 0.3 && macroF1 < 0.001)
                return true;
        }

        return false;
    }
}
