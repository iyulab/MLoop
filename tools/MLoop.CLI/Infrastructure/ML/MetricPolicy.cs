using System.Globalization;
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
    /// Resolves a user-facing metric name (e.g. "f1", "r2", "log-loss") to the canonical key
    /// actually present among <paramref name="availableKeys"/> (e.g. "f1_score", "r_squared",
    /// "log_loss"). The EvaluationEngine stores canonical keys, while the CLI accepts any
    /// spelling — without this mapping a raw lookup silently misses (the root cause of blocked
    /// auto-promotion and of Compare's ignored --sort). Returns the matching key, or null if no
    /// known variant is present.
    /// </summary>
    /// <remarks>
    /// Spelling is <see cref="MetricNames"/>' business — this method once kept its own alias
    /// switch, the fourth such vocabulary in the tree. What remains here is gate policy: under
    /// which <i>stored key</i> a canonical name may be found when the task reports a family of it
    /// (an accuracy under a multiclass task is stored as micro or macro accuracy).
    /// </remarks>
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

        var canonical = MetricNames.Canonical(metricName)
            ?? metricName.Trim().ToLowerInvariant().Replace("-", "_");

        // The stored keys a canonical name may live under, most-specific first.
        var candidates = canonical switch
        {
            "f1_score" => new[] { "f1_score", "macro_f1" },
            "accuracy" => new[] { "accuracy", "micro_accuracy", "macro_accuracy" },
            _ => new[] { canonical }
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
    /// engage the gate instead of silently skipping it. Returns null when neither the
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
        // promotion gate evaluates the same metric init writes and AutoML optimizes.
        var taskDefault = TaskMetadata.PrimaryMetric(task);
        return taskDefault != null ? ResolveMetricKey(taskDefault, keys) : null;
    }

    /// <summary>
    /// Returns the minimum viable metric threshold for the quality gate.
    /// Models scoring below this threshold are not promoted to production.
    /// Returns null for metrics without a universal minimum (e.g., error metrics).
    /// </summary>
    /// <param name="majorityClassRatio">
    /// The label's no-information rate (see <see cref="MLoop.Core.Models.ColumnSchema.MajorityClassRatio"/>),
    /// or null when it was not measured. Only <c>accuracy</c>/<c>micro_accuracy</c> use it, because
    /// only those two are what a majority-class predictor actually scores. The macro metrics keep the
    /// <c>1/N</c> floor and that floor is <i>correct</i> for them: always predicting one class gives
    /// recall 1 on that class and 0 on the rest, so its macro accuracy is exactly <c>1/N</c>.
    /// </param>
    public static double? GetMinimumMetricThreshold(
        string metricName, int? classCount = null, double? majorityClassRatio = null)
    {
        return metricName.ToLowerInvariant() switch
        {
            "r_squared" or "r2" => 0.0,                // Must be better than mean prediction
            "auc" or "area_under_roc_curve" => BinaryDecisionShape.ChanceAuc, // Must be better than random
            // Must beat the better of the two trivial models: random guessing (1/N) and always
            // predicting the majority class. On imbalanced data the second is far stronger — a
            // measured 99.2%-majority label makes 1/N = 0.5 meaningless, and a 98.6%-accurate model
            // that is *worse than a constant* used to promote cleanly.
            "accuracy" or "micro_accuracy" => Math.Max(
                classCount is > 1 ? 1.0 / classCount.Value : 0.0,
                majorityClassRatio ?? 0.0),
            "macro_accuracy" => classCount.HasValue && classCount.Value > 1
                ? 1.0 / classCount.Value                 // Must be better than random (1/N)
                : 0.0,
            "f1" or "f1_score" => classCount.HasValue && classCount.Value > 1
                ? 1.0 / classCount.Value                 // Must be better than random (1/N)
                : 0.0,
            "macro_f1" => classCount.HasValue && classCount.Value > 1
                ? 1.0 / classCount.Value                 // Must be better than random (1/N)
                : 0.0,
            // Question answering: a model whose answers share nothing with the truth has learned nothing.
            "char_f1" or "exact_match" => 0.0,
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
    /// The R² of a model judged on an error metric, when it is at or below 0 — no better than answering
    /// every row with the mean; otherwise <c>null</c>. An error metric (rmse, mae, mse) has no floor of
    /// its own: what counts as a small error depends on the label's scale. The regression evaluator
    /// reports R² beside it, and R² ≤ 0 is the trivial baseline the higher-is-better floors stand for.
    /// Measured: a recommendation model optimized on rmse reached production at R² −0.65.
    /// </summary>
    public static double? NoBetterThanTheMean(string metricKey, IReadOnlyDictionary<string, double> metrics)
        => IsErrorMetric(metricKey) && metrics.TryGetValue("r_squared", out var r2) && r2 <= 0 ? r2 : null;

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
    /// Detects degenerate classification models: ones whose accuracy only reflects the class
    /// balance because they learned nothing. Returns true when
    /// <list type="bullet">
    /// <item>a binary model answers one class at its decision threshold — accuracy &gt; 0.5 with
    /// F1 ≈ 0 (always negative) or negative recall ≈ 0 (always positive) — <b>and</b> its ranking
    /// does not rescue it: AUC is at or below 0.5, or no AUC was recorded; or</item>
    /// <item>a multiclass model has macro accuracy above chance but macro F1 ≈ 0.</item>
    /// </list>
    /// <para>
    /// F1 alone only sees the first binary shape (it is computed against the positive class, so an
    /// always-positive model keeps a high F1); negative recall catches the second — a live run
    /// promoted a model with accuracy 0.747, F1 0.855, AUC 0.772 that never once predicted the
    /// minority class.
    /// </para>
    /// <para>
    /// Answering one class at the threshold is not, by itself, "learned nothing". A model can rank
    /// the classes well and still sit on one side of the default 0.5 cut — measured: accuracy 0.969,
    /// F1 0, <b>AUC 0.968</b>, served for a month by a consumer that decides by score. Its predicted
    /// labels are useless, its scores are not. The model that learned nothing had AUC 0.5. AUC is
    /// threshold-free, so it is what tells the two apart: that shape is
    /// <see cref="SingleClassAtThreshold"/> — a warning, not a refusal.
    /// </para>
    /// </summary>
    public static bool IsClassificationDegenerateModel(IReadOnlyDictionary<string, double> metrics)
    {
        if (BinaryDecisionShape.AnswersOneClass(metrics) != null && !BinaryDecisionShape.RanksAboveChance(metrics))
            return true;

        // Check multiclass: macro_accuracy > random but macro_f1 == 0
        if (metrics.TryGetValue("macro_f1", out var macroF1) &&
            metrics.TryGetValue("macro_accuracy", out var macroAcc))
        {
            if (macroAcc > 0.3 && macroF1 < 0.001)
                return true;
        }

        return false;
    }

    /// <summary>
    /// Why a binary model that ranks above chance is still not usable for its predicted labels —
    /// it answers one class at its default decision threshold — or <c>null</c> when that is not
    /// its shape. The other half of <see cref="IsClassificationDegenerateModel"/>: the gate lets
    /// such a model through (its scores carry the ranking) and says this instead.
    /// </summary>
    public static string? SingleClassAtThreshold(IReadOnlyDictionary<string, double> metrics)
    {
        if (BinaryDecisionShape.AnswersOneClass(metrics) is not { } answers || !BinaryDecisionShape.RanksAboveChance(metrics))
            return null;

        return string.Create(CultureInfo.InvariantCulture,
            $"it ranks the classes (auc {metrics["auc"]:F4}) but answers {answers} for every row at its default decision threshold, so its predicted labels carry no information while its scores do — decide by its probability with a threshold chosen for the class you need");
    }
}
