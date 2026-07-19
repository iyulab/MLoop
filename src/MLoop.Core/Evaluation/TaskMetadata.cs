namespace MLoop.Core.Evaluation;

/// <summary>
/// Single source of truth for per-ML.NET-task metadata. Converges the task→primary-metric
/// mapping that previously drifted across separate switch statements
/// (<c>ModelRegistry.DefaultMetricForTask</c>, <c>InitCommand</c>'s yaml template, and
/// <c>TrainingEngine.GetPrimaryMetricValue</c>), plus the <c>ValidateCommand</c> metric
/// allowlist. That duplication produced BUG-46 (init wrote "auto" for a task that has a
/// canonical metric, so the promotion gate silently skipped) and F-17 (validate's allowlist
/// fell out of sync and flagged init's own metrics as "Unknown"). Add a task here once and
/// every consumer stays consistent (TD-06 / D4).
/// <para>
/// <c>ConfigMerger</c> joined the consumer list late: it filled the metric default from
/// <c>ConfigDefaults.DefaultMetric</c> ("auto") without consulting this map even though it had
/// already resolved the task, so any model lacking an explicit <c>training.metric</c> in
/// mloop.yaml persisted the unresolved sentinel as its experiment's metric name. Resolve the
/// sentinel wherever a task is known — the sentinel is a request, never a recorded result.
/// </para>
/// <para>
/// Lives in <c>MLoop.Core.Evaluation</c> alongside <see cref="MetricDirection"/> so the two
/// halves of ML-metric knowledge — a task's canonical metric <i>name</i> here, that metric's
/// optimization <i>direction</i> there — share one home and both <c>MLoop.CLI</c> and
/// <c>MLoop.Ops</c> converge on it, rather than the name half being marooned in CLI.
/// </para>
/// </summary>
public static class TaskMetadata
{
    /// <summary>
    /// task (normalized) → canonical primary metric. A task absent from this map has no scalar
    /// primary metric with a defined threshold (object detection's mAP, unknown tasks); consumers
    /// treat that as the deferred "auto" / gate-skip case.
    /// </summary>
    private static readonly IReadOnlyDictionary<string, string> PrimaryMetrics =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["binary-classification"] = "accuracy",
            ["multiclass-classification"] = "macro_accuracy",
            ["image-classification"] = "micro_accuracy",
            ["text-classification"] = "micro_accuracy",
            ["regression"] = "r_squared",
            ["anomaly-detection"] = "auc",
            ["clustering"] = "average_distance",
            ["ranking"] = "ndcg",
            ["forecasting"] = "mae",
            ["time-series-anomaly"] = "detection_rate",
            ["recommendation"] = "rmse",
        };

    /// <summary>
    /// The canonical metric a task optimizes when none is explicitly chosen, or <c>null</c> for
    /// tasks without a thresholded scalar primary (object detection, unknown). Input is trimmed
    /// and matched case-insensitively.
    /// </summary>
    public static string? PrimaryMetric(string? task)
        => task != null && PrimaryMetrics.TryGetValue(task.Trim(), out var metric) ? metric : null;

    /// <summary>
    /// The task's primary metric, or the literal <c>"auto"</c> when none is defined — the value
    /// the init yaml template emits so AutoML defers metric selection.
    /// </summary>
    public static string PrimaryMetricOrAuto(string? task) => PrimaryMetric(task) ?? "auto";

    /// <summary>
    /// All distinct canonical primary metrics across tasks. Metric allowlists (e.g. validate)
    /// union this so adding a task can never again leave them out of sync — the F-17 drift.
    /// </summary>
    public static IEnumerable<string> AllPrimaryMetrics => PrimaryMetrics.Values.Distinct();

    /// <summary>
    /// Resolves "which number is this experiment's score" from a metrics dictionary — the single
    /// source of truth co-located with the primary-metric <i>names</i> above. Resolution order:
    /// the explicitly-configured <paramref name="metricName"/> if present, then the task's canonical
    /// primary metric (0 when that metric is defined for the task but absent from the dictionary —
    /// a degenerate result), then the first available value for tasks with no defined primary.
    /// Returns <c>null</c> only when the dictionary is null or empty.
    /// <para>
    /// Bypassing this (e.g. <c>Metrics.Values.FirstOrDefault()</c>) makes the reported score depend
    /// on dictionary insertion order rather than the metric the experiment actually optimized — the
    /// F-28 residual that left <c>ExperimentSummary.BestMetric</c> disagreeing with its own
    /// <c>MetricName</c>, so ranking compared apples to oranges.
    /// </para>
    /// </summary>
    public static double? ResolvePrimaryMetricValue(
        IReadOnlyDictionary<string, double>? metrics, string? metricName, string? task)
    {
        if (metrics is null || metrics.Count == 0)
            return null;

        if (metricName != null && metrics.TryGetValue(metricName, out var configured))
            return configured;

        var primary = PrimaryMetric(task);
        return primary != null
            ? metrics.GetValueOrDefault(primary, 0)
            : metrics.Values.First();
    }
}
