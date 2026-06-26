namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Single source of truth for per-ML.NET-task metadata. Converges the task→primary-metric
/// mapping that previously drifted across separate switch statements
/// (<c>ModelRegistry.DefaultMetricForTask</c>, <c>InitCommand</c>'s yaml template, and
/// <c>TrainingEngine.GetPrimaryMetricValue</c>), plus the <c>ValidateCommand</c> metric
/// allowlist. That duplication produced BUG-46 (init wrote "auto" for a task that has a
/// canonical metric, so the promotion gate silently skipped) and F-17 (validate's allowlist
/// fell out of sync and flagged init's own metrics as "Unknown"). Add a task here once and
/// every consumer stays consistent (TD-06 / D4).
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
}
