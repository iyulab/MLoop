namespace MLoop.Core.Evaluation;

/// <summary>
/// Single source of truth for per-ML.NET-task metadata. Converges the task→primary-metric
/// mapping that previously drifted across separate switch statements
/// (<c>ModelRegistry.DefaultMetricForTask</c>, <c>InitCommand</c>'s yaml template, and
/// <c>TrainingEngine.GetPrimaryMetricValue</c>), plus the <c>ValidateCommand</c> metric
/// allowlist. That duplication produced two defects: init wrote "auto" for a task that has a
/// canonical metric, so the promotion gate silently skipped it; and validate's allowlist fell
/// out of sync and flagged init's own metrics as "Unknown". Add a task here once and every
/// consumer stays consistent.
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
    /// primary metric (unknown tasks); consumers treat that as the deferred "auto" / gate-skip case.
    /// </summary>
    private static readonly IReadOnlyDictionary<string, string> PrimaryMetrics =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["binary-classification"] = "accuracy",
            ["multiclass-classification"] = "macro_accuracy",
            ["image-classification"] = "micro_accuracy",
            ["text-classification"] = "micro_accuracy",
            // The two text tasks below report these keys (DeepLearningHandlers); without an entry
            // here "auto" stayed unresolved, the quality gate had nothing to test, and a
            // sentence-similarity model with R² below zero replaced a production model at 0.73.
            ["sentence-similarity"] = "r_squared",
            ["ner"] = "macro_accuracy",
            // Character F1 over the best answer (AnswerOverlap) — the graded half of the KorQuAD pair;
            // exact match is reported beside it.
            ["question-answering"] = "char_f1",
            ["regression"] = "r_squared",
            ["anomaly-detection"] = "auc",
            ["clustering"] = "average_distance",
            ["ranking"] = "ndcg",
            ["forecasting"] = "mae",
            ["time-series-anomaly"] = "detection_rate",
            ["recommendation"] = "rmse",
            // COCO's primary score: mAP averaged over IoU 0.5–0.95, so a box must be placed well, not
            // only near the object. map_50 (PASCAL VOC) is reported beside it and can be chosen.
            ["object-detection"] = "map_50_95",
        };

    /// <summary>
    /// The canonical metric a task optimizes when none is explicitly chosen, or <c>null</c> for
    /// tasks without a scalar primary (unknown). Input is trimmed
    /// and matched case-insensitively.
    /// </summary>
    public static string? PrimaryMetric(string? task)
        => task != null && PrimaryMetrics.TryGetValue(task.Trim(), out var metric) ? metric : null;

    /// <summary>
    /// Whether a run has nothing to judge one model by: a table anomaly run without a label column fits one
    /// model and has no score to compare it with. The share of rows that model flags is the threshold's
    /// doing, not a quality - so no page may rank by it or call it the thing that was optimized.
    /// </summary>
    public static bool IsUnjudged(string? task, string? labelColumn)
        => string.Equals(task?.Trim(), "anomaly-detection", StringComparison.OrdinalIgnoreCase)
           && string.IsNullOrWhiteSpace(labelColumn);

    /// <summary>
    /// The task's primary metric, or the literal <c>"auto"</c> when none is defined — the value
    /// the init yaml template emits so AutoML defers metric selection.
    /// </summary>
    public static string PrimaryMetricOrAuto(string? task) => PrimaryMetric(task) ?? "auto";

    /// <summary>
    /// The tasks whose answer is a number on a scale — what a user's feedback can only be <i>near</i>, never <i>the same as</i>:
    /// a regression target, a similarity score, a forecast, a predicted rating, a rank score. The classes, tags and answer
    /// texts of the other tasks are matched; these are judged by an interval the answer states or by an error, and a
    /// numeric <i>class</i> label (<c>1</c> / <c>2</c>) belongs to a classification task, not here.
    /// </summary>
    private static readonly HashSet<string> ValueTasks = new(StringComparer.OrdinalIgnoreCase)
    {
        "regression", "sentence-similarity", "forecasting", "recommendation", "ranking",
    };

    /// <summary>
    /// Whether <paramref name="task"/> answers with a value (see above) — the one place that says so. Feedback accuracy
    /// reads "the same value" for everything else; for these it is not a measure (≈ 0 for every model, however good),
    /// so a value answer that states no interval has no accuracy. Trimmed, case-insensitive; an unknown task or none is
    /// <c>false</c>, which keeps the reading it has always had.
    /// </summary>
    public static bool AnswersAreValues(string? task) => task is not null && ValueTasks.Contains(task.Trim());

    /// <summary>
    /// Whether an answer of <paramref name="task"/> can be named by feedback — whether it carries a prediction id. A forecast does not: it is the
    /// horizon after the training series, made in one stateful replay (<c>ForecastReplayService</c>), and neither <c>mloop predict --log</c> nor
    /// <c>POST /predict</c> logs it or returns an id (measured on a trained forecasting model: the log stays empty and <c>feedback add</c> finds no
    /// prediction). A condition that reads feedback has nothing to read for such a task, so it does not apply — saying "no data" would promise data
    /// that cannot arrive. Trimmed, case-insensitive; an unknown task or none is <c>true</c>, which keeps the reading it has always had.
    /// </summary>
    public static bool TakesFeedback(string? task) => !string.Equals(task?.Trim(), "forecasting", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// All distinct canonical primary metrics across tasks. Metric allowlists (e.g. validate)
    /// union this so adding a task can never again leave them out of sync.
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
    /// The residual that left <c>ExperimentSummary.BestMetric</c> disagreeing with its own
    /// <c>MetricName</c>, so ranking compared apples to oranges.
    /// </para>
    /// </summary>
    public static double? ResolvePrimaryMetricValue(
        IReadOnlyDictionary<string, double>? metrics, string? metricName, string? task)
        => ResolvePrimaryMetric(metrics, metricName, task)?.Value;

    /// <summary>
    /// The metric <see cref="ResolvePrimaryMetricValue"/> reads, together with its name — so a
    /// record that stores the value can name what it holds. The configured name is read the way the
    /// optimizer reads it (<see cref="MetricNames.Canonical(string?, string?)"/>): <c>f1</c> is
    /// <c>f1_score</c> and <c>auto</c> is the task's primary metric. Recording the configured
    /// spelling beside a value resolved from another key labelled an accuracy of 0.99 as
    /// <c>f1</c> on a model whose F1 was 0.
    /// </summary>
    public static (string Name, double Value)? ResolvePrimaryMetric(
        IReadOnlyDictionary<string, double>? metrics, string? metricName, string? task)
    {
        if (metrics is null || metrics.Count == 0)
            return null;

        var configured = MetricNames.Canonical(task, metricName);
        if (configured != null && metrics.TryGetValue(configured, out var canonicalValue))
            return (configured, canonicalValue);

        if (metricName != null && metrics.TryGetValue(metricName, out var literalValue))
            return (metricName, literalValue);

        var primary = PrimaryMetric(task);
        if (primary != null)
            return (primary, metrics.GetValueOrDefault(primary, 0));

        var first = metrics.First();
        return (first.Key, first.Value);
    }
}
