namespace MLoop.Core.Models;

/// <summary>
/// Single source of truth for the ML task vocabulary — the exact set of values
/// <c>mloop.yaml</c>'s <c>task</c> field and the <c>--task</c> option accept.
/// </summary>
/// <remarks>
/// <para>
/// The vocabulary was previously written out three times: <c>InitCommand</c>'s scaffolding
/// allowlist, <c>ValidateCommand</c>'s config allowlist, and the <c>--task</c> option's help text
/// on <c>train</c>. Nothing tied them together, so adding a task meant remembering three edits and
/// the failure of forgetting one is silent in the direction that hurts most — a task <c>init</c>
/// scaffolds but <c>validate</c> rejects, or one the trainer supports that help never names.
/// The three agreed when this type was extracted; the point is that agreeing was luck, not a
/// guarantee.
/// </para>
/// <para>
/// Distinct from <see cref="MLoop.Core.Evaluation.TaskMetadata"/>, which maps a task to its
/// canonical scalar metric and deliberately holds fewer entries — tasks without a thresholded
/// scalar primary (object detection's mAP, the sentence/NLP tasks) are absent there by design.
/// That map is metric knowledge; this list is the vocabulary itself, and
/// <c>TaskVocabularyContractTests</c> asserts the former stays a subset of the latter so a typo in
/// either surfaces as a failing test rather than a task that validates but never scores.
/// </para>
/// </remarks>
public static class TaskTypes
{
    /// <summary>
    /// Every accepted task, ordered by the engine family that serves it. The grouping is how the
    /// list stays readable as it grows, and it is the order help text presents.
    /// </summary>
    public static readonly IReadOnlyList<string> All =
    [
        // Supervised (AutoML)
        "regression",
        "binary-classification",
        "multiclass-classification",
        // Unsupervised / semi-supervised (direct trainer)
        "anomaly-detection",
        "clustering",
        "ranking",
        // Time series (Microsoft.ML.TimeSeries)
        "forecasting",
        "time-series-anomaly",
        // Deep learning (Microsoft.ML.Vision / TorchSharp)
        "image-classification",
        "object-detection",
        "text-classification",
        "sentence-similarity",
        "ner",
        "question-answering",
        // Collaborative filtering (Microsoft.ML.Recommender)
        "recommendation",
    ];

    private static readonly HashSet<string> Lookup = new(All, StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Whether <paramref name="task"/> names an accepted task. Trimmed and matched
    /// case-insensitively, the same tolerance a hand-edited yaml field needs.
    /// </summary>
    public static bool IsValid(string? task) =>
        task != null && Lookup.Contains(task.Trim());

    /// <summary>
    /// The vocabulary as one comma-separated line, for help text and error messages that have to
    /// name every accepted value. Generated rather than transcribed — a transcription is the
    /// fourth copy this type exists to prevent.
    /// </summary>
    public static string Listed => string.Join(", ", All);
}
