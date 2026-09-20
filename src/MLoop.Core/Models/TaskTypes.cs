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
    /// Spellings that are not the canonical name but mean one of them, mapped to the name they
    /// mean. Aliases are accepted, never advertised — <see cref="All"/> and <see cref="Listed"/>
    /// stay the vocabulary a user is shown.
    /// </summary>
    /// <remarks>
    /// Two families, both already honoured somewhere in the product before this map existed.
    /// <c>classification</c> is the pre-multi-class spelling of binary classification, which
    /// training, sampling and evaluation all still accept. The run-together forms come from a
    /// diagnostics path that normalized by deleting hyphens and spaces, which made it the only
    /// place in the product where <c>binary classification</c> worked.
    /// </remarks>
    private static readonly IReadOnlyDictionary<string, string> Aliases =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["classification"] = "binary-classification",
            ["binaryclassification"] = "binary-classification",
            ["multiclassclassification"] = "multiclass-classification",
            ["anomalydetection"] = "anomaly-detection",
            ["timeseriesanomaly"] = "time-series-anomaly",
            ["imageclassification"] = "image-classification",
            ["objectdetection"] = "object-detection",
            ["textclassification"] = "text-classification",
            ["sentencesimilarity"] = "sentence-similarity",
            ["questionanswering"] = "question-answering",
        };

    /// <summary>
    /// The canonical name for <paramref name="task"/>, or <c>null</c> when it names no task MLoop
    /// knows. Tolerates surrounding space, any casing, and underscores or run-together words where
    /// the canonical name has hyphens.
    /// </summary>
    /// <remarks>
    /// <para>This is the answer to a question that had four different answers. Every predicate that
    /// asked "which task is this" normalized the string itself, and the four forms in use were:
    /// nothing at all (an ordinal <c>is</c> pattern, so <c>Binary-Classification</c> failed),
    /// <c>ToLowerInvariant()</c>, <c>OrdinalIgnoreCase</c>, and lowercase-with-hyphens-and-spaces
    /// deleted. The two authorities that already existed disagreed with each other: this type's
    /// <see cref="IsValid"/> trimmed but did not fold underscores, while
    /// <c>AutoMLRunner.RequiresLabel</c> folded underscores but did not trim.</para>
    /// <para>It reached users. <c>task: classification</c> was rejected by <c>mloop validate</c>
    /// and accepted as binary by <c>train</c> and <c>evaluate</c>; <c>task: binary_classification</c>
    /// split the other way.</para>
    /// <para>Unknown returns <c>null</c> rather than an invented default — the same contract
    /// <c>MetricNames.Canonical</c> holds, so a caller decides what to do about not knowing
    /// instead of being handed a guess.</para>
    /// </remarks>
    public static string? Canonical(string? task)
    {
        if (string.IsNullOrWhiteSpace(task))
            return null;

        var folded = task.Trim().ToLowerInvariant().Replace('_', '-');

        if (Lookup.TryGetValue(folded, out var exact))
            return exact;

        // Run-together and spaced forms: `binary classification`, `BinaryClassification`.
        var collapsed = folded.Replace("-", string.Empty).Replace(" ", string.Empty);
        return Aliases.TryGetValue(collapsed, out var aliased) ? aliased : null;
    }

    /// <summary>
    /// Whether <paramref name="task"/> names an accepted task, in any spelling
    /// <see cref="Canonical"/> accepts.
    /// </summary>
    public static bool IsValid(string? task) => Canonical(task) != null;

    /// <summary>
    /// The vocabulary as one comma-separated line, for help text and error messages that have to
    /// name every accepted value. Generated rather than transcribed — a transcription is the
    /// fourth copy this type exists to prevent.
    /// </summary>
    public static string Listed => string.Join(", ", All);
}
