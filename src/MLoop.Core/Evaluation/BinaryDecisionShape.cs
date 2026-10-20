namespace MLoop.Core.Evaluation;

/// <summary>
/// What a binary classifier's recorded metrics say about the shape of its decisions: whether it
/// answers one class for every row at its decision threshold, and whether its scores rank the
/// classes better than chance. The one statement of both — the promotion gate and the training
/// diagnostics read it, so a model is not "degenerate" in one and "usable" in the other.
/// </summary>
/// <remarks>
/// The two are independent. A model can rank well and still have every score on one side of the
/// default 0.5 cut — measured: accuracy 0.969, F1 0, AUC 0.968, served for a month by a consumer
/// that decides by score. Its predicted labels carry nothing; its scores do. A model that learned
/// nothing answers one class too, and ranks like chance (AUC 0.5).
/// </remarks>
public static class BinaryDecisionShape
{
    /// <summary>AUC of a ranking no better than chance — the floor a model must clear to rank at all.</summary>
    public const double ChanceAuc = 0.5;

    /// <summary>
    /// The class a binary model answers for every row — <c>"negative"</c> or <c>"positive"</c> — or
    /// <c>null</c> when it answers both. Read from accuracy above 0.5 together with F1 ≈ 0 (no
    /// positive is ever predicted) or negative recall ≈ 0 (no negative is). F1 alone only sees the
    /// first: it is computed on the positive class, so an always-positive model keeps a high F1.
    /// </summary>
    public static string? AnswersOneClass(IReadOnlyDictionary<string, double> metrics)
    {
        if (!metrics.TryGetValue("accuracy", out var acc) || acc <= 0.5)
            return null;

        if (metrics.TryGetValue("f1_score", out var f1) && f1 < 0.001)
            return "negative";

        return metrics.TryGetValue("negative_recall", out var negRecall) && negRecall < 0.001
            ? "positive"
            : null;
    }

    /// <summary>
    /// True when the model's scores rank the classes better than chance (AUC above
    /// <see cref="ChanceAuc"/>). False when no AUC was recorded — nothing shows that it ranks.
    /// </summary>
    public static bool RanksAboveChance(IReadOnlyDictionary<string, double> metrics)
        => metrics.TryGetValue("auc", out var auc) && auc > ChanceAuc;
}
