using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Evaluation;

/// <summary>
/// The confusion counts of binary predictions, and the metrics they define on their own.
/// </summary>
/// <remarks>
/// ML.NET's binary evaluators compute AUC together with everything else and throw when the data
/// holds a single class, where AUC is undefined. A subset of the data — one device, one log type —
/// is often exactly that, and on it the question asked is "how many of these did the model get
/// wrong", which needs no AUC. These counts answer it; a metric whose denominator is zero is left
/// out rather than reported as 0.
/// </remarks>
public static class BinaryCounts
{
    /// <summary>
    /// Accuracy, precision, recall, negative recall and F1 from <c>PredictedLabel</c> against the
    /// boolean label, each present only when defined. <c>null</c> when either column is missing.
    /// </summary>
    public static Dictionary<string, double>? Metrics(IDataView predictions, string labelColumn)
    {
        var predicted = predictions.Schema.GetColumnOrNull("PredictedLabel");
        var label = predictions.Schema.GetColumnOrNull(labelColumn);
        if (predicted is null || label is null
            || predicted.Value.Type != BooleanDataViewType.Instance
            || label.Value.Type != BooleanDataViewType.Instance)
            return null;

        long tp = 0, fp = 0, tn = 0, fn = 0;
        using var cursor = predictions.GetRowCursor([predicted.Value, label.Value]);
        var getPredicted = cursor.GetGetter<bool>(predicted.Value);
        var getLabel = cursor.GetGetter<bool>(label.Value);
        bool p = default, y = default;
        while (cursor.MoveNext())
        {
            getPredicted(ref p);
            getLabel(ref y);
            if (p && y) tp++;
            else if (p) fp++;
            else if (y) fn++;
            else tn++;
        }

        var total = tp + fp + tn + fn;
        if (total == 0)
            return null;

        var metrics = new Dictionary<string, double> { ["accuracy"] = (double)(tp + tn) / total };
        if (tp + fp > 0) metrics["precision"] = (double)tp / (tp + fp);
        if (tp + fn > 0) metrics["recall"] = (double)tp / (tp + fn);
        if (tn + fp > 0) metrics["negative_recall"] = (double)tn / (tn + fp);
        if (tn + fn > 0) metrics["negative_precision"] = (double)tn / (tn + fn);
        // F1 is defined whenever the rows hold a positive, even when the model predicted none: it found none of them, so
        // it is 0 — not absent, which would leave the evaluator's own F1 (counted from the raw scores) standing next to
        // answers it did not give.
        if (metrics.TryGetValue("recall", out var rc))
            metrics["f1_score"] = metrics.TryGetValue("precision", out var pr) && pr + rc > 0 ? 2 * pr * rc / (pr + rc) : 0;
        return metrics;
    }

    /// <summary>The metrics that depend on the answers a model gives rather than on its scores.</summary>
    public static readonly string[] AnswerMetrics = ["accuracy", "precision", "recall", "negative_precision", "negative_recall", "f1_score"];

    /// <summary>
    /// Replaces, in <paramref name="metrics"/> (ML.NET's evaluation, which re-derives each answer from the raw score at 0),
    /// the metrics that depend on the answers with the ones counted from the answers the model gives. One that the
    /// answers leave undefined is removed: it would otherwise stay as the raw-score value beside the counted ones
    /// (measured: <c>recall</c> 0 next to <c>precision</c> 1 and <c>f1_score</c> 1 for a model answering "no" to every row).
    /// </summary>
    public static void ReplaceAnswerMetrics(IDictionary<string, double> metrics, IReadOnlyDictionary<string, double> counted)
    {
        ArgumentNullException.ThrowIfNull(metrics);
        ArgumentNullException.ThrowIfNull(counted);
        foreach (var key in AnswerMetrics)
        {
            if (counted.TryGetValue(key, out var value)) metrics[key] = value;
            else metrics.Remove(key);
        }
    }
}
