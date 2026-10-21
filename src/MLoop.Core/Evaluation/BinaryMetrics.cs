using Microsoft.ML.Data;

namespace MLoop.Core.Evaluation;

/// <summary>
/// A binary classifier's metrics from ML.NET's evaluator, in MLoop's vocabulary — the one place the
/// field-to-name mapping lives, read by training (the final result, each trial, the small-data
/// fallback) and by <c>evaluate</c>.
/// </summary>
/// <remarks>
/// It used to be written out four times, and the copies disagreed: training reported neither
/// <c>auprc</c> nor <c>negative_precision</c>, both of which <c>train --metric</c> accepts, so a run
/// optimizing either recorded no value for the metric it optimized. The set is
/// <see cref="MetricNames.BinaryFamily"/> — a name the optimizer takes is a name the report carries.
/// A value ML.NET cannot compute (an AUC over one class) is absent rather than <c>NaN</c>.
/// </remarks>
public static class BinaryMetrics
{
    /// <summary>The metrics of <paramref name="m"/>, keyed by canonical name; non-finite values omitted.</summary>
    public static Dictionary<string, double> Describe(BinaryClassificationMetrics m)
    {
        ArgumentNullException.ThrowIfNull(m);
        var all = new Dictionary<string, double>
        {
            ["accuracy"] = m.Accuracy,
            ["auc"] = m.AreaUnderRocCurve,
            ["auprc"] = m.AreaUnderPrecisionRecallCurve,
            ["f1_score"] = m.F1Score,
            ["precision"] = m.PositivePrecision,
            ["recall"] = m.PositiveRecall,
            ["negative_precision"] = m.NegativePrecision,
            ["negative_recall"] = m.NegativeRecall,
        };
        return all.Where(kv => double.IsFinite(kv.Value)).ToDictionary(kv => kv.Key, kv => kv.Value);
    }
}
