using Microsoft.ML.Data;

namespace MLoop.Core.Evaluation;

/// <summary>
/// A model's metrics from ML.NET's evaluator, in MLoop's vocabulary — the one place each task
/// family's field-to-name mapping lives, read by training (the final result, each trial, the
/// small-data fallback) and by <c>evaluate</c>.
/// </summary>
/// <remarks>
/// It used to be written out per call site, and the copies disagreed: binary training reported
/// neither <c>auprc</c> nor <c>negative_precision</c> and multiclass training not
/// <c>log_loss_reduction</c>, all names <c>train --metric</c> accepts — so a run optimizing one of
/// them recorded no value for the metric it optimized. Each family's set is the optimizer's own
/// vocabulary (<see cref="MetricNames.OptimizableFor"/>): a name the optimizer takes is a name the
/// report carries. Values are as ML.NET gives them, non-finite included: what a non-finite value
/// means is the caller's policy (<c>evaluate</c> sanitizes it to the direction's worst through
/// <see cref="MetricSanitizer"/>; training leaves out a ranking metric one class makes undefined).
/// </remarks>
public static class MetricReport
{
    /// <summary>A binary classifier's metrics, keyed by canonical name.</summary>
    public static Dictionary<string, double> Describe(BinaryClassificationMetrics m)
    {
        ArgumentNullException.ThrowIfNull(m);
        return new()
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
    }

    /// <summary>A multiclass classifier's metrics, keyed by canonical name.</summary>
    public static Dictionary<string, double> Describe(MulticlassClassificationMetrics m)
    {
        ArgumentNullException.ThrowIfNull(m);
        return new()
        {
            ["macro_accuracy"] = m.MacroAccuracy,
            ["micro_accuracy"] = m.MicroAccuracy,
            ["log_loss"] = m.LogLoss,
            ["log_loss_reduction"] = m.LogLossReduction,
        };
    }

    /// <summary>
    /// A clustering model's metrics, keyed by canonical name. <c>average_distance</c> is as ML.NET gives it (the caller's
    /// policy decides what a non-finite one means); the other two are present only when ML.NET could compute them. It
    /// answers 0 for a Davies-Bouldin index when fewer than two clusters hold a scored row (the index of two distinct
    /// clusters is above 0, and 0 would read as the best possible), and NaN for a normalized mutual information when the
    /// rows carry no label — which no unlabeled clustering run has.
    /// </summary>
    public static Dictionary<string, double> Describe(ClusteringMetrics m)
    {
        ArgumentNullException.ThrowIfNull(m);
        var metrics = new Dictionary<string, double> { ["average_distance"] = m.AverageDistance };
        if (double.IsFinite(m.DaviesBouldinIndex) && m.DaviesBouldinIndex > 0)
            metrics["davies_bouldin_index"] = m.DaviesBouldinIndex;
        if (double.IsFinite(m.NormalizedMutualInformation))
            metrics["normalized_mutual_information"] = m.NormalizedMutualInformation;
        return metrics;
    }

    /// <summary>A regression model's metrics, keyed by canonical name.</summary>
    public static Dictionary<string, double> Describe(RegressionMetrics m)
    {
        ArgumentNullException.ThrowIfNull(m);
        return new()
        {
            ["r_squared"] = m.RSquared,
            ["rmse"] = m.RootMeanSquaredError,
            ["mae"] = m.MeanAbsoluteError,
            ["mse"] = m.MeanSquaredError,
        };
    }
}
