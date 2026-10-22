using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Data;

namespace MLoop.Core.Evaluation;

/// <summary>
/// Test metrics where each row counts by its weight. ML.NET's evaluators take no weights, so a model
/// fitted on weighted rows would otherwise be judged — and promoted — on the unweighted distribution
/// it was trained away from. The keys are the ones the unweighted evaluators report, so the gate and
/// every reader treat them alike; the experiment records the weight column they were computed under.
/// </summary>
/// <remarks>
/// A metric whose weighted denominator is zero is left out, as in <see cref="BinaryCounts"/>. AUC is
/// the weighted Mann–Whitney statistic: the weighted share of (positive, negative) pairs the score
/// orders correctly, ties counting half.
/// </remarks>
public static class WeightedMetrics
{
    /// <summary>
    /// Weighted binary metrics — or, with no <paramref name="weightColumn"/>, each row counting once — or
    /// <c>null</c> when the columns they need are missing. Unlike ML.NET's evaluator, rows of one class are
    /// scored: what they leave undefined (AUC, and recall or precision of the absent class) is left out.
    /// </summary>
    public static Dictionary<string, double>? Binary(IDataView predictions, string labelColumn, string? weightColumn)
    {
        var label = predictions.Schema.GetColumnOrNull(labelColumn);
        var predicted = predictions.Schema.GetColumnOrNull("PredictedLabel");
        var score = predictions.Schema.GetColumnOrNull("Score");
        var probability = predictions.Schema.GetColumnOrNull("Probability");
        if (label is null || predicted is null || score is null
            || predicted.Value.Type != BooleanDataViewType.Instance
            || score.Value.Type != NumberDataViewType.Single
            || (label.Value.Type != BooleanDataViewType.Instance && label.Value.Type != NumberDataViewType.Single))
            return null;
        var hasProbability = probability is { } pc && pc.Type == NumberDataViewType.Single;

        var weights = weightColumn is null ? null : WeightedResampling.ReadWeights(predictions, weightColumn);
        var rows = new List<(bool Y, bool P, float S, float Prob, double W)>(weights?.Length ?? 0);
        var columns = new List<DataViewSchema.Column> { label.Value, predicted.Value, score.Value };
        if (hasProbability) columns.Add(probability!.Value);
        using (var cursor = predictions.GetRowCursor(columns))
        {
            var isPositive = Positive(cursor, label.Value);
            var getPredicted = cursor.GetGetter<bool>(predicted.Value);
            var getScore = cursor.GetGetter<float>(score.Value);
            var getProbability = hasProbability ? cursor.GetGetter<float>(probability!.Value) : null;
            var i = 0;
            bool p = default;
            float s = default, prob = float.NaN;
            while (cursor.MoveNext())
            {
                getPredicted(ref p);
                getScore(ref s);
                getProbability?.Invoke(ref prob);
                rows.Add((isPositive(), p, s, prob, weights is null ? 1.0 : weights[i++]));
            }
        }

        double tp = 0, fp = 0, tn = 0, fn = 0;
        foreach (var r in rows)
        {
            if (r.P && r.Y) tp += r.W;
            else if (r.P) fp += r.W;
            else if (r.Y) fn += r.W;
            else tn += r.W;
        }

        var total = tp + fp + tn + fn;
        if (total <= 0)
            return null;

        var metrics = new Dictionary<string, double> { ["accuracy"] = (tp + tn) / total };
        if (tp + fp > 0) metrics["precision"] = tp / (tp + fp);
        if (tp + fn > 0) metrics["recall"] = tp / (tp + fn);
        if (tn + fp > 0) metrics["negative_recall"] = tn / (tn + fp);
        if (tn + fn > 0) metrics["negative_precision"] = tn / (tn + fn);
        if (metrics.TryGetValue("precision", out var pr) && metrics.TryGetValue("recall", out var rc))
            metrics["f1_score"] = pr + rc > 0 ? 2 * pr * rc / (pr + rc) : 0;
        if (Auc(rows.Where(r => !float.IsNaN(r.S)).Select(r => (r.Y, (double)r.S, r.W)).ToList()) is { } auc)
            metrics["auc"] = auc;
        if (hasProbability && Ece(rows.Where(r => !float.IsNaN(r.Prob)).Select(r => ((double)r.Prob, r.Y, r.W)).ToList()) is { } ece)
            metrics[CalibrationError.MetricName] = ece;
        return metrics;
    }

    /// <summary>Weighted multiclass metrics, or <c>null</c> when the columns they need are missing.</summary>
    public static Dictionary<string, double>? Multiclass(IDataView predictions, string labelColumn, string weightColumn)
    {
        var score = predictions.Schema.GetColumnOrNull("Score");
        var label = predictions.Schema.GetColumnOrNull(labelColumn);
        if (score is null || label is null
            || score.Value.Type is not VectorDataViewType { ItemType: NumberDataViewType itemType } vector
            || itemType != NumberDataViewType.Single
            || label.Value.Type is not KeyDataViewType key
            || vector.Size != (int)key.Count)
            return null;

        var classes = vector.Size;
        var weights = WeightedResampling.ReadWeights(predictions, weightColumn);
        var classWeight = new double[classes];
        var classHit = new double[classes];
        var predictedWeight = new double[classes];
        var predictedHit = new double[classes];
        double total = 0, hit = 0, logLoss = 0;
        var calibration = new List<(double, bool, double)>();

        using (var cursor = predictions.GetRowCursor([score.Value, label.Value]))
        {
            var getScore = cursor.GetGetter<VBuffer<float>>(score.Value);
            var getLabel = cursor.GetGetter<uint>(label.Value);
            VBuffer<float> scores = default;
            uint labelKey = default;
            var i = 0;
            while (cursor.MoveNext())
            {
                getScore(ref scores);
                getLabel(ref labelKey);
                var w = weights[i++];
                if (labelKey == 0 || w == 0)
                    continue; // missing label, or a row that counts for nothing

                var values = scores.DenseValues().ToArray();
                var top = 0;
                for (var c = 1; c < values.Length; c++)
                    if (values[c] > values[top])
                        top = c;
                var truth = (int)labelKey - 1;
                var correct = top == truth;

                total += w;
                classWeight[truth] += w;
                predictedWeight[top] += w;
                if (correct)
                {
                    hit += w;
                    classHit[truth] += w;
                    predictedHit[top] += w;
                }
                // Clamped as ML.NET's evaluator does, so one confident miss does not make the sum infinite.
                logLoss -= w * Math.Log(Math.Max(values[truth], 1e-15));
                if (!float.IsNaN(values[top]))
                    calibration.Add((values[top], correct, w));
            }
        }

        if (total <= 0)
            return null;

        var present = Enumerable.Range(0, classes).Where(c => classWeight[c] > 0).ToArray();
        double f1Sum = 0;
        foreach (var c in present)
        {
            var recall = classHit[c] / classWeight[c];
            var precision = predictedWeight[c] > 0 ? predictedHit[c] / predictedWeight[c] : 0;
            f1Sum += precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
        }

        var metrics = new Dictionary<string, double>
        {
            ["micro_accuracy"] = hit / total,
            ["macro_accuracy"] = present.Average(c => classHit[c] / classWeight[c]),
            ["log_loss"] = logLoss / total,
            ["macro_f1"] = f1Sum / present.Length,
        };
        if (Ece(calibration) is { } ece)
            metrics[CalibrationError.MetricName] = ece;
        return metrics;
    }

    /// <summary>Weighted regression metrics, or <c>null</c> when the columns they need are missing.</summary>
    public static Dictionary<string, double>? Regression(IDataView predictions, string labelColumn, string weightColumn)
    {
        var label = predictions.Schema.GetColumnOrNull(labelColumn);
        var score = predictions.Schema.GetColumnOrNull("Score");
        if (label is null || score is null
            || label.Value.Type != NumberDataViewType.Single || score.Value.Type != NumberDataViewType.Single)
            return null;

        var weights = WeightedResampling.ReadWeights(predictions, weightColumn);
        var rows = new List<(double Y, double S, double W)>(weights.Length);
        using (var cursor = predictions.GetRowCursor([label.Value, score.Value]))
        {
            var getLabel = cursor.GetGetter<float>(label.Value);
            var getScore = cursor.GetGetter<float>(score.Value);
            float y = default, s = default;
            var i = 0;
            while (cursor.MoveNext())
            {
                getLabel(ref y);
                getScore(ref s);
                var w = weights[i++];
                if (w > 0 && float.IsFinite(y) && float.IsFinite(s))
                    rows.Add((y, s, w));
            }
        }

        var total = rows.Sum(r => r.W);
        if (total <= 0)
            return null;
        var mean = rows.Sum(r => r.W * r.Y) / total;
        var sse = rows.Sum(r => r.W * (r.Y - r.S) * (r.Y - r.S));
        var sst = rows.Sum(r => r.W * (r.Y - mean) * (r.Y - mean));
        var mse = sse / total;
        var metrics = new Dictionary<string, double>
        {
            ["mse"] = mse,
            ["rmse"] = Math.Sqrt(mse),
            ["mae"] = rows.Sum(r => r.W * Math.Abs(r.Y - r.S)) / total,
        };
        // ML.NET reports R² 0 for a constant label; undefined here, and left out like any 0/0.
        if (sst > 0)
            metrics["r_squared"] = 1 - sse / sst;
        return metrics;
    }

    /// <summary>Weighted AUC, or <c>null</c> when one of the classes has no weight.</summary>
    public static double? Auc(IReadOnlyList<(bool Positive, double Score, double Weight)> rows)
    {
        var positive = rows.Where(r => r.Positive).Sum(r => r.Weight);
        var negative = rows.Where(r => !r.Positive).Sum(r => r.Weight);
        if (positive <= 0 || negative <= 0)
            return null;

        // Ascending by score; each tie group: its positives beat every negative below, and half of its own.
        double negativesBelow = 0, concordant = 0;
        foreach (var group in rows.GroupBy(r => r.Score).OrderBy(g => g.Key))
        {
            var groupPositive = group.Where(r => r.Positive).Sum(r => r.Weight);
            var groupNegative = group.Where(r => !r.Positive).Sum(r => r.Weight);
            concordant += groupPositive * (negativesBelow + 0.5 * groupNegative);
            negativesBelow += groupNegative;
        }
        return concordant / (positive * negative);
    }

    /// <summary>
    /// <see cref="CalibrationError"/>'s ten-bin ECE with each row counting by its weight — rows are
    /// (confidence, whether it was right, weight). <c>null</c> when no row has weight.
    /// </summary>
    public static double? Ece(IReadOnlyList<(double Confidence, bool Hit, double Weight)> rows)
    {
        var total = rows.Sum(r => r.Weight);
        if (total <= 0)
            return null;
        var bins = CalibrationError.Bins;
        var weight = new double[bins];
        var confidence = new double[bins];
        var hits = new double[bins];
        foreach (var (p, hit, w) in rows)
        {
            var bin = Math.Clamp((int)(Math.Clamp(p, 0, 1) * bins), 0, bins - 1);
            weight[bin] += w;
            confidence[bin] += w * p;
            if (hit) hits[bin] += w;
        }
        double ece = 0;
        for (var b = 0; b < bins; b++)
            if (weight[b] > 0)
                ece += weight[b] / total * Math.Abs(hits[b] / weight[b] - confidence[b] / weight[b]);
        return ece;
    }

    private static Func<bool> Positive(DataViewRowCursor cursor, DataViewSchema.Column label)
    {
        if (label.Type == BooleanDataViewType.Instance)
        {
            var get = cursor.GetGetter<bool>(label);
            return () => { bool v = default; get(ref v); return v; };
        }
        var getFloat = cursor.GetGetter<float>(label);
        return () => { float v = default; getFloat(ref v); return v > 0.5f; };
    }
}
