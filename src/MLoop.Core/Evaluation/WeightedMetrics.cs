using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Data;

namespace MLoop.Core.Evaluation;

/// <summary>
/// Scored test rows held in memory, ready to be turned into metrics under any set of row weights. Reading
/// the rows out of an <see cref="IDataView"/> is done once; computing the metrics from them is a pure
/// function of the weights, so one set of rows can be measured as given (<see cref="MetricRowsExtensions.Compute(IMetricRows)"/>) or
/// under resampled weights (a bootstrap) without a second metric implementation.
/// </summary>
public interface IMetricRows
{
    /// <summary>Rows held (including rows that count for nothing under the base weights).</summary>
    int Count { get; }

    /// <summary>The weight each row was read with — 1 when no weight column was given.</summary>
    double[] Weights { get; }

    /// <summary>The metrics of these rows under <paramref name="weights"/> (one per row), or <c>null</c> when nothing counts.</summary>
    Dictionary<string, double>? Compute(double[] weights);
}

/// <summary><see cref="IMetricRows"/> helpers.</summary>
public static class MetricRowsExtensions
{
    /// <summary>The metrics of the rows under the weights they were read with.</summary>
    public static Dictionary<string, double>? Compute(this IMetricRows rows) => rows.Compute(rows.Weights);
}

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
        => ReadBinary(predictions, labelColumn, weightColumn)?.Compute();

    /// <summary>The scored rows behind <see cref="Binary"/>, or <c>null</c> when the columns it needs are missing.</summary>
    public static IMetricRows? ReadBinary(IDataView predictions, string labelColumn, string? weightColumn)
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

        var weights = weightColumn is null ? null : RowWeights.Read(predictions, weightColumn);
        var rows = new List<(bool Y, bool P, float S, float Prob)>(weights?.Length ?? 0);
        var columns = new List<DataViewSchema.Column> { label.Value, predicted.Value, score.Value };
        if (hasProbability) columns.Add(probability!.Value);
        using (var cursor = predictions.GetRowCursor(columns))
        {
            var isPositive = Positive(cursor, label.Value);
            var getPredicted = cursor.GetGetter<bool>(predicted.Value);
            var getScore = cursor.GetGetter<float>(score.Value);
            var getProbability = hasProbability ? cursor.GetGetter<float>(probability!.Value) : null;
            bool p = default;
            float s = default, prob = float.NaN;
            while (cursor.MoveNext())
            {
                getPredicted(ref p);
                getScore(ref s);
                getProbability?.Invoke(ref prob);
                rows.Add((isPositive(), p, s, prob));
            }
        }

        return new BinaryRows(
            [.. rows.Select(r => r.Y)], [.. rows.Select(r => r.P)],
            [.. rows.Select(r => r.S)], [.. rows.Select(r => r.Prob)],
            weights ?? Ones(rows.Count), hasProbability);
    }

    private sealed class BinaryRows(bool[] y, bool[] p, float[] s, float[] prob, double[] weights, bool hasProbability) : IMetricRows
    {
        public int Count => y.Length;

        public double[] Weights => weights;

        public Dictionary<string, double>? Compute(double[] w)
        {
            double tp = 0, fp = 0, tn = 0, fn = 0;
            for (var i = 0; i < y.Length; i++)
            {
                if (p[i] && y[i]) tp += w[i];
                else if (p[i]) fp += w[i];
                else if (y[i]) fn += w[i];
                else tn += w[i];
            }

            var total = tp + fp + tn + fn;
            if (total <= 0)
                return null;

            var metrics = new Dictionary<string, double> { ["accuracy"] = (tp + tn) / total };
            if (tp + fp > 0) metrics["precision"] = tp / (tp + fp);
            if (tp + fn > 0) metrics["recall"] = tp / (tp + fn);
            if (tn + fp > 0) metrics["negative_recall"] = tn / (tn + fp);
            if (tn + fn > 0) metrics["negative_precision"] = tn / (tn + fn);
            // As in BinaryCounts: defined whenever a positive is in the rows, 0 when the model predicted none of them.
            if (metrics.TryGetValue("recall", out var rc))
                metrics["f1_score"] = metrics.TryGetValue("precision", out var pr) && pr + rc > 0 ? 2 * pr * rc / (pr + rc) : 0;

            var scored = new List<(bool Positive, double Score, double Weight)>(y.Length);
            for (var i = 0; i < y.Length; i++)
                if (!float.IsNaN(s[i]))
                    scored.Add((y[i], s[i], w[i]));
            if (Auc(scored) is { } auc)
                metrics["auc"] = auc;

            if (hasProbability)
            {
                var calibrated = new List<(double Confidence, bool Hit, double Weight)>(y.Length);
                for (var i = 0; i < y.Length; i++)
                    if (!float.IsNaN(prob[i]))
                        calibrated.Add((prob[i], y[i], w[i]));
                if (Ece(calibrated) is { } ece)
                    metrics[CalibrationError.MetricName] = ece;
            }
            return metrics;
        }
    }

    /// <summary>Weighted multiclass metrics, or <c>null</c> when the columns they need are missing.</summary>
    public static Dictionary<string, double>? Multiclass(IDataView predictions, string labelColumn, string weightColumn)
        => ReadMulticlass(predictions, labelColumn, weightColumn)?.Compute();

    /// <summary>The scored rows behind <see cref="Multiclass"/>, or <c>null</c> when the columns it needs are missing.</summary>
    public static IMetricRows? ReadMulticlass(IDataView predictions, string labelColumn, string? weightColumn)
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
        var truths = new List<int>();
        var tops = new List<int>();
        var topScores = new List<float>();
        var truthLogs = new List<double>();
        using (var cursor = predictions.GetRowCursor([score.Value, label.Value]))
        {
            var getScore = cursor.GetGetter<VBuffer<float>>(score.Value);
            var getLabel = cursor.GetGetter<uint>(label.Value);
            VBuffer<float> scores = default;
            uint labelKey = default;
            while (cursor.MoveNext())
            {
                getScore(ref scores);
                getLabel(ref labelKey);
                if (labelKey == 0)
                {
                    // Missing label: the row never counts, whatever its weight.
                    truths.Add(-1);
                    tops.Add(0);
                    topScores.Add(float.NaN);
                    truthLogs.Add(0);
                    continue;
                }

                var values = scores.DenseValues().ToArray();
                var top = 0;
                for (var c = 1; c < values.Length; c++)
                    if (values[c] > values[top])
                        top = c;
                var truth = (int)labelKey - 1;
                truths.Add(truth);
                tops.Add(top);
                topScores.Add(values[top]);
                // Clamped as ML.NET's evaluator does, so one confident miss does not make the sum infinite.
                truthLogs.Add(Math.Log(Math.Max(values[truth], 1e-15)));
            }
        }

        var weights = weightColumn is null ? Ones(truths.Count) : RowWeights.Read(predictions, weightColumn);
        return new MulticlassRows(classes, [.. truths], [.. tops], [.. topScores], [.. truthLogs], weights);
    }

    private sealed class MulticlassRows(int classes, int[] truth, int[] top, float[] topScore, double[] truthLog, double[] weights) : IMetricRows
    {
        public int Count => truth.Length;

        public double[] Weights => weights;

        public Dictionary<string, double>? Compute(double[] w)
        {
            var classWeight = new double[classes];
            var classHit = new double[classes];
            var predictedWeight = new double[classes];
            var predictedHit = new double[classes];
            double total = 0, hit = 0, logLoss = 0;
            var calibration = new List<(double, bool, double)>();

            for (var i = 0; i < truth.Length; i++)
            {
                if (truth[i] < 0 || w[i] == 0)
                    continue; // missing label, or a row that counts for nothing

                var correct = top[i] == truth[i];
                total += w[i];
                classWeight[truth[i]] += w[i];
                predictedWeight[top[i]] += w[i];
                if (correct)
                {
                    hit += w[i];
                    classHit[truth[i]] += w[i];
                    predictedHit[top[i]] += w[i];
                }
                logLoss -= w[i] * truthLog[i];
                if (!float.IsNaN(topScore[i]))
                    calibration.Add((topScore[i], correct, w[i]));
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
    }

    /// <summary>Weighted regression metrics, or <c>null</c> when the columns they need are missing.</summary>
    public static Dictionary<string, double>? Regression(IDataView predictions, string labelColumn, string weightColumn)
        => ReadRegression(predictions, labelColumn, weightColumn)?.Compute();

    /// <summary>The scored rows behind <see cref="Regression"/>, or <c>null</c> when the columns it needs are missing.</summary>
    public static IMetricRows? ReadRegression(IDataView predictions, string labelColumn, string? weightColumn)
    {
        var label = predictions.Schema.GetColumnOrNull(labelColumn);
        var score = predictions.Schema.GetColumnOrNull("Score");
        if (label is null || score is null
            || label.Value.Type != NumberDataViewType.Single || score.Value.Type != NumberDataViewType.Single)
            return null;

        var ys = new List<double>();
        var ss = new List<double>();
        // Rows without a finite label or score never count; kept as NaN so the row index still lines up with the weights.
        using (var cursor = predictions.GetRowCursor([label.Value, score.Value]))
        {
            var getLabel = cursor.GetGetter<float>(label.Value);
            var getScore = cursor.GetGetter<float>(score.Value);
            float y = default, s = default;
            while (cursor.MoveNext())
            {
                getLabel(ref y);
                getScore(ref s);
                ys.Add(y);
                ss.Add(s);
            }
        }

        var weights = weightColumn is null ? Ones(ys.Count) : RowWeights.Read(predictions, weightColumn);
        return new RegressionRows([.. ys], [.. ss], weights);
    }

    private sealed class RegressionRows(double[] y, double[] s, double[] weights) : IMetricRows
    {
        public int Count => y.Length;

        public double[] Weights => weights;

        public Dictionary<string, double>? Compute(double[] w)
        {
            double total = 0, sumY = 0;
            for (var i = 0; i < y.Length; i++)
            {
                if (!(w[i] > 0) || !double.IsFinite(y[i]) || !double.IsFinite(s[i]))
                    continue;
                total += w[i];
                sumY += w[i] * y[i];
            }
            if (total <= 0)
                return null;

            var mean = sumY / total;
            double sse = 0, sst = 0, sae = 0;
            for (var i = 0; i < y.Length; i++)
            {
                if (!(w[i] > 0) || !double.IsFinite(y[i]) || !double.IsFinite(s[i]))
                    continue;
                sse += w[i] * (y[i] - s[i]) * (y[i] - s[i]);
                sst += w[i] * (y[i] - mean) * (y[i] - mean);
                sae += w[i] * Math.Abs(y[i] - s[i]);
            }
            var mse = sse / total;
            var metrics = new Dictionary<string, double>
            {
                ["mse"] = mse,
                ["rmse"] = Math.Sqrt(mse),
                ["mae"] = sae / total,
            };
            // ML.NET reports R² 0 for a constant label; undefined here, and left out like any 0/0.
            if (sst > 0)
                metrics["r_squared"] = 1 - sse / sst;
            return metrics;
        }
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

    private static double[] Ones(int count)
    {
        var ones = new double[count];
        Array.Fill(ones, 1.0);
        return ones;
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
