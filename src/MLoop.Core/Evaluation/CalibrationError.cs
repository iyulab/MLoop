using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Evaluation;

/// <summary>
/// Expected calibration error (ECE) of a classifier's probabilities on held-out predictions: how far
/// "the model said p" is from "it was right a fraction p of the time", averaged over ten equal-width
/// confidence bins weighted by how many predictions fall in each.
/// </summary>
/// <remarks>
/// <para>
/// A consumer that skips a costlier check whenever the model's confidence clears a threshold is
/// trusting that the number is a probability. Nothing measured that: accuracy, AUC and F1 are all
/// blind to calibration (AUC is a ranking statistic and does not change under any monotone
/// rescaling of the scores). ECE is the standard measurement, and 0 means calibrated.
/// </para>
/// <para>
/// Binary: the reliability of the positive-class probability (bin by p, compare with the observed
/// positive rate). Multiclass: top-label calibration (bin by the largest class probability, compare
/// with how often that class was right). A model without probabilities has no ECE — <c>null</c>,
/// never a number made up from raw scores.
/// </para>
/// </remarks>
public static class CalibrationError
{
    /// <summary>The metric key. Lower is better; it is reported, never optimized.</summary>
    public const string MetricName = "ece";

    /// <summary>Equal-width bins over [0, 1].</summary>
    public const int Bins = 10;

    /// <summary>ECE of the positive-class probability, or <c>null</c> without a probability column.</summary>
    public static double? Binary(IDataView predictions, string labelColumn)
    {
        var probability = predictions.Schema.GetColumnOrNull("Probability");
        var label = predictions.Schema.GetColumnOrNull(labelColumn);
        if (probability is null || label is null || probability.Value.Type != NumberDataViewType.Single)
            return null;

        var labelType = label.Value.Type;
        if (labelType != BooleanDataViewType.Instance && labelType != NumberDataViewType.Single)
            return null;

        var pairs = new List<(double Confidence, bool Hit)>();
        using var cursor = predictions.GetRowCursor([probability.Value, label.Value]);
        var getProbability = cursor.GetGetter<float>(probability.Value);
        Func<bool> isPositive;
        if (labelType == BooleanDataViewType.Instance)
        {
            var getLabel = cursor.GetGetter<bool>(label.Value);
            isPositive = () => { bool v = default; getLabel(ref v); return v; };
        }
        else
        {
            var getLabel = cursor.GetGetter<float>(label.Value);
            isPositive = () => { float v = default; getLabel(ref v); return v > 0.5f; };
        }

        while (cursor.MoveNext())
        {
            float p = default;
            getProbability(ref p);
            if (float.IsNaN(p))
                continue;
            pairs.Add((p, isPositive()));
        }

        return Compute(pairs);
    }

    /// <summary>
    /// Top-label ECE from the per-class probabilities in <c>Score</c>, or <c>null</c> when the label is
    /// not a key column or the scores are not a vector of the label's width.
    /// </summary>
    public static double? Multiclass(IDataView predictions, string labelColumn)
    {
        var score = predictions.Schema.GetColumnOrNull("Score");
        var label = predictions.Schema.GetColumnOrNull(labelColumn);
        if (score is null || label is null
            || score.Value.Type is not VectorDataViewType { ItemType: NumberDataViewType itemType } vector
            || itemType != NumberDataViewType.Single
            || label.Value.Type is not KeyDataViewType key
            || vector.Size != (int)key.Count)
            return null;

        var pairs = new List<(double Confidence, bool Hit)>();
        using var cursor = predictions.GetRowCursor([score.Value, label.Value]);
        var getScore = cursor.GetGetter<VBuffer<float>>(score.Value);
        var getLabel = cursor.GetGetter<uint>(label.Value);
        VBuffer<float> scores = default;
        uint labelKey = default;

        while (cursor.MoveNext())
        {
            getScore(ref scores);
            getLabel(ref labelKey);
            if (labelKey == 0)
                continue; // missing label

            var values = scores.DenseValues().ToArray();
            var top = 0;
            for (var i = 1; i < values.Length; i++)
                if (values[i] > values[top])
                    top = i;

            if (float.IsNaN(values[top]))
                continue;

            // Score slots are in key order: slot i is the class whose key is i + 1.
            pairs.Add((values[top], top == labelKey - 1));
        }

        return Compute(pairs);
    }

    /// <summary>
    /// ECE over (confidence, hit) pairs — for the binary case the confidence is P(positive) and a hit
    /// is a positive label; for top-label calibration it is the top probability and whether that class
    /// was right. <c>null</c> for no pairs.
    /// </summary>
    public static double? Compute(IReadOnlyCollection<(double Confidence, bool Hit)> pairs)
    {
        if (pairs.Count == 0)
            return null;

        var count = new int[Bins];
        var confidence = new double[Bins];
        var hits = new double[Bins];

        foreach (var (p, hit) in pairs)
        {
            var bin = Math.Clamp((int)(Math.Clamp(p, 0, 1) * Bins), 0, Bins - 1);
            count[bin]++;
            confidence[bin] += p;
            if (hit)
                hits[bin]++;
        }

        double ece = 0;
        for (var b = 0; b < Bins; b++)
        {
            if (count[b] == 0)
                continue;
            ece += (double)count[b] / pairs.Count * Math.Abs(hits[b] / count[b] - confidence[b] / count[b]);
        }

        return ece;
    }
}
