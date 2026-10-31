using Microsoft.ML;

namespace MLoop.Core.Evaluation;

/// <summary>
/// What a regression holdout can honestly be scored on. ML.NET's regression evaluator answers every holdout,
/// including ones that carry no information: with no row it reports r² = 0 and rmse = 0 together (a perfect error
/// beside no explained variance), and with one row, or rows whose label never varies, it divides by a variance of 0
/// and reports an r² of 1.8e7 — numbers a quality gate then reads as a model.
/// </summary>
public static class RegressionHoldout
{
    /// <summary>The rows a holdout can be scored on and how much its label varies.</summary>
    /// <param name="Rows">Rows whose label and score are both finite.</param>
    /// <param name="LabelVariance">Population variance of those rows' labels (0 for fewer than two).</param>
    public readonly record struct Census(int Rows, double LabelVariance);

    /// <summary>Counts the scoreable rows of <paramref name="scored"/> and the variance of their labels.</summary>
    public static Census Count(IDataView scored, string labelColumn, string scoreColumn = "Score")
    {
        ArgumentNullException.ThrowIfNull(scored);
        var label = scored.Schema.GetColumnOrNull(labelColumn);
        var score = scored.Schema.GetColumnOrNull(scoreColumn);
        if (label is null || score is null
            || label.Value.Type != Microsoft.ML.Data.NumberDataViewType.Single
            || score.Value.Type != Microsoft.ML.Data.NumberDataViewType.Single)
            return new Census(-1, double.NaN);   // not a scalar-Single pair: nothing to say

        // Welford, so a label with a large mean and a small spread does not lose its variance to cancellation.
        long n = 0;
        double mean = 0, m2 = 0;
        using var cursor = scored.GetRowCursor([label.Value, score.Value]);
        var getLabel = cursor.GetGetter<float>(label.Value);
        var getScore = cursor.GetGetter<float>(score.Value);
        float y = 0, s = 0;
        while (cursor.MoveNext())
        {
            getLabel(ref y);
            getScore(ref s);
            if (!float.IsFinite(y) || !float.IsFinite(s))
                continue;
            n++;
            var delta = y - mean;
            mean += delta / n;
            m2 += delta * (y - mean);
        }
        return new Census((int)n, n > 1 ? m2 / n : 0);
    }

    /// <summary>
    /// Marks, in <paramref name="metrics"/>, the metrics the holdout leaves undefined as NaN — every one for no scoreable
    /// row, <c>r_squared</c> for a single row or a label that never varies — which the metric sanitizer turns into the
    /// worst value of the metric's direction, so the quality gate refuses the model. Returns the reason, or
    /// <c>null</c> when the holdout can be scored (or could not be counted).
    /// </summary>
    public static string? MarkUndefined(Dictionary<string, double> metrics, Census census, string subject = "test partition")
    {
        ArgumentNullException.ThrowIfNull(metrics);
        if (census.Rows < 0)
            return null;
        if (census.Rows == 0)
        {
            foreach (var key in metrics.Keys.ToArray())
                metrics[key] = double.NaN;
            return $"The {subject} holds no row the model could be scored on, so no metric can be measured: it is recorded as the worst value (a quality gate refuses such a value). Provide more rows.";
        }
        if (census.Rows < 2 || !(census.LabelVariance > 0))
        {
            if (metrics.ContainsKey("r_squared"))
                metrics["r_squared"] = double.NaN;
            return census.Rows < 2
                ? $"The {subject} holds a single row, so r_squared is undefined (it compares the error to the label's spread); it is recorded as the worst value (a quality gate refuses such a value). Provide more rows."
                : $"The {subject}'s label never varies, so r_squared is undefined (it compares the error to the label's spread); it is recorded as the worst value (a quality gate refuses such a value). Provide more rows.";
        }
        return null;
    }
}
