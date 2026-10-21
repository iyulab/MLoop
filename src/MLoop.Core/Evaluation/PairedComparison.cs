using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Evaluation;

/// <summary>What a paired comparison concluded about the candidate. Serialized by name.</summary>
[System.Text.Json.Serialization.JsonConverter(typeof(System.Text.Json.Serialization.JsonStringEnumConverter<PairedVerdict>))]
public enum PairedVerdict
{
    /// <summary>The interval holds zero: on these rows the two models cannot be told apart.</summary>
    Indistinguishable = 0,

    /// <summary>The candidate's loss is lower, and the interval excludes zero.</summary>
    CandidateBetter = 1,

    /// <summary>The baseline's loss is lower, and the interval excludes zero.</summary>
    BaselineBetter = 2,
}

/// <summary>
/// Two models scored on the same rows, compared row by row: the weighted mean of each one's per-row
/// loss, their difference (candidate − baseline, so negative favors the candidate), and a 95% paired
/// bootstrap interval for it.
/// </summary>
/// <param name="Measure">What a row's loss is — <see cref="PairedComparison.ErrorRate"/> or <see cref="PairedComparison.SquaredError"/>.</param>
/// <param name="Rows">Rows both models answered with a usable label and a positive weight.</param>
/// <param name="Candidate">The candidate's weighted mean loss.</param>
/// <param name="Baseline">The baseline's weighted mean loss.</param>
/// <param name="Difference">Candidate − baseline; negative favors the candidate.</param>
/// <param name="Low">Lower end of the 95% paired bootstrap interval for <paramref name="Difference"/>.</param>
/// <param name="High">Upper end of that interval.</param>
/// <param name="Verdict">Which side the interval falls on, if either.</param>
public sealed record PairedResult(
    string Measure,
    int Rows,
    double Candidate,
    double Baseline,
    double Difference,
    double Low,
    double High,
    PairedVerdict Verdict);

/// <summary>
/// Compares a candidate model with a baseline on one labeled, optionally weighted set of rows. Each
/// model's aggregate metrics say how good it is; they do not say whether the gap between two of them
/// is more than which rows happened to be drawn. Pairing does: both models answer the same row, so the
/// row-to-row variation they share cancels, and a resample of rows moves both together.
/// </summary>
/// <remarks>
/// The bootstrap resamples rows uniformly and recomputes the weighted mean difference, so a weight
/// (a recheck sample's 1/π) changes what each row counts for but not how likely it is to be drawn —
/// the rows are the sample, the weights are how the sample stands for the population. Fixed seed:
/// the same rows give the same interval.
/// </remarks>
public static class PairedComparison
{
    /// <summary>A classifier's per-row loss: 1 for a wrong class, 0 for the right one.</summary>
    public const string ErrorRate = "error_rate";

    /// <summary>A regressor's per-row loss: the squared difference from the label.</summary>
    public const string SquaredError = "squared_error";

    /// <summary>Bootstrap resamples drawn.</summary>
    public const int Resamples = 2_000;

    /// <summary>The per-row loss <paramref name="task"/> is compared by, or <c>null</c> when it has none.</summary>
    public static string? MeasureOf(string? task)
        => Models.TaskTypes.Canonical(task) switch
        {
            "binary-classification" or "multiclass-classification" => ErrorRate,
            "regression" => SquaredError,
            _ => null,
        };

    /// <summary>
    /// The per-row loss of <paramref name="scored"/> (a model's output over the test rows), in row order;
    /// <see cref="double.NaN"/> for a row without a usable label. <c>null</c> when the task has no
    /// measure or the columns it needs are missing.
    /// </summary>
    public static double[]? RowLosses(IDataView scored, string task, string labelColumn)
    {
        ArgumentNullException.ThrowIfNull(scored);
        return MeasureOf(task) switch
        {
            ErrorRate when Models.TaskTypes.Canonical(task) == "binary-classification" => BinaryMisses(scored, labelColumn),
            ErrorRate => MulticlassMisses(scored, labelColumn),
            SquaredError => SquaredErrors(scored, labelColumn),
            _ => null,
        };
    }

    /// <summary>Compares the paired losses; rows where either is NaN, or the weight is not positive, are left out of both.</summary>
    public static PairedResult Compare(
        string measure, IReadOnlyList<double> candidate, IReadOnlyList<double> baseline, IReadOnlyList<double>? weights = null)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        ArgumentNullException.ThrowIfNull(baseline);
        if (candidate.Count != baseline.Count || (weights is not null && weights.Count != candidate.Count))
            throw new ArgumentException("Both models' losses (and the weights) must be over the same rows.");

        var rows = new List<(double C, double B, double W)>(candidate.Count);
        for (var i = 0; i < candidate.Count; i++)
        {
            var w = weights?[i] ?? 1.0;
            if (double.IsFinite(candidate[i]) && double.IsFinite(baseline[i]) && w > 0 && double.IsFinite(w))
                rows.Add((candidate[i], baseline[i], w));
        }
        if (rows.Count == 0)
            return new PairedResult(measure, 0, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, PairedVerdict.Indistinguishable);

        static double Mean(IEnumerable<(double C, double B, double W)> r, Func<(double C, double B, double W), double> f)
        {
            double num = 0, den = 0;
            foreach (var x in r) { num += x.W * f(x); den += x.W; }
            return num / den;
        }

        var c = Mean(rows, x => x.C);
        var b = Mean(rows, x => x.B);
        var differences = new double[Resamples];
        var random = new Random(20261021);
        for (var k = 0; k < Resamples; k++)
        {
            double num = 0, den = 0;
            for (var j = 0; j < rows.Count; j++)
            {
                var x = rows[random.Next(rows.Count)];
                num += x.W * (x.C - x.B);
                den += x.W;
            }
            differences[k] = num / den;
        }
        Array.Sort(differences);
        var low = differences[(int)Math.Floor(0.025 * (Resamples - 1))];
        var high = differences[(int)Math.Ceiling(0.975 * (Resamples - 1))];
        var verdict = high < 0 ? PairedVerdict.CandidateBetter
            : low > 0 ? PairedVerdict.BaselineBetter
            : PairedVerdict.Indistinguishable;
        return new PairedResult(measure, rows.Count, c, b, c - b, low, high, verdict);
    }

    private static double[]? BinaryMisses(IDataView scored, string labelColumn)
    {
        var label = scored.Schema.GetColumnOrNull(labelColumn);
        var predicted = scored.Schema.GetColumnOrNull("PredictedLabel");
        if (label is null || predicted is null || predicted.Value.Type != BooleanDataViewType.Instance)
            return null;
        var losses = new List<double>();
        using var cursor = scored.GetRowCursor([label.Value, predicted.Value]);
        var getPredicted = cursor.GetGetter<bool>(predicted.Value);
        Func<bool?> truth = label.Value.Type switch
        {
            BooleanDataViewType => Getter<bool, bool?>(cursor, label.Value, v => v),
            NumberDataViewType n when n == NumberDataViewType.Single => Getter<float, bool?>(cursor, label.Value, v => float.IsNaN(v) ? null : v > 0),
            _ => () => null,
        };
        bool p = default;
        while (cursor.MoveNext())
        {
            getPredicted(ref p);
            losses.Add(truth() is { } y ? (y == p ? 0 : 1) : double.NaN);
        }
        return [.. losses];
    }

    private static double[]? MulticlassMisses(IDataView scored, string labelColumn)
    {
        var label = scored.Schema.GetColumnOrNull(labelColumn);
        var score = scored.Schema.GetColumnOrNull("Score");
        if (label is null || score is null || label.Value.Type is not KeyDataViewType
            || score.Value.Type is not VectorDataViewType { ItemType: NumberDataViewType })
            return null;
        var losses = new List<double>();
        using var cursor = scored.GetRowCursor([label.Value, score.Value]);
        var getLabel = cursor.GetGetter<uint>(label.Value);
        var getScore = cursor.GetGetter<VBuffer<float>>(score.Value);
        uint key = default;
        VBuffer<float> scores = default;
        while (cursor.MoveNext())
        {
            getLabel(ref key);
            getScore(ref scores);
            if (key == 0)
            {
                losses.Add(double.NaN);
                continue;
            }
            var values = scores.DenseValues().ToArray();
            var top = 0;
            for (var c = 1; c < values.Length; c++)
                if (values[c] > values[top])
                    top = c;
            losses.Add(top == (int)key - 1 ? 0 : 1);
        }
        return [.. losses];
    }

    private static double[]? SquaredErrors(IDataView scored, string labelColumn)
    {
        var label = scored.Schema.GetColumnOrNull(labelColumn);
        var score = scored.Schema.GetColumnOrNull("Score");
        if (label is null || score is null || label.Value.Type != NumberDataViewType.Single || score.Value.Type != NumberDataViewType.Single)
            return null;
        var losses = new List<double>();
        using var cursor = scored.GetRowCursor([label.Value, score.Value]);
        var getLabel = cursor.GetGetter<float>(label.Value);
        var getScore = cursor.GetGetter<float>(score.Value);
        float y = default, s = default;
        while (cursor.MoveNext())
        {
            getLabel(ref y);
            getScore(ref s);
            losses.Add(float.IsFinite(y) && float.IsFinite(s) ? (double)(y - s) * (y - s) : double.NaN);
        }
        return [.. losses];
    }

    private static Func<TOut> Getter<TIn, TOut>(DataViewRowCursor cursor, DataViewSchema.Column column, Func<TIn, TOut> map)
    {
        var get = cursor.GetGetter<TIn>(column);
        TIn value = default!;
        return () => { get(ref value); return map(value); };
    }
}
