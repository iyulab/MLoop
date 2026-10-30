namespace MLoop.Core.Evaluation;

/// <summary>One input column's permutation importance: how much the model's score falls when that column's values are shuffled among the rows.</summary>
/// <param name="Column">The input column.</param>
/// <param name="Importance">The mean fall of the score over the repeats, in the metric's own units — positive when the model needs the column, near zero when it does not, negative when shuffling happened to help.</param>
/// <param name="StandardDeviation">How much the fall varied between repeats (0 with one).</param>
/// <param name="Falls">The fall of each repeat.</param>
public sealed record FeatureImportance(string Column, double Importance, double StandardDeviation, IReadOnlyList<double> Falls);

/// <summary>
/// Permutation importance of a trained model's input columns: score the model on a labeled file, shuffle one column's values among the
/// rows (breaking its link to the label and to the other columns, keeping its distribution), score again, and read the fall of the
/// score as how much the model relied on the column. Model-agnostic, measured on rows the caller chooses, and in the units of the
/// metric the caller scores by.
/// </summary>
/// <remarks>
/// This is the model's reliance, not the data's: a column the model never reads falls by nothing however strongly it relates to the
/// label, and two correlated columns share what either would show alone. It is the question to ask of a held-out file when a score
/// looks too good (which column carries it?) or a column is expensive to collect (does the model use it?).
/// </remarks>
public static class PermutationImportance
{
    /// <summary>A permutation of <c>0 … count − 1</c>, fixed by <paramref name="seed"/> (Fisher–Yates).</summary>
    public static int[] Order(int count, int seed)
    {
        if (count < 0)
            throw new ArgumentOutOfRangeException(nameof(count), count, "A count is not negative.");
        var order = new int[count];
        for (var i = 0; i < count; i++)
            order[i] = i;
        var rng = new Random(seed);
        for (var i = count - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
        return order;
    }

    /// <summary>The seed of one column's one repeat — the same run twice shuffles the same way, and no two (column, repeat) shuffle alike.</summary>
    public static int SeedOf(int seed, int column, int repeat)
        => unchecked(seed * 1_000_003 + column * 7_919 + repeat * 104_729 + 17);

    /// <summary>
    /// The importances, most important first (ties by column name): for each column the fall of <paramref name="baseline"/> to each of its
    /// <paramref name="scores"/>, read in the direction <paramref name="higherIsBetter"/> says is good.
    /// </summary>
    public static IReadOnlyList<FeatureImportance> Summarize(
        double baseline, bool higherIsBetter, IReadOnlyList<(string Column, IReadOnlyList<double> Scores)> scores)
    {
        ArgumentNullException.ThrowIfNull(scores);
        var result = new List<FeatureImportance>(scores.Count);
        foreach (var (column, shuffled) in scores)
        {
            if (shuffled.Count == 0)
                continue;
            var falls = shuffled.Select(score => higherIsBetter ? baseline - score : score - baseline).ToList();
            var mean = falls.Average();
            var deviation = falls.Count > 1 ? Math.Sqrt(falls.Sum(f => (f - mean) * (f - mean)) / (falls.Count - 1)) : 0;
            result.Add(new FeatureImportance(column, mean, deviation, falls));
        }
        return [.. result.OrderByDescending(f => f.Importance).ThenBy(f => f.Column, StringComparer.Ordinal)];
    }
}
