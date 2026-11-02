namespace MLoop.Search.Ensemble;

/// <summary>
/// The members of a blend and how often each is counted: the outcome of <see cref="ForwardSelection.Select"/>.
/// </summary>
/// <param name="Counts">How many times each member was added in the best prefix — the weights, once divided by their sum.</param>
/// <param name="Score">The metric of the blend with those counts, on the answers the selection was made on.</param>
/// <param name="SingleScores">The metric of each member alone on the same answers.</param>
public sealed record BlendSelection(IReadOnlyList<int> Counts, double Score, IReadOnlyList<double> SingleScores)
{
    /// <summary>How many additions were kept — the sum of <see cref="Counts"/>.</summary>
    public int Steps => Counts.Sum();

    /// <summary>The members counted at least once.</summary>
    public int Members => Counts.Count(c => c > 0);

    /// <summary>The weight of each member: its count over <see cref="Steps"/>.</summary>
    public IReadOnlyList<double> Weights => [.. Counts.Select(c => (double)c / Steps)];

    /// <summary>Whether the selection ended on one member — a blend that is no blend, which a caller keeps as the plain model.</summary>
    public bool IsSingle => Members == 1;
}

/// <summary>
/// Forward selection with replacement (Caruana et al. 2004) over the out-of-fold answers of a few trials: from nothing, add
/// the member whose addition most improves the running average's metric, again and again, and keep the prefix that scored
/// best. The members' weights are the counts of that prefix. Pure: the answers and the metric are the caller's, so one
/// routine serves a probability for each row (binary) and a predicted value (regression), weighted or not.
/// </summary>
/// <remarks>
/// This is the arm measured against the best single trial on 16 tables (a blend was better on 8, worse on 1); it adds
/// nothing the measurement did not include. Ties go to the earlier member and a later prefix must score strictly better
/// to replace an earlier one, so the same answers always give the same counts.
/// </remarks>
public static class ForwardSelection
{
    /// <summary>The additions tried, as in the measurement.</summary>
    public const int DefaultSteps = 20;

    /// <summary>
    /// Selects the counts of <paramref name="answers"/>' members. A candidate whose metric is not finite is skipped; the
    /// selection ends early when none is finite.
    /// </summary>
    /// <param name="answers">Each member's answer for every row, in the same row order; all the same length.</param>
    /// <param name="metric">Scores a blended answer for every row (the caller holds the truth and the row weights).</param>
    /// <param name="higherIsBetter">Whether a larger metric is better.</param>
    /// <param name="steps">How many additions to try.</param>
    /// <exception cref="ArgumentException">No members, no rows, or members of unequal lengths.</exception>
    public static BlendSelection Select(
        IReadOnlyList<float[]> answers, Func<double[], double> metric, bool higherIsBetter, int steps = DefaultSteps)
    {
        ArgumentNullException.ThrowIfNull(answers);
        ArgumentNullException.ThrowIfNull(metric);
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);
        if (answers.Count == 0)
            throw new ArgumentException("A blend needs at least one member.", nameof(answers));
        var rows = answers[0].Length;
        if (rows == 0 || answers.Any(a => a.Length != rows))
            throw new ArgumentException("The members' answers must be non-empty and one for every row, in the same order.", nameof(answers));

        bool Better(double a, double b) => higherIsBetter ? a > b : a < b;
        var singles = new double[answers.Count];
        var scratch = new double[rows];
        for (var j = 0; j < answers.Count; j++)
        {
            for (var r = 0; r < rows; r++)
                scratch[r] = answers[j][r];
            singles[j] = metric(scratch);
        }

        var counts = new int[answers.Count];
        var sum = new double[rows];
        var total = 0;
        var bestScore = double.NaN;
        var bestCounts = new int[answers.Count];
        for (var step = 0; step < steps; step++)
        {
            var (pick, pickScore) = (-1, double.NaN);
            for (var j = 0; j < answers.Count; j++)
            {
                for (var r = 0; r < rows; r++)
                    scratch[r] = (sum[r] + answers[j][r]) / (total + 1);
                var score = metric(scratch);
                if (double.IsFinite(score) && (pick < 0 || Better(score, pickScore)))
                    (pick, pickScore) = (j, score);
            }
            if (pick < 0)
                break;
            for (var r = 0; r < rows; r++)
                sum[r] += answers[pick][r];
            counts[pick]++;
            total++;
            if (double.IsNaN(bestScore) || Better(pickScore, bestScore))
                (bestScore, bestCounts) = (pickScore, (int[])counts.Clone());
        }

        return bestCounts.Sum() == 0
            ? throw new InvalidOperationException("No member gave a finite metric, so there is nothing to blend.")
            : new BlendSelection(bestCounts, bestScore, singles);
    }
}
