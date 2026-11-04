using Microsoft.ML;
using MLoop.Search.Validation;

namespace MLoop.Search;

/// <summary>
/// Searching a large table on a sample of its rows, and refitting the trial it picks on all of them.
/// </summary>
/// <remarks>
/// A search always finishes its first trial, and a trainer's cost grows with the rows — and, for a one-versus-all trainer, with the
/// classes: measured, an 80,000-row partition × 100 classes cost 23 s for the cheapest first trial of a 60 s budget (one trial, a near-random
/// model), and 10⁶ rows × 100 classes cost 400 s. A search that cannot try more than one setting does not search. So a table from
/// <see cref="ProbeRows"/> rows is probed first — the search's own first trial, on <see cref="ProbeSample"/> rows — and when the
/// first trial on all the rows would take more than <see cref="FirstTrialShare"/> of the budget, the search runs on the largest
/// sample whose first trial would not. A refit that would take more than <see cref="RefitShare"/> of the budget is not made on all the
/// rows: the model is fitted on the sample, which is what the time bought. The functions here are the arithmetic and the sampling; the
/// probing is <see cref="TrialRunner"/>'s.
/// </remarks>
public static class SearchSample
{
    /// <summary>Rows (of the training partition) from which a search is probed; below it the search is the same as without this.</summary>
    public const int ProbeRows = 50_000;

    /// <summary>Rows of the probe, and the least a sampled search runs on.</summary>
    public const int ProbeSample = 10_000;

    /// <summary>The share of the time budget the first trial may take; a larger first trial is run on a sample.</summary>
    public const double FirstTrialShare = 0.1;

    /// <summary>The share of the time budget a refit on all the rows may take; a larger refit is made on the sample instead.</summary>
    public const double RefitShare = 0.5;

    /// <summary>
    /// A sample of at most one in this many of the rows, taken from a table already in memory, is copied there: every pass of a trainer
    /// over a view that skips the rest reads all of the rows (a model of hundreds of iterations makes hundreds of passes), and a copy this small
    /// costs a quarter of the table's memory at most. A larger sample is read through the table in memory instead of being copied.
    /// </summary>
    internal const int CopyBelow = 4;

    /// <summary>
    /// How many seconds a trial takes to fit one row, from a probe that took <paramref name="probeSeconds"/> on a sample of
    /// <paramref name="probeRows"/> rows: three folds of two thirds, so twice the rows are fitted.
    /// </summary>
    internal static double SecondsPerFittedRow(double probeSeconds, int probeRows) => probeSeconds / (2.0 * probeRows);

    /// <summary>Rows a trial on <paramref name="rows"/> rows fits: a fold's training rows, or nine tenths of a holdout.</summary>
    internal static double FittedRows(long rows)
        => rows < FoldPlan.HoldoutThreshold ? 2.0 * rows : (1.0 - 1.0 / FoldPlan.HoldoutParts) * rows;

    /// <summary>
    /// The rows to search: all <paramref name="rows"/> when the first trial on them is predicted to take no more than
    /// <see cref="FirstTrialShare"/> of <paramref name="budgetSeconds"/> (or the table is below <see cref="ProbeRows"/>), else the largest
    /// sample of <see cref="ProbeSample"/> × 2^k rows whose first trial is (at least <see cref="ProbeSample"/>).
    /// </summary>
    /// <param name="rows">Rows of the training partition.</param>
    /// <param name="budgetSeconds">The search's time budget.</param>
    /// <param name="probeSeconds">How long the probe took.</param>
    /// <param name="probeRows">Rows the probe ran on.</param>
    public static int RowsToSearch(long rows, double budgetSeconds, double probeSeconds, int probeRows)
    {
        if (rows < ProbeRows)
            return checked((int)rows);
        var allowed = FirstTrialShare * budgetSeconds;
        var rate = SecondsPerFittedRow(probeSeconds, probeRows);
        if (!(rate > 0) || rate * FittedRows(rows) <= allowed)
            return checked((int)rows);
        // cost(s) = rate × FittedRows(s), increasing in s: the largest s within the allowance, by the two shapes of FittedRows.
        var holdoutRows = (long)(allowed / (rate * (1.0 - 1.0 / FoldPlan.HoldoutParts)));
        var sample = holdoutRows >= FoldPlan.HoldoutThreshold ? holdoutRows : (long)(allowed / (2.0 * rate));
        // A step of ProbeSample × 2^k, rounded down: a time measured on a busy machine would otherwise give every search its own sample size.
        var step = (long)ProbeSample;
        while (step * 2 <= sample)
            step *= 2;
        return step >= rows ? checked((int)rows) : (int)step;
    }

    /// <summary>
    /// Whether a refit on all <paramref name="rows"/> rows is made, given that a refit on the <paramref name="sampleRows"/>-row sample took
    /// <paramref name="sampleRefitSeconds"/>: its linear prediction is no more than <see cref="RefitShare"/> of the budget.
    /// </summary>
    public static bool RefitsOnAllRows(long rows, int sampleRows, double sampleRefitSeconds, double budgetSeconds)
        => sampleRows >= rows || sampleRefitSeconds * rows / sampleRows <= RefitShare * budgetSeconds;

    /// <summary>
    /// A sample of <paramref name="size"/> rows: with <paramref name="stratify"/>, <paramref name="keys"/> are the rows' label keys and each key keeps its share, rows drawn by <paramref name="seed"/>. The keep flags are by row position; the same
    /// keys, size and seed give the same sample.
    /// </summary>
    internal static bool[] Stratified(IReadOnlyList<string> keys, bool stratify, int size, int seed)
    {
        var keep = new bool[keys.Count];
        if (size >= keys.Count)
        {
            Array.Fill(keep, true);
            return keep;
        }
        var groups = (stratify
            ? keys.Select((k, i) => (k, i)).GroupBy(x => x.k, StringComparer.Ordinal).OrderBy(g => g.Key, StringComparer.Ordinal).Select(g => g.Select(x => x.i).ToArray())
            : [Enumerable.Range(0, keys.Count).ToArray()]).ToList();
        // The shares by largest remainder: every class keeps its proportion, and the sizes sum to size.
        var exact = groups.Select(g => (double)size * g.Length / keys.Count).ToArray();
        var take = exact.Select(e => (int)Math.Floor(e)).ToArray();
        var left = size - take.Sum();
        foreach (var i in Enumerable.Range(0, groups.Count).OrderByDescending(i => exact[i] - take[i]).ThenBy(i => i).Take(left))
            take[i]++;
        var random = new Random(unchecked(seed + 7919));
        for (var g = 0; g < groups.Count; g++)
        {
            var rows = groups[g];
            for (var i = rows.Length - 1; i > 0; i--)
            {
                var j = random.Next(i + 1);
                (rows[i], rows[j]) = (rows[j], rows[i]);
            }
            foreach (var row in rows.Take(take[g]))
                keep[row] = true;
        }
        return keep;
    }

    /// <summary>
    /// The rows of <paramref name="data"/> whose <paramref name="keep"/> flag is set, in their order. Held in memory when
    /// <paramref name="cache"/> — not when <paramref name="data"/> already is: a second copy of most of it is memory for nothing.
    /// </summary>
    internal static IDataView Rows(MLContext mlContext, IDataView data, bool[] keep, bool cache = true)
    {
        var flagged = new HoldoutView(data, [.. keep.Select(k => k ? 1 : 0)], 1);
        var kept = mlContext.Data.FilterRowsByColumn(flagged, HoldoutView.ColumnName, 0.5, 1.5);
        var plain = mlContext.Transforms.DropColumns(HoldoutView.ColumnName).Fit(kept).Transform(kept);
        return cache ? RowCache.Of(mlContext, plain) : plain;
    }
}
