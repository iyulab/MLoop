using MLoop.Search.Validation;

namespace MLoop.Search.Ensemble;

/// <summary>A blend chosen from a search's retained trials: who is in it, how often each counts, and what it answered out of fold.</summary>
/// <param name="Members">The trials the blend counts, in the order they were retained (best first); every one counts at least once.</param>
/// <param name="Counts">How many times each member was added — the weights once divided by their sum.</param>
/// <param name="Score">The metric of the blend on the out-of-fold answers.</param>
/// <param name="BestSingleScore">The metric of the best retained trial alone on the same answers — what the blend has to beat.</param>
/// <param name="OutOfFold">
/// The blend's own answer for every row (a probability for a binary classifier, a value for a regression) — the weighted mean
/// of its members' out-of-fold answers; what a calibration of the blend is fitted on.
/// </param>
/// <param name="Reference">One member's rows: the labels or targets and the weights every member's answers are lined up with.</param>
public sealed record EnsemblePlan(
    IReadOnlyList<RetainedTrial> Members,
    IReadOnlyList<int> Counts,
    double Score,
    double BestSingleScore,
    float[] OutOfFold,
    OutOfFoldScores Reference)
{
    /// <summary>How many additions were kept — the sum of <see cref="Counts"/>.</summary>
    public int Steps => Counts.Sum();
}

/// <summary>
/// Chooses a blend from the trials a search retained (<see cref="SearchRequest.KeepTopOutOfFold"/>): forward selection
/// (<see cref="ForwardSelection"/>) over their out-of-fold answers, then at most <see cref="DefaultMaxMembers"/> distinct
/// members — the number the blend was measured with. Pure: the metric is the caller's, so the host's own metric authority
/// (weights included) scores the blend as it scores a trial.
/// </summary>
public static class EnsembleSelection
{
    /// <summary>The most distinct members a blend holds.</summary>
    public const int DefaultMaxMembers = 5;

    /// <summary>
    /// Chooses the blend, or returns <see langword="null"/> when there is none to make: fewer than two trials whose answers
    /// can be blended, or a selection that ends on one member (the plain model is the answer then).
    /// </summary>
    /// <param name="retained">The retained trials, best first.</param>
    /// <param name="metric">Scores a blended answer for every row, given the rows' labels or targets and weights (the first argument).</param>
    /// <param name="higherIsBetter">Whether a larger metric is better.</param>
    /// <param name="maxMembers">The most distinct members.</param>
    /// <param name="steps">How many additions forward selection tries.</param>
    public static EnsemblePlan? Choose(
        IReadOnlyList<RetainedTrial> retained,
        Func<OutOfFoldScores, double[], double> metric,
        bool higherIsBetter,
        int maxMembers = DefaultMaxMembers,
        int steps = ForwardSelection.DefaultSteps)
    {
        ArgumentNullException.ThrowIfNull(retained);
        ArgumentNullException.ThrowIfNull(metric);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxMembers, 2);

        var eligible = Blendable(retained);
        if (eligible.Count < 2)
            return null;
        var reference = eligible[0].OutOfFold;
        var answers = eligible.Select(t => AnswerOf(t.OutOfFold)).ToArray();
        double Metric(double[] blended) => metric(reference, blended);

        var selection = ForwardSelection.Select(answers, Metric, higherIsBetter, steps);
        if (selection.Members > maxMembers)
        {
            // More members counted than a blend holds: keep the ones counted most (ties to the better trial) and select again among them.
            var keep = Enumerable.Range(0, eligible.Count)
                .Where(i => selection.Counts[i] > 0)
                .OrderByDescending(i => selection.Counts[i]).ThenBy(i => i)
                .Take(maxMembers).Order().ToArray();
            eligible = [.. keep.Select(i => eligible[i])];
            answers = [.. keep.Select(i => answers[i])];
            selection = ForwardSelection.Select(answers, Metric, higherIsBetter, steps);
        }
        if (selection.IsSingle)
            return null;

        var counted = Enumerable.Range(0, eligible.Count).Where(i => selection.Counts[i] > 0).ToArray();
        var blend = new float[reference.Count];
        foreach (var i in counted)
            for (var r = 0; r < blend.Length; r++)
                blend[r] += selection.Counts[i] * answers[i][r];
        for (var r = 0; r < blend.Length; r++)
            blend[r] /= selection.Steps;

        var bestSingle = higherIsBetter ? selection.SingleScores.Where(double.IsFinite).DefaultIfEmpty(double.NaN).Max()
            : selection.SingleScores.Where(double.IsFinite).DefaultIfEmpty(double.NaN).Min();
        return new EnsemblePlan(
            [.. counted.Select(i => eligible[i])], [.. counted.Select(i => selection.Counts[i])], selection.Score, bestSingle, blend, reference);
    }

    // The trials whose answers line up with the best one's: the same kind, the same rows in the same order, and an answer a
    // blend can average — a binary classifier's probability (a trainer with none is left out), a regression's value.
    private static List<RetainedTrial> Blendable(IReadOnlyList<RetainedTrial> retained)
    {
        var kept = new List<RetainedTrial>();
        foreach (var trial in retained)
        {
            var rows = trial.OutOfFold;
            if (rows.IsMulticlass || rows.Count == 0 || (!rows.IsRegression && rows.Probability is null))
                continue;
            if (kept.Count > 0 && !SameRows(kept[0].OutOfFold, rows))
                continue;
            kept.Add(trial);
        }
        return kept;
    }

    private static bool SameRows(OutOfFoldScores a, OutOfFoldScores b)
        => a.IsRegression == b.IsRegression && a.Count == b.Count
           && (a.IsRegression ? a.Target.SequenceEqual(b.Target) : a.Label.SequenceEqual(b.Label))
           && a.Weight.SequenceEqual(b.Weight);

    private static float[] AnswerOf(OutOfFoldScores rows) => [.. rows.IsRegression ? rows.Score : rows.Probability!];
}
