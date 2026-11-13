namespace MLoop.Core.Models;

/// <summary>
/// What the search validated the chosen model's score at, on the training rows: the mean of the metric it ranked trials by
/// and its standard error — how far that mean could move on other rows. Next to the score on the held-out partition it
/// says how much of that single number is the split: a model scored on a hundred held-out rows is a point estimate whose
/// neighbours, on another split of the same table, can differ by a great deal.
/// </summary>
/// <param name="Metric">The canonical metric the search ranked trials by.</param>
/// <param name="Value">The metric's mean over the validation folds.</param>
/// <param name="StandardError">The folds' spread over √folds — for a holdout (one fold) the spread of its slices.</param>
/// <param name="Folds">How many folds the mean is over.</param>
public sealed record ValidatedScore(string Metric, double Value, double StandardError, int Folds)
{
    /// <summary>
    /// The validated score of the trial the experiment's model was refit from — the trial marked <see cref="TrialRecord.Selected"/>,
    /// or when none is, the best one on <paramref name="metric"/> — or <c>null</c> when that trial carries no spread (a path that scores
    /// one split, a trial with no standard error, a metric nothing was ranked by).
    /// </summary>
    public static ValidatedScore? OfWinner(IReadOnlyList<TrialRecord>? trials, string? metric)
    {
        if (trials is null || trials.Count == 0 || string.IsNullOrWhiteSpace(metric))
            return null;
        var scored = trials.Where(t => t.Error is null && t.Metrics.TryGetValue(metric, out var v) && double.IsFinite(v)).ToList();
        if (scored.Count == 0)
            return null;
        var lowerIsBetter = Evaluation.MetricDirection.IsLowerBetter(metric);
        var winner = scored.FirstOrDefault(t => t.Selected)
            ?? (lowerIsBetter ? scored.MinBy(t => t.Metrics[metric]) : scored.MaxBy(t => t.Metrics[metric]))!;
        return winner.Validation is { StandardError: { } errors, Folds: var folds }
            && errors.TryGetValue(metric, out var standardError) && double.IsFinite(standardError)
            ? new ValidatedScore(metric, winner.Metrics[metric], standardError, folds)
            : null;
    }
}
