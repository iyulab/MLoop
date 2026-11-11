namespace MLoop.Core.Prediction;

/// <summary>
/// A time-series anomaly detector judges a point against a window of the points before it, so the first rows of what it is
/// given carry no verdict: the detector answers an alert of 0 and a score of exactly 0 for them. Read as an answer, that is
/// "normal" — a spike in the first window of a file (or the only row of a served request) was reported as nothing to see.
/// Those rows say they were not judged instead: no verdict, no score, and one warning naming how many.
/// </summary>
public static class TimeSeriesAnomalyWarmup
{
    /// <summary>
    /// The rows with the leading unjudged ones (no alert and a raw score of exactly 0, from the first row on) stripped of their
    /// verdict, and the warning that says so — <c>null</c> when every row was judged. A judged point has a score that is
    /// never exactly 0, so only the run that opens the series is read this way.
    /// </summary>
    public static (List<PredictionRow> Rows, string? Warning) Mark(List<PredictionRow> rows)
    {
        ArgumentNullException.ThrowIfNull(rows);
        var unjudged = 0;
        while (unjudged < rows.Count && rows[unjudged].IsAnomaly != true && rows[unjudged].AnomalyScore is 0.0)
            unjudged++;
        if (unjudged == 0)
            return (rows, null);

        var marked = new List<PredictionRow>(rows.Count);
        for (var i = 0; i < rows.Count; i++)
            marked.Add(i < unjudged ? rows[i] with { IsAnomaly = null, AnomalyScore = null } : rows[i]);

        var warning = unjudged == rows.Count
            ? $"None of the {rows.Count} row(s) was judged: the detector needs a window of earlier points before it can say whether a point belongs to the series, so these rows carry no verdict (isAnomaly and anomalyScore are absent). Send the series with the earlier rows in front of the ones you ask about."
            : $"The first {unjudged} of {rows.Count} rows were not judged: the detector needs a window of earlier points before it can say whether a point belongs to the series, so these rows carry no verdict (isAnomaly and anomalyScore are absent). Send the series with at least {unjudged} earlier rows in front of the ones you ask about.";
        return (marked, warning);
    }
}
