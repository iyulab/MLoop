namespace MLoop.Core.Evaluation;

/// <summary>
/// The order a table shows a run's metrics in — what a reader meets first is the metric the run was judged by, then the
/// scores, then the errors, and last the numbers that only describe how wide a prediction interval is. Sorting by the value
/// itself (what the tables did) puts a metric in front of another because its number is larger: on a regression run the interval
/// widths, in the label's units, stood above the R² the search had optimized, which sat at the bottom.
/// </summary>
public static class MetricDisplayOrder
{
    /// <summary>
    /// <paramref name="metrics"/> in display order: <paramref name="primary"/> (the metric the run ranked its trials by) first; then the
    /// other scores, highest first; then the errors and losses, lowest first; then the prediction-interval numbers by name.
    /// Equal values are ordered by name, so one run always prints one way.
    /// </summary>
    public static IReadOnlyList<KeyValuePair<string, double>> Of(IEnumerable<KeyValuePair<string, double>> metrics, string? primary)
    {
        var rows = metrics.ToList();
        return
        [
            .. rows.Where(m => Group(m.Key, primary) == 0).OrderBy(m => m.Key, StringComparer.Ordinal),
            .. rows.Where(m => Group(m.Key, primary) == 1).OrderByDescending(m => m.Value).ThenBy(m => m.Key, StringComparer.Ordinal),
            .. rows.Where(m => Group(m.Key, primary) == 2).OrderBy(m => m.Value).ThenBy(m => m.Key, StringComparer.Ordinal),
            .. rows.Where(m => Group(m.Key, primary) == 3).OrderBy(m => m.Key, StringComparer.Ordinal),
        ];
    }

    private static int Group(string name, string? primary)
    {
        if (primary is not null && string.Equals(name, primary, StringComparison.OrdinalIgnoreCase))
            return 0;
        if (name.StartsWith("interval", StringComparison.OrdinalIgnoreCase) || name.StartsWith("norm_interval", StringComparison.OrdinalIgnoreCase))
            return 3;
        return MetricDirection.IsLowerBetter(name) || string.Equals(name, "residual_std", StringComparison.OrdinalIgnoreCase) ? 2 : 1;
    }
}
