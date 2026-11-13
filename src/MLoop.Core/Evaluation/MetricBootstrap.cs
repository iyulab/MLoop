namespace MLoop.Core.Evaluation;

/// <summary>
/// A 95% bootstrap interval for one metric, and how it was made.
/// </summary>
/// <param name="Low">Lower end (2.5th percentile of the resampled metric).</param>
/// <param name="High">Upper end (97.5th percentile).</param>
/// <param name="Resamples">Resamples in which the metric was defined — a resample whose rows hold one class has no AUC.</param>
public sealed record MetricInterval(double Low, double High, int Resamples);

/// <summary>What <see cref="MetricBootstrap.Measure"/> found out about a set of scored rows.</summary>
/// <param name="Rows">Rows that count (positive weight): the sample the interval speaks for.</param>
/// <param name="Level">The coverage aimed at.</param>
/// <param name="Intervals">An interval per metric the rows' own metrics report and the resamples defined often enough; empty when none.</param>
/// <param name="Note">Why there are no (or fewer) intervals, when there are not; otherwise <c>null</c>.</param>
public sealed record MetricBootstrapResult(
    int Rows, double Level, IReadOnlyDictionary<string, MetricInterval> Intervals, string? Note);

/// <summary>
/// How far a test file's metrics move with the rows that happened to be in it: the rows are resampled with
/// replacement and the metrics recomputed, by the same arithmetic that produced the reported value
/// (<see cref="IMetricRows"/>). A draw is a multinomial count per row multiplied into its weight, so a user's
/// weights keep their meaning — the rows are the sample, the weights say how the sample stands for the population,
/// the stance <see cref="PairedComparison"/> takes too. Seeded, so the same rows give the same interval.
/// </summary>
/// <remarks>
/// This is the sampling noise of one fixed model on one file. It is not the spread of a model's score across
/// training folds (<c>ValidatedScore</c>) — a different quantity that answers a different question, and the two
/// are never merged. A percentile bootstrap is known to cover somewhat less than its level on small samples; the
/// measured coverage is recorded beside the tests rather than hidden.
/// </remarks>
public static class MetricBootstrap
{
    /// <summary>The coverage aimed at.</summary>
    public const double Level = 0.95;

    /// <summary>Resamples drawn when the work cap allows.</summary>
    public const int Resamples = 1_000;

    /// <summary>Fewest resamples worth reporting an interval from.</summary>
    public const int MinimumResamples = 200;

    /// <summary>
    /// Upper bound on rows × resamples. It bounds the cost of a very large file — whose interval is narrow anyway —
    /// and says nothing about what a «small» file is.
    /// </summary>
    public const long WorkCap = 200_000_000;

    private const int Seed = 20261113;

    /// <summary>Resamples <paramref name="rows"/> and reports an interval for each of their metrics.</summary>
    public static MetricBootstrapResult Measure(IMetricRows rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        var baseWeights = rows.Weights;
        var counting = new List<int>(rows.Count);
        for (var i = 0; i < rows.Count; i++)
            if (baseWeights[i] > 0 && double.IsFinite(baseWeights[i]))
                counting.Add(i);

        var point = rows.Compute(baseWeights);
        if (point is null || counting.Count < 2)
            return new MetricBootstrapResult(counting.Count, Level, Empty, "too few rows count to resample");

        var resamples = (int)Math.Min(Resamples, WorkCap / counting.Count);
        if (resamples < MinimumResamples)
            return new MetricBootstrapResult(counting.Count, Level, Empty,
                $"{counting.Count:N0} rows: the sample is large enough that its interval is narrow, and resampling it would be slow — not computed");

        var values = point.Keys.ToDictionary(k => k, _ => new List<double>(resamples));
        var random = new Random(Seed);
        var weights = new double[rows.Count];
        for (var b = 0; b < resamples; b++)
        {
            Array.Clear(weights);
            for (var j = 0; j < counting.Count; j++)
                weights[counting[random.Next(counting.Count)]] += 1;
            foreach (var i in counting)
                weights[i] *= baseWeights[i];

            if (rows.Compute(weights) is not { } drawn)
                continue;
            foreach (var (key, list) in values)
                if (drawn.TryGetValue(key, out var v) && double.IsFinite(v))
                    list.Add(v);
        }

        var intervals = new Dictionary<string, MetricInterval>();
        foreach (var (key, list) in values)
        {
            if (list.Count < MinimumResamples || list.Count < resamples / 2)
                continue;
            list.Sort();
            var low = list[(int)Math.Floor((1 - Level) / 2 * (list.Count - 1))];
            var high = list[(int)Math.Ceiling((1 + Level) / 2 * (list.Count - 1))];
            intervals[key] = new MetricInterval(low, high, list.Count);
        }
        return new MetricBootstrapResult(counting.Count, Level, intervals,
            intervals.Count == 0 ? "no metric was defined in enough resamples" : null);
    }

    private static readonly IReadOnlyDictionary<string, MetricInterval> Empty = new Dictionary<string, MetricInterval>();
}
