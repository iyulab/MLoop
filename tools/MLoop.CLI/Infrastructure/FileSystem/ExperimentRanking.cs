using MLoop.Core.Evaluation;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Ranks experiment summaries by metric quality, honoring each metric's optimization direction
/// . Status, Compare, and the REST API all previously sorted <see cref="ExperimentSummary.BestMetric"/>
/// descending unconditionally — so for lower-is-better metrics (clustering's <c>average_distance</c>,
/// forecasting's <c>mae</c>, recommendation's <c>rmse</c>) they ranked the <i>worst</i> experiment as
/// "best", contradicting the promotion gate. Uses the experiment's own <see cref="ExperimentSummary.MetricName"/>
/// plus the shared <see cref="MetricDirection"/> so ranking matches what would actually be promoted.
/// </summary>
public static class ExperimentRanking
{
    // Sort key where smaller = better. Lower-is-better metrics sort by their value directly;
    // higher-is-better by the negated value; experiments with no metric sort last.
    private static double QualityKey(ExperimentSummary e)
        => !e.BestMetric.HasValue
            ? double.MaxValue
            : MetricDirection.IsLowerBetter(e.MetricName ?? string.Empty)
                ? e.BestMetric.Value
                : -e.BestMetric.Value;

    /// <summary>
    /// What a recorded metric value is a measurement of: the metric and the row weighting. Two values
    /// compare only when both agree — a metric weighted toward rare, re-checked cases and an
    /// unweighted one of the same name describe different distributions.
    /// </summary>
    public readonly record struct Measurement(string Metric, string? WeightColumn)
    {
        public static Measurement Of(ExperimentSummary e) => new(
            (e.MetricName ?? string.Empty).ToLowerInvariant(),
            string.IsNullOrEmpty(e.WeightColumn) ? null : e.WeightColumn.ToLowerInvariant());

        /// <summary>"auc" or "auc, weighted by 'w'" — names the scale a value is on.</summary>
        public override string ToString()
            => WeightColumn is null ? Metric : $"{Metric}, weighted by '{WeightColumn}'";
    }

    /// <summary>
    /// The best experiment and what it was compared against: <see cref="NotCompared"/> experiments
    /// carry a metric measured differently, so they had no place in the ranking.
    /// </summary>
    public sealed record BestPick(ExperimentSummary Best, Measurement Measurement, int NotCompared);

    /// <summary>
    /// Experiments ordered best-first, honoring metric direction. Those measured like the newest
    /// experiment with a metric come first, then the ones measured differently (each still best-first),
    /// then metric-less ones — so a top-N taken from the front stays within one scale whenever it can.
    /// </summary>
    public static IEnumerable<ExperimentSummary> OrderByQuality(IEnumerable<ExperimentSummary> experiments)
    {
        var list = experiments.ToList();
        var newest = list.Where(e => e.BestMetric.HasValue).OrderByDescending(e => e.Timestamp).FirstOrDefault();
        if (newest is null)
            return list;
        var reference = Measurement.Of(newest);
        return list
            .OrderBy(e => !e.BestMetric.HasValue ? 2 : Measurement.Of(e) == reference ? 0 : 1)
            .ThenBy(e => Measurement.Of(e).ToString(), StringComparer.Ordinal)
            .ThenBy(QualityKey);
    }

    /// <summary>
    /// The best experiment among those measured like the newest one with a metric — the model's
    /// current definition — or null when none has a metric. Experiments measured differently (another
    /// metric, another weighting) are counted in <see cref="BestPick.NotCompared"/>, never ranked
    /// against it.
    /// </summary>
    public static BestPick? SelectBest(IEnumerable<ExperimentSummary> experiments)
    {
        var scored = experiments.Where(e => e.BestMetric.HasValue).ToList();
        if (scored.Count == 0)
            return null;

        var reference = Measurement.Of(scored.OrderByDescending(e => e.Timestamp).First());
        var comparable = scored.Where(e => Measurement.Of(e) == reference).ToList();
        var best = comparable.OrderBy(QualityKey).ThenByDescending(e => e.Timestamp).First();
        return new BestPick(best, reference, scored.Count - comparable.Count);
    }
}
