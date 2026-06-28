using MLoop.Core.Evaluation;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Ranks experiment summaries by metric quality, honoring each metric's optimization direction
/// (F-28). Status, Compare, and the REST API all previously sorted <see cref="ExperimentSummary.BestMetric"/>
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

    /// <summary>Experiments ordered best-first, honoring metric direction; metric-less ones last.</summary>
    public static IEnumerable<ExperimentSummary> OrderByQuality(IEnumerable<ExperimentSummary> experiments)
        => experiments.OrderBy(QualityKey);

    /// <summary>The single best experiment by metric quality, or null when none has a metric.</summary>
    public static ExperimentSummary? SelectBest(IEnumerable<ExperimentSummary> experiments)
        => experiments.Where(e => e.BestMetric.HasValue).OrderBy(QualityKey).FirstOrDefault();
}
