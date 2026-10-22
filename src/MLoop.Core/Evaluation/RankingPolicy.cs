using MLoop.Search;

namespace MLoop.Core.Evaluation;

/// <summary>
/// MLoop's metric vocabulary as the policy a search ranks trials by: the name is the canonical
/// metric key (<see cref="MetricNames"/>) and the direction comes from <see cref="MetricDirection"/>,
/// so the search ranks the way every other MLoop ranking does.
/// </summary>
public sealed class RankingPolicy : IMetricPolicy
{
    private RankingPolicy(string name, bool higherIsBetter)
    {
        Name = name;
        HigherIsBetter = higherIsBetter;
    }

    public string Name { get; }

    public bool HigherIsBetter { get; }

    /// <summary>
    /// The policy for a canonical metric key. A metric <see cref="MetricDirection"/> does not know
    /// is ranked higher-is-better, as MLoop's leaderboard ranks it.
    /// </summary>
    public static RankingPolicy For(string canonicalMetric)
    {
        if (string.IsNullOrWhiteSpace(canonicalMetric))
            throw new ArgumentException("A metric policy needs a metric name.", nameof(canonicalMetric));
        return new RankingPolicy(canonicalMetric, !MetricDirection.IsLowerBetter(canonicalMetric));
    }
}
