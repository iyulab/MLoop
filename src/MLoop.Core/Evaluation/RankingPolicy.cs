using MLoop.Search;

namespace MLoop.Core.Evaluation;

/// <summary>
/// MLoop's metric vocabulary as the policy a search ranks trials by: the name is the canonical
/// metric key (<see cref="MetricNames"/>) and the direction comes from <see cref="MetricDirection"/>,
/// so the search ranks the way every other MLoop ranking does.
/// </summary>
public sealed class RankingPolicy : IMetricPolicy
{
    private RankingPolicy(string name, bool higherIsBetter, double absoluteSlack)
    {
        Name = name;
        HigherIsBetter = higherIsBetter;
        AbsoluteSlack = absoluteSlack;
    }

    public string Name { get; }

    public bool HigherIsBetter { get; }

    /// <summary>
    /// 0.05 nats of log loss and 0.02 of calibration error: the losses a classifier is driven toward zero on, where a share of
    /// the score is no slack at all. A metric on another scale (an error in the label's unit) has no such floor the policy can know.
    /// </summary>
    public double AbsoluteSlack { get; }

    /// <summary>
    /// The policy for a canonical metric key. A metric <see cref="MetricDirection"/> does not know
    /// is ranked higher-is-better, as MLoop's leaderboard ranks it.
    /// </summary>
    public static RankingPolicy For(string canonicalMetric)
    {
        if (string.IsNullOrWhiteSpace(canonicalMetric))
            throw new ArgumentException("A metric policy needs a metric name.", nameof(canonicalMetric));
        return new RankingPolicy(canonicalMetric, !MetricDirection.IsLowerBetter(canonicalMetric), canonicalMetric switch
        {
            "log_loss" => 0.05,
            "ece" => 0.02,
            _ => 0,
        });
    }
}
