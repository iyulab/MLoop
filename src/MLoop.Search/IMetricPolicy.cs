namespace MLoop.Search;

/// <summary>
/// The metric a search ranks trials by, and which way is better. The host supplies it: what a
/// metric is called and which direction is better is the host's vocabulary, not the search's.
/// </summary>
public interface IMetricPolicy
{
    /// <summary>The metric's name — the key it has in a trial's metrics.</summary>
    string Name { get; }

    /// <summary>True when a larger value is better (accuracy); false when smaller is (log loss).</summary>
    bool HigherIsBetter { get; }
}

/// <summary>Ranking by an <see cref="IMetricPolicy"/>.</summary>
public static class MetricPolicyExtensions
{
    /// <summary>
    /// The trial's value of the policy's metric, when it has a finite one. A trial without it, or
    /// with NaN or an infinity, cannot be ranked.
    /// </summary>
    public static bool TryScore(this IMetricPolicy policy, TrialResult trial, out double score)
    {
        score = double.NaN;
        return trial.Metrics is { } metrics
               && metrics.TryGetValue(policy.Name, out score)
               && double.IsFinite(score);
    }

    /// <summary>True when <paramref name="candidate"/> is strictly better than <paramref name="incumbent"/>.</summary>
    public static bool IsBetter(this IMetricPolicy policy, double candidate, double incumbent) =>
        policy.HigherIsBetter ? candidate > incumbent : candidate < incumbent;
}
