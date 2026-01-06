namespace MLoop.Core.Preprocessing.Incremental.HITL.Models;

/// <summary>
/// Summary statistics for HITL decisions.
/// </summary>
public sealed class HITLDecisionSummary
{
    /// <summary>
    /// Total number of decisions logged.
    /// </summary>
    public int TotalDecisions { get; init; }

    /// <summary>
    /// Number of times AI recommendations were followed.
    /// </summary>
    public int RecommendationsFollowed { get; init; }

    /// <summary>
    /// Number of times AI recommendations were overridden.
    /// </summary>
    public int RecommendationsOverridden { get; init; }

    /// <summary>
    /// Average time taken to make decisions (in seconds).
    /// </summary>
    public double AverageDecisionTimeSeconds { get; init; }

    /// <summary>
    /// Distribution of decision types.
    /// </summary>
    public IReadOnlyDictionary<string, int> DecisionTypeDistribution { get; init; } =
        new Dictionary<string, int>();

    /// <summary>
    /// Distribution of actions selected.
    /// </summary>
    public IReadOnlyDictionary<ActionType, int> ActionDistribution { get; init; } =
        new Dictionary<ActionType, int>();

    /// <summary>
    /// Date range of decisions in this summary.
    /// </summary>
    public DateTime EarliestDecision { get; init; }

    /// <summary>
    /// Latest decision timestamp.
    /// </summary>
    public DateTime LatestDecision { get; init; }
}
