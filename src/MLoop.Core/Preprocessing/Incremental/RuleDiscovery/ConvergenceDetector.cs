using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery;

/// <summary>
/// Detects when rule discovery has converged (no new patterns found).
/// Convergence indicates that additional sampling is unlikely to discover new rules.
/// </summary>
public sealed class ConvergenceDetector
{
    /// <summary>
    /// Check if rule discovery has converged between two stages.
    /// </summary>
    /// <param name="previousRules">Rules from Stage N-1.</param>
    /// <param name="currentRules">Rules from Stage N.</param>
    /// <param name="threshold">Change threshold (default 2% = 0.02).</param>
    /// <returns>True if converged (can proceed to bulk processing).</returns>
    public bool HasConverged(
        IReadOnlyList<PreprocessingRule> previousRules,
        IReadOnlyList<PreprocessingRule> currentRules,
        double threshold = 0.02)
    {
        // No previous rules = first stage, not converged
        if (previousRules.Count == 0)
        {
            return false;
        }

        // Calculate change metrics
        var newRulesCount = CountNewRules(previousRules, currentRules);
        var modifiedRulesCount = CountModifiedRules(previousRules, currentRules);
        var removedRulesCount = CountRemovedRules(previousRules, currentRules);
        var totalChange = newRulesCount + modifiedRulesCount + removedRulesCount;

        // Calculate change rate
        var baselineCount = Math.Max(previousRules.Count, 1);
        var changeRate = (double)totalChange / baselineCount;

        // Converged if change rate is below threshold
        return changeRate <= threshold;
    }

    /// <summary>
    /// Get detailed convergence information.
    /// </summary>
    public ConvergenceInfo GetConvergenceInfo(
        IReadOnlyList<PreprocessingRule> previousRules,
        IReadOnlyList<PreprocessingRule> currentRules,
        double threshold = 0.02)
    {
        var newRulesCount = CountNewRules(previousRules, currentRules);
        var modifiedRulesCount = CountModifiedRules(previousRules, currentRules);
        var removedRulesCount = CountRemovedRules(previousRules, currentRules);
        var stableRulesCount = currentRules.Count - newRulesCount - modifiedRulesCount;

        var totalChange = newRulesCount + modifiedRulesCount + removedRulesCount;
        var baselineCount = Math.Max(previousRules.Count, 1);
        var changeRate = (double)totalChange / baselineCount;

        var hasConverged = changeRate <= threshold;

        return new ConvergenceInfo
        {
            HasConverged = hasConverged,
            ChangeRate = changeRate,
            Threshold = threshold,
            NewRules = newRulesCount,
            ModifiedRules = modifiedRulesCount,
            RemovedRules = removedRulesCount,
            StableRules = stableRulesCount,
            TotalRules = currentRules.Count,
            PreviousRules = previousRules.Count
        };
    }

    /// <summary>
    /// Count new rules (present in current but not in previous).
    /// </summary>
    private static int CountNewRules(
        IReadOnlyList<PreprocessingRule> previousRules,
        IReadOnlyList<PreprocessingRule> currentRules)
    {
        var previousSignatures = previousRules
            .Select(r => r.GetSignature())
            .ToHashSet();

        return currentRules.Count(r => !previousSignatures.Contains(r.GetSignature()));
    }

    /// <summary>
    /// Count modified rules (same signature but different parameters/confidence).
    /// </summary>
    private static int CountModifiedRules(
        IReadOnlyList<PreprocessingRule> previousRules,
        IReadOnlyList<PreprocessingRule> currentRules)
    {
        var previousRuleMap = previousRules.ToDictionary(r => r.GetSignature());
        int modifiedCount = 0;

        foreach (var currentRule in currentRules)
        {
            var signature = currentRule.GetSignature();

            if (previousRuleMap.TryGetValue(signature, out var previousRule))
            {
                // Check if confidence or affected rows changed significantly
                var confidenceChange = Math.Abs(currentRule.Confidence - previousRule.Confidence);
                var rowsChange = Math.Abs(currentRule.AffectedRows - previousRule.AffectedRows);
                var rowsChangeRatio = previousRule.AffectedRows > 0
                    ? (double)rowsChange / previousRule.AffectedRows
                    : 0.0;

                if (confidenceChange > 0.05 || rowsChangeRatio > 0.10)
                {
                    modifiedCount++;
                }
            }
        }

        return modifiedCount;
    }

    /// <summary>
    /// Count removed rules (present in previous but not in current).
    /// </summary>
    private static int CountRemovedRules(
        IReadOnlyList<PreprocessingRule> previousRules,
        IReadOnlyList<PreprocessingRule> currentRules)
    {
        var currentSignatures = currentRules
            .Select(r => r.GetSignature())
            .ToHashSet();

        return previousRules.Count(r => !currentSignatures.Contains(r.GetSignature()));
    }
}

/// <summary>
/// Detailed convergence information.
/// </summary>
public sealed class ConvergenceInfo
{
    public required bool HasConverged { get; init; }
    public required double ChangeRate { get; init; }
    public required double Threshold { get; init; }
    public required int NewRules { get; init; }
    public required int ModifiedRules { get; init; }
    public required int RemovedRules { get; init; }
    public required int StableRules { get; init; }
    public required int TotalRules { get; init; }
    public required int PreviousRules { get; init; }

    /// <summary>
    /// Human-readable convergence status.
    /// </summary>
    public string Status => HasConverged
        ? $"Converged (change rate: {ChangeRate:P1} ≤ {Threshold:P1})"
        : $"Not converged (change rate: {ChangeRate:P1} > {Threshold:P1})";

    /// <summary>
    /// Detailed change summary.
    /// </summary>
    public string Summary =>
        $"{TotalRules} rules total: " +
        $"{StableRules} stable, {NewRules} new, {ModifiedRules} modified, {RemovedRules} removed";
}
