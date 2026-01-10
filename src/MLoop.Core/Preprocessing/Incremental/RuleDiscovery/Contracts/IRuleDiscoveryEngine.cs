using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;

/// <summary>
/// Interface for the rule discovery engine.
/// Orchestrates pattern detection and rule generation from samples.
/// </summary>
public interface IRuleDiscoveryEngine
{
    /// <summary>
    /// Discover preprocessing rules from a sample and its analysis.
    /// </summary>
    /// <param name="sample">DataFrame sample to analyze.</param>
    /// <param name="analysis">Pre-computed sample analysis (from SampleAnalyzer).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of discovered rules, prioritized by importance.</returns>
    Task<IReadOnlyList<PreprocessingRule>> DiscoverRulesAsync(
        DataFrame sample,
        SampleAnalysis analysis,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Calculate confidence score for a rule across two samples.
    /// Used to measure rule stability as sample size grows.
    /// </summary>
    /// <param name="rule">Rule to evaluate.</param>
    /// <param name="previousSample">Previous sample (Stage N-1).</param>
    /// <param name="currentSample">Current sample (Stage N).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Confidence score with consistency, coverage, and stability metrics.</returns>
    Task<ConfidenceScore> CalculateConfidenceAsync(
        PreprocessingRule rule,
        DataFrame previousSample,
        DataFrame currentSample,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Check if rule discovery has converged (no new patterns found).
    /// Convergence indicates that additional sampling is unlikely to discover new rules.
    /// </summary>
    /// <param name="previousRules">Rules from Stage N-1.</param>
    /// <param name="currentRules">Rules from Stage N.</param>
    /// <param name="threshold">Change threshold (default 2% = 0.02).</param>
    /// <returns>True if converged (can proceed to bulk processing).</returns>
    bool HasConverged(
        IReadOnlyList<PreprocessingRule> previousRules,
        IReadOnlyList<PreprocessingRule> currentRules,
        double threshold = 0.02);
}
