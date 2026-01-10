using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleApplication.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleApplication.Contracts;

/// <summary>
/// Applies preprocessing rules to DataFrame instances.
/// Supports individual rule application and bulk processing.
/// </summary>
public interface IRuleApplier
{
    /// <summary>
    /// Applies a single preprocessing rule to a DataFrame.
    /// </summary>
    /// <param name="data">The DataFrame to transform.</param>
    /// <param name="rule">The preprocessing rule to apply.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Result of the rule application including affected rows and duration.</returns>
    Task<RuleApplicationResult> ApplyRuleAsync(
        DataFrame data,
        PreprocessingRule rule,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Applies multiple preprocessing rules in sequence to a DataFrame.
    /// </summary>
    /// <param name="data">The DataFrame to transform.</param>
    /// <param name="rules">The preprocessing rules to apply (in order).</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Bulk application result with individual rule results and aggregated statistics.</returns>
    /// <remarks>
    /// Rules are applied in order. If a rule fails and ContinueOnRuleFailure is true,
    /// the next rule will still be attempted. The DataFrame is modified in place.
    /// </remarks>
    Task<BulkApplicationResult> ApplyRulesAsync(
        DataFrame data,
        IReadOnlyList<PreprocessingRule> rules,
        IProgress<RuleApplicationProgress>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Validates that a rule can be applied to the given DataFrame.
    /// </summary>
    /// <param name="data">The DataFrame to validate against.</param>
    /// <param name="rule">The rule to validate.</param>
    /// <returns>True if the rule can be applied, false otherwise.</returns>
    /// <remarks>
    /// Checks:
    /// <list type="bullet">
    /// <item><description>Required columns exist</description></item>
    /// <item><description>Column types are compatible</description></item>
    /// <item><description>Rule parameters are valid</description></item>
    /// </list>
    /// </remarks>
    bool ValidateRule(DataFrame data, PreprocessingRule rule);
}
