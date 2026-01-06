using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleApplication.Models;

/// <summary>
/// Result of applying a single preprocessing rule to a DataFrame.
/// </summary>
public sealed class RuleApplicationResult
{
    /// <summary>
    /// The rule that was applied.
    /// </summary>
    public required PreprocessingRule Rule { get; init; }

    /// <summary>
    /// Number of rows affected by this rule.
    /// </summary>
    public required int RowsAffected { get; init; }

    /// <summary>
    /// Number of rows skipped (didn't match rule criteria).
    /// </summary>
    public required int RowsSkipped { get; init; }

    /// <summary>
    /// How long the rule application took.
    /// </summary>
    public required TimeSpan Duration { get; init; }

    /// <summary>
    /// Whether the rule was applied successfully.
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Error message if application failed.
    /// </summary>
    public string? ErrorMessage { get; init; }

    /// <summary>
    /// Optional validation message or warning.
    /// </summary>
    public string? ValidationMessage { get; init; }
}

/// <summary>
/// Result of applying multiple rules to a DataFrame.
/// </summary>
public sealed class BulkApplicationResult
{
    /// <summary>
    /// Total number of rules attempted.
    /// </summary>
    public required int TotalRules { get; init; }

    /// <summary>
    /// Number of rules applied successfully.
    /// </summary>
    public required int SuccessfulRules { get; init; }

    /// <summary>
    /// Number of rules that failed.
    /// </summary>
    public required int FailedRules { get; init; }

    /// <summary>
    /// Individual results for each rule.
    /// </summary>
    public required List<RuleApplicationResult> Results { get; init; }

    /// <summary>
    /// Total time taken to apply all rules.
    /// </summary>
    public required TimeSpan TotalDuration { get; init; }

    /// <summary>
    /// Total rows affected across all rules.
    /// </summary>
    public int TotalRowsAffected => Results.Sum(r => r.RowsAffected);

    /// <summary>
    /// Overall success rate (0.0 to 1.0).
    /// </summary>
    public double SuccessRate => TotalRules > 0 ? (double)SuccessfulRules / TotalRules : 0.0;
}

/// <summary>
/// Progress information for rule application.
/// </summary>
public sealed class RuleApplicationProgress
{
    /// <summary>
    /// Current rule being applied.
    /// </summary>
    public required PreprocessingRule CurrentRule { get; init; }

    /// <summary>
    /// Rule index (0-based).
    /// </summary>
    public required int RuleIndex { get; init; }

    /// <summary>
    /// Total number of rules to apply.
    /// </summary>
    public required int TotalRules { get; init; }

    /// <summary>
    /// Progress percentage (0.0 to 1.0).
    /// </summary>
    public double Percentage => TotalRules > 0 ? (double)RuleIndex / TotalRules : 0.0;

    /// <summary>
    /// Current status message.
    /// </summary>
    public string? Message { get; init; }
}
