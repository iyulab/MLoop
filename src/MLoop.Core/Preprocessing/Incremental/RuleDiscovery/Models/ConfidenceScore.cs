namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

/// <summary>
/// Statistical confidence score for a preprocessing rule.
/// Measures how consistently and reliably a rule applies across samples.
/// </summary>
public sealed class ConfidenceScore
{
    /// <summary>
    /// Consistency: Rule applies consistently to affected data (0-1).
    /// Example: 0.95 = rule works for 95% of cases it should apply to.
    /// Weight: 50%
    /// </summary>
    public required double Consistency { get; init; }

    /// <summary>
    /// Coverage: Percentage of data covered by this rule (0-1).
    /// Example: 0.80 = rule applies to 80% of the dataset.
    /// Weight: 30%
    /// </summary>
    public required double Coverage { get; init; }

    /// <summary>
    /// Stability: Rule definition unchanged across samples (0-1).
    /// Example: 1.0 = identical rule in stage N-1 and stage N.
    /// Weight: 20%
    /// </summary>
    public required double Stability { get; init; }

    /// <summary>
    /// Overall confidence score (0-1).
    /// Formula: Consistency * 0.5 + Coverage * 0.3 + Stability * 0.2
    /// </summary>
    public required double Overall { get; init; }

    /// <summary>
    /// Number of exceptions encountered (failures to apply rule).
    /// </summary>
    public int ExceptionCount { get; init; }

    /// <summary>
    /// Total number of attempts to apply the rule.
    /// </summary>
    public int TotalAttempts { get; init; }

    /// <summary>
    /// High confidence threshold (98%).
    /// Rules above this threshold can be applied without additional validation.
    /// </summary>
    public const double HighConfidenceThreshold = 0.98;

    /// <summary>
    /// Medium confidence threshold (90%).
    /// Rules above this threshold can be applied with caution.
    /// </summary>
    public const double MediumConfidenceThreshold = 0.90;

    /// <summary>
    /// Whether this rule has high confidence (≥98%).
    /// </summary>
    public bool IsHighConfidence => Overall >= HighConfidenceThreshold;

    /// <summary>
    /// Whether this rule has medium confidence (≥90%).
    /// </summary>
    public bool IsMediumConfidence => Overall >= MediumConfidenceThreshold;

    /// <summary>
    /// Confidence level as string (High/Medium/Low).
    /// </summary>
    public string Level => Overall switch
    {
        >= HighConfidenceThreshold => "High",
        >= MediumConfidenceThreshold => "Medium",
        _ => "Low"
    };

    /// <summary>
    /// Exception rate (0-1).
    /// </summary>
    public double ExceptionRate => TotalAttempts > 0
        ? (double)ExceptionCount / TotalAttempts
        : 0.0;
}
