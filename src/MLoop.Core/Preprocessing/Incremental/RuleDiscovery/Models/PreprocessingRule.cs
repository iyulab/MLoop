namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

/// <summary>
/// Base class for all preprocessing rules discovered from data samples.
/// Rules can be auto-fixable or require human-in-the-loop (HITL) approval.
/// </summary>
public class PreprocessingRule
{
    /// <summary>
    /// Unique identifier for this rule.
    /// Format: {RuleType}_{ColumnName}_{Pattern}
    /// Example: "MissingValue_Age_Nulls"
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Type of preprocessing rule.
    /// </summary>
    public required PreprocessingRuleType Type { get; init; }

    /// <summary>
    /// Column name(s) this rule applies to.
    /// </summary>
    public required IReadOnlyList<string> ColumnNames { get; init; }

    /// <summary>
    /// Human-readable description of the rule.
    /// Example: "Replace NULL values in Age column with median (35)"
    /// </summary>
    public required string Description { get; init; }

    /// <summary>
    /// Detected pattern that led to this rule.
    /// </summary>
    public required PatternType PatternType { get; init; }

    /// <summary>
    /// Confidence score for this rule (0-1).
    /// Higher confidence = more stable across samples.
    /// </summary>
    public double Confidence { get; set; } = 0.0;

    /// <summary>
    /// Statistical confidence details.
    /// </summary>
    public ConfidenceScore? ConfidenceScore { get; set; }

    /// <summary>
    /// Whether this rule requires human-in-the-loop approval.
    /// Auto-fixable rules can be applied automatically.
    /// </summary>
    public required bool RequiresHITL { get; init; }

    /// <summary>
    /// Priority for applying this rule (1-10).
    /// Higher priority = applied first.
    /// 10 = Critical (data integrity)
    /// 5 = Normal (data quality)
    /// 1 = Low (cosmetic)
    /// </summary>
    public required int Priority { get; init; }

    /// <summary>
    /// Rule-specific parameters as key-value pairs.
    /// Example: { "impute_value": "median", "target": 35.0 }
    /// </summary>
    public Dictionary<string, object> Parameters { get; init; } = new();

    /// <summary>
    /// Example rows where this rule applies (for HITL context).
    /// </summary>
    public IReadOnlyList<string>? Examples { get; init; }

    /// <summary>
    /// Suggested action to take.
    /// Example: "Impute with median" or "Remove outliers"
    /// </summary>
    public string? SuggestedAction { get; init; }

    /// <summary>
    /// Whether this rule has been approved by a human (for HITL rules).
    /// </summary>
    public bool IsApproved { get; set; }

    /// <summary>
    /// User feedback or modification (for HITL rules).
    /// </summary>
    public string? UserFeedback { get; set; }

    /// <summary>
    /// Number of rows this rule affects.
    /// </summary>
    public int AffectedRows { get; set; }

    /// <summary>
    /// When this rule was discovered.
    /// </summary>
    public DateTime DiscoveredAt { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// Which sample stage this rule was discovered in (1-5).
    /// </summary>
    public int DiscoveredInStage { get; init; }

    /// <summary>
    /// Whether this rule is active and should be applied.
    /// </summary>
    public bool IsActive { get; set; } = true;

    /// <summary>
    /// Unique string representation of this rule for equality comparison.
    /// </summary>
    public virtual string GetSignature()
    {
        var columns = string.Join(",", ColumnNames.OrderBy(c => c));
        return $"{Type}|{columns}|{PatternType}|{Description}";
    }

    /// <summary>
    /// Check if this rule is equivalent to another (same signature).
    /// </summary>
    public virtual bool IsEquivalentTo(PreprocessingRule other)
    {
        return GetSignature() == other.GetSignature();
    }
}
