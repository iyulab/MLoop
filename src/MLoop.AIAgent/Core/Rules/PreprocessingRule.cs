using System.Text.Json.Serialization;

namespace MLoop.AIAgent.Core.Rules;

/// <summary>
/// Types of preprocessing rules discovered by the agent.
/// </summary>
public enum RuleType
{
    // Auto-resolvable (no HITL needed)
    /// <summary>Standardize multiple date formats to ISO-8601.</summary>
    DateFormatStandardization,

    /// <summary>Convert encoding to UTF-8.</summary>
    EncodingNormalization,

    /// <summary>Trim whitespace and collapse multiple spaces.</summary>
    WhitespaceNormalization,

    /// <summary>Standardize case in categorical values.</summary>
    CaseNormalization,

    /// <summary>Parse numeric strings to actual numbers.</summary>
    NumericParsing,

    /// <summary>Convert units (e.g., lb to kg).</summary>
    UnitConversion,

    // Requires HITL
    /// <summary>Strategy for handling missing values.</summary>
    MissingValueStrategy,

    /// <summary>How to handle statistical outliers.</summary>
    OutlierHandling,

    /// <summary>Mapping for unknown category values.</summary>
    UnknownCategoryMapping,

    /// <summary>Domain-specific business logic decision.</summary>
    BusinessLogicDecision,

    /// <summary>How to handle duplicate records.</summary>
    DuplicateHandling,

    /// <summary>Custom transformation defined by user.</summary>
    CustomTransformation
}

/// <summary>
/// Severity of a data issue requiring preprocessing.
/// </summary>
public enum IssueSeverity
{
    /// <summary>Informational, may not need action.</summary>
    Info,

    /// <summary>Minor issue, can be auto-fixed.</summary>
    Low,

    /// <summary>Moderate issue, should be addressed.</summary>
    Medium,

    /// <summary>Significant issue affecting data quality.</summary>
    High,

    /// <summary>Critical issue that may invalidate analysis.</summary>
    Critical
}

/// <summary>
/// A preprocessing rule discovered from data patterns.
/// </summary>
public class PreprocessingRule
{
    /// <summary>Unique identifier for the rule.</summary>
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..8];

    /// <summary>Human-readable name.</summary>
    public required string Name { get; set; }

    /// <summary>Detailed description of what this rule does.</summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>Type of preprocessing rule.</summary>
    public RuleType Type { get; set; }

    /// <summary>Column(s) this rule applies to.</summary>
    public List<string> Columns { get; set; } = [];

    /// <summary>Pattern to match (regex or literal).</summary>
    public string? Pattern { get; set; }

    /// <summary>Transformation to apply.</summary>
    public required string Transformation { get; set; }

    /// <summary>Whether this rule requires HITL approval.</summary>
    public bool RequiresHITL { get; set; }

    /// <summary>Confidence score (0.0 - 1.0).</summary>
    public double Confidence { get; set; }

    /// <summary>Number of records matching this pattern.</summary>
    public int MatchCount { get; set; }

    /// <summary>Percentage of sample affected.</summary>
    public double AffectedPercentage { get; set; }

    /// <summary>Severity of the issue this rule addresses.</summary>
    public IssueSeverity Severity { get; set; } = IssueSeverity.Medium;

    /// <summary>Whether this rule has been approved (by auto or HITL).</summary>
    public bool IsApproved { get; set; }

    /// <summary>Who approved this rule (System or user identifier).</summary>
    public string? ApprovedBy { get; set; }

    /// <summary>Timestamp when rule was discovered.</summary>
    public DateTime DiscoveredAt { get; set; } = DateTime.UtcNow;

    /// <summary>Sample stage when rule was discovered (1-5).</summary>
    public int DiscoveredAtStage { get; set; }

    /// <summary>
    /// Determines if this is an auto-resolvable rule type.
    /// </summary>
    [JsonIgnore]
    public bool IsAutoResolvable => Type switch
    {
        RuleType.DateFormatStandardization => true,
        RuleType.EncodingNormalization => true,
        RuleType.WhitespaceNormalization => true,
        RuleType.CaseNormalization => true,
        RuleType.NumericParsing => true,
        RuleType.UnitConversion => true,
        _ => false
    };

    /// <summary>
    /// Creates a summary string for display.
    /// </summary>
    public string ToSummary()
    {
        var status = IsApproved ? "✅" : (RequiresHITL ? "⏳" : "🔄");
        var columns = Columns.Count > 0 ? string.Join(", ", Columns) : "all";
        return $"{status} [{Type}] {Name}: {columns} → {Transformation} (Confidence: {Confidence:P0})";
    }
}

/// <summary>
/// Result of validating a rule against new data.
/// </summary>
public class RuleValidationResult
{
    /// <summary>The rule being validated.</summary>
    public required PreprocessingRule Rule { get; set; }

    /// <summary>Whether the rule is still valid.</summary>
    public bool IsValid { get; set; }

    /// <summary>Updated confidence score.</summary>
    public double UpdatedConfidence { get; set; }

    /// <summary>Number of records that matched.</summary>
    public int MatchCount { get; set; }

    /// <summary>Number of records that didn't match expected pattern.</summary>
    public int ExceptionCount { get; set; }

    /// <summary>New patterns discovered that the rule doesn't cover.</summary>
    public List<string> NewPatterns { get; set; } = [];

    /// <summary>Validation message.</summary>
    public string Message { get; set; } = string.Empty;
}
