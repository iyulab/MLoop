namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

/// <summary>
/// Types of preprocessing rules that can be applied to data.
/// Rules are classified by whether they require human-in-the-loop (HITL) approval.
/// </summary>
public enum PreprocessingRuleType
{
    // ===== Auto-Resolvable Rules (No HITL Required) =====

    /// <summary>
    /// Standardize date formats to ISO-8601 (YYYY-MM-DD).
    /// Example: "12/31/2023" → "2023-12-31"
    /// </summary>
    DateFormatStandardization,

    /// <summary>
    /// Normalize text encoding to UTF-8.
    /// Example: Fix mojibake, encoding corruption.
    /// </summary>
    EncodingNormalization,

    /// <summary>
    /// Normalize whitespace: trim, collapse multiple spaces.
    /// Example: "  hello  world  " → "hello world"
    /// </summary>
    WhitespaceNormalization,

    /// <summary>
    /// Standardize numeric formats (remove separators, fix decimals).
    /// Example: "1,000.50" → 1000.5
    /// </summary>
    NumericFormatStandardization,

    // ===== HITL-Required Rules (Business Logic Decisions) =====

    /// <summary>
    /// Strategy for handling missing values (delete, impute, default).
    /// Requires user decision: What to do with missing data?
    /// </summary>
    MissingValueStrategy,

    /// <summary>
    /// Strategy for handling outliers (keep, remove, cap).
    /// Requires user decision: Are outliers errors or valid data?
    /// </summary>
    OutlierHandling,

    /// <summary>
    /// Mapping for unknown or rare categories (merge, keep, default).
    /// Requires user decision: How to handle new categories?
    /// </summary>
    CategoryMapping,

    /// <summary>
    /// Type conversion strategy for mixed-type columns.
    /// Requires user decision: Which type should dominate?
    /// </summary>
    TypeConversion,

    /// <summary>
    /// Domain-specific business logic decisions.
    /// Requires user decision: Business rules and constraints.
    /// </summary>
    BusinessLogicDecision
}
