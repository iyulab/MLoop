namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

/// <summary>
/// Types of data quality patterns that can be detected in samples.
/// </summary>
public enum PatternType
{
    /// <summary>
    /// Missing values: nulls, "N/A", empty strings, "NULL", etc.
    /// </summary>
    MissingValue,

    /// <summary>
    /// Type inconsistencies: mixed numeric/string, coercion issues.
    /// </summary>
    TypeInconsistency,

    /// <summary>
    /// Format variations: date formats, number formats, boolean representations.
    /// </summary>
    FormatVariation,

    /// <summary>
    /// Outlier anomalies: statistical outliers, domain-specific anomalies.
    /// </summary>
    OutlierAnomaly,

    /// <summary>
    /// Category variations: typos, case differences, similar categories.
    /// </summary>
    CategoryVariation,

    /// <summary>
    /// Encoding issues: UTF-8 vs Latin-1, mojibake, character corruption.
    /// </summary>
    EncodingIssue,

    /// <summary>
    /// Whitespace issues: leading/trailing spaces, multiple spaces, tabs.
    /// </summary>
    WhitespaceIssue,

    /// <summary>
    /// Business rule violations: domain-specific constraints.
    /// </summary>
    BusinessRule
}
