namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

/// <summary>
/// Severity level of a detected pattern or data quality issue.
/// </summary>
public enum Severity
{
    /// <summary>
    /// Low severity: Minor issue, can be ignored or fixed.
    /// Example: 1-5% missing values, minor whitespace.
    /// </summary>
    Low,

    /// <summary>
    /// Medium severity: Notable issue, should be addressed.
    /// Example: 5-20% missing values, some type inconsistencies.
    /// </summary>
    Medium,

    /// <summary>
    /// High severity: Critical issue, must be resolved.
    /// Example: >20% missing values, significant data corruption.
    /// </summary>
    High,

    /// <summary>
    /// Critical severity: Blocking issue, cannot proceed.
    /// Example: >50% missing values, complete data corruption.
    /// </summary>
    Critical
}
