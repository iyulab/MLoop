namespace MLoop.Core.DataQuality;

/// <summary>
/// Represents a data quality issue detected in a dataset.
/// </summary>
public class DataQualityIssue
{
    /// <summary>
    /// Type of quality issue detected.
    /// </summary>
    public required DataQualityIssueType Type { get; init; }

    /// <summary>
    /// Severity level of the issue.
    /// </summary>
    public required IssueSeverity Severity { get; init; }

    /// <summary>
    /// Column name where issue was detected (null for dataset-level issues).
    /// </summary>
    public string? ColumnName { get; init; }

    /// <summary>
    /// Human-readable description of the issue.
    /// </summary>
    public required string Description { get; init; }

    /// <summary>
    /// Suggested FilePrepper transformation to fix the issue.
    /// </summary>
    public string? SuggestedFix { get; init; }

    /// <summary>
    /// Additional metadata about the issue.
    /// </summary>
    public Dictionary<string, object>? Metadata { get; init; }

    public override string ToString() =>
        $"[{Severity}] {Type}: {Description}" +
        (ColumnName != null ? $" (Column: {ColumnName})" : "");
}

/// <summary>
/// Types of data quality issues that can be detected.
/// </summary>
public enum DataQualityIssueType
{
    /// <summary>
    /// File encoding is not UTF-8.
    /// </summary>
    EncodingIssue,

    /// <summary>
    /// Missing values detected in column.
    /// </summary>
    MissingValues,

    /// <summary>
    /// Duplicate rows detected.
    /// </summary>
    DuplicateRows,

    /// <summary>
    /// Data type inconsistency in column.
    /// </summary>
    TypeInconsistency,

    /// <summary>
    /// Statistical outliers detected.
    /// </summary>
    Outliers,

    /// <summary>
    /// Class imbalance in target column.
    /// </summary>
    ClassImbalance,

    /// <summary>
    /// Column has very high cardinality (many unique values).
    /// </summary>
    HighCardinality,

    /// <summary>
    /// Column has constant value (zero variance).
    /// </summary>
    ConstantColumn,

    /// <summary>
    /// Whitespace issues (leading/trailing spaces).
    /// </summary>
    WhitespaceIssues,

    /// <summary>
    /// Date/time format inconsistency.
    /// </summary>
    DateFormatIssue
}

/// <summary>
/// Severity level of a data quality issue.
/// </summary>
public enum IssueSeverity
{
    /// <summary>
    /// Critical issue that will likely cause training to fail.
    /// Example: All values missing in label column.
    /// </summary>
    Critical,

    /// <summary>
    /// High priority issue that significantly impacts model quality.
    /// Example: 50% missing values, severe class imbalance.
    /// </summary>
    High,

    /// <summary>
    /// Medium priority issue that may impact model quality.
    /// Example: 10-20% missing values, moderate outliers.
    /// </summary>
    Medium,

    /// <summary>
    /// Low priority issue, minor impact.
    /// Example: Trailing whitespace, minor type inconsistencies.
    /// </summary>
    Low,

    /// <summary>
    /// Informational only, no action required.
    /// Example: High cardinality detected (might be normal).
    /// </summary>
    Info
}
