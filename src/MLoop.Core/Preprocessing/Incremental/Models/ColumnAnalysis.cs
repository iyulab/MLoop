namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Comprehensive analysis of a single DataFrame column.
/// </summary>
public sealed class ColumnAnalysis
{
    /// <summary>
    /// Gets the column name.
    /// </summary>
    public required string ColumnName { get; init; }

    /// <summary>
    /// Gets the zero-based column index in the DataFrame.
    /// </summary>
    public required int ColumnIndex { get; init; }

    /// <summary>
    /// Gets the detected data type.
    /// </summary>
    public required DataType DataType { get; init; }

    /// <summary>
    /// Gets the total number of rows in the column.
    /// </summary>
    public required long TotalRows { get; init; }

    /// <summary>
    /// Gets the count of non-null values.
    /// </summary>
    public required long NonNullCount { get; init; }

    /// <summary>
    /// Gets the count of null/missing values.
    /// </summary>
    public required long NullCount { get; init; }

    /// <summary>
    /// Gets the percentage of missing values.
    /// </summary>
    public double MissingPercentage => TotalRows > 0 ? (double)NullCount / TotalRows * 100.0 : 0.0;

    /// <summary>
    /// Gets numeric statistics (if column is numeric).
    /// </summary>
    public NumericStats? NumericStats { get; init; }

    /// <summary>
    /// Gets categorical statistics (if column is categorical).
    /// </summary>
    public CategoricalStats? CategoricalStats { get; init; }

    /// <summary>
    /// Gets whether this column is numeric.
    /// </summary>
    public bool IsNumeric => DataType is DataType.Integer or DataType.Float or DataType.Double or DataType.Decimal;

    /// <summary>
    /// Gets whether this column is categorical.
    /// </summary>
    public bool IsCategorical => DataType is DataType.String or DataType.Boolean;

    /// <summary>
    /// Gets whether this column is temporal.
    /// </summary>
    public bool IsTemporal => DataType == DataType.DateTime;

    /// <summary>
    /// Gets data quality issues detected in this column.
    /// </summary>
    public List<DataQualityIssue> QualityIssues { get; init; } = new();

    /// <summary>
    /// Gets recommended preprocessing actions for this column.
    /// </summary>
    public List<string> RecommendedActions { get; init; } = new();
}

/// <summary>
/// Detected data types for columns.
/// </summary>
public enum DataType
{
    /// <summary>
    /// Unknown or mixed type.
    /// </summary>
    Unknown,

    /// <summary>
    /// Integer numeric type.
    /// </summary>
    Integer,

    /// <summary>
    /// Float numeric type.
    /// </summary>
    Float,

    /// <summary>
    /// Double numeric type.
    /// </summary>
    Double,

    /// <summary>
    /// Decimal numeric type.
    /// </summary>
    Decimal,

    /// <summary>
    /// String/text type.
    /// </summary>
    String,

    /// <summary>
    /// Boolean type.
    /// </summary>
    Boolean,

    /// <summary>
    /// DateTime type.
    /// </summary>
    DateTime
}

/// <summary>
/// Represents a data quality issue detected in a column.
/// </summary>
public sealed class DataQualityIssue
{
    /// <summary>
    /// Gets the severity of the issue.
    /// </summary>
    public required IssueSeverity Severity { get; init; }

    /// <summary>
    /// Gets the issue type.
    /// </summary>
    public required string IssueType { get; init; }

    /// <summary>
    /// Gets the issue description.
    /// </summary>
    public required string Description { get; init; }

    /// <summary>
    /// Gets the suggested fix.
    /// </summary>
    public string? SuggestedFix { get; init; }

    /// <summary>
    /// Gets additional metadata about the issue.
    /// </summary>
    public Dictionary<string, object> Metadata { get; init; } = new();
}

/// <summary>
/// Severity levels for data quality issues.
/// </summary>
public enum IssueSeverity
{
    /// <summary>
    /// Informational - no action required.
    /// </summary>
    Info,

    /// <summary>
    /// Low severity - should be addressed but not critical.
    /// </summary>
    Low,

    /// <summary>
    /// Medium severity - should be addressed before training.
    /// </summary>
    Medium,

    /// <summary>
    /// High severity - must be addressed before training.
    /// </summary>
    High,

    /// <summary>
    /// Critical - prevents training, must be fixed immediately.
    /// </summary>
    Critical
}
