using MLoop.AIAgent.Core.Models;
using MLoop.Core.Data;

namespace MLoop.AIAgent.Core;

/// <summary>
/// Pre-training dataset compatibility checker.
/// Validates datasets before ML training to prevent common failures.
/// </summary>
public class DatasetCompatibilityChecker
{
    /// <summary>
    /// Validates a dataset for ML training compatibility.
    /// </summary>
    /// <param name="report">Data analysis report from DataAnalyzer.</param>
    /// <param name="labelColumn">Target label column name.</param>
    /// <param name="taskType">ML task type (regression, binary-classification, etc.).</param>
    /// <returns>Compatibility check result with issues and suggestions.</returns>
    public CompatibilityResult Check(
        DataAnalysisReport report,
        string? labelColumn = null,
        string? taskType = null)
    {
        var issues = new List<CompatibilityIssue>();

        // Check 1: Minimum row count
        if (report.RowCount < 10)
        {
            issues.Add(new CompatibilityIssue
            {
                Severity = IssueSeverity.Critical,
                Category = IssueCategory.DataSize,
                Message = $"Dataset has only {report.RowCount} rows. ML.NET requires at least 10 rows for training.",
                Suggestion = "Add more data samples or consider alternative approaches for very small datasets."
            });
        }
        else if (report.RowCount < 100)
        {
            issues.Add(new CompatibilityIssue
            {
                Severity = IssueSeverity.Warning,
                Category = IssueCategory.DataSize,
                Message = $"Dataset has only {report.RowCount} rows. This may lead to overfitting.",
                Suggestion = "Consider using cross-validation or gathering more data for reliable model training."
            });
        }

        // Check 2: Label column validation
        if (!string.IsNullOrEmpty(labelColumn))
        {
            var labelColumnAnalysis = report.Columns.FirstOrDefault(
                c => c.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase));

            if (labelColumnAnalysis == null)
            {
                issues.Add(new CompatibilityIssue
                {
                    Severity = IssueSeverity.Critical,
                    Category = IssueCategory.Schema,
                    Message = $"Label column '{labelColumn}' not found in dataset.",
                    Suggestion = $"Available columns: {string.Join(", ", report.Columns.Select(c => c.Name))}"
                });
            }
            else
            {
                // Check missing values in label
                if (labelColumnAnalysis.NullCount > 0)
                {
                    var missingPct = labelColumnAnalysis.MissingPercentage;
                    if (missingPct > 50)
                    {
                        issues.Add(new CompatibilityIssue
                        {
                            Severity = IssueSeverity.Critical,
                            Category = IssueCategory.MissingValues,
                            Message = $"Label column '{labelColumn}' has {missingPct:F1}% missing values.",
                            Suggestion = "Use --drop-missing-labels flag to auto-remove rows with missing labels."
                        });
                    }
                    else if (missingPct > 10)
                    {
                        issues.Add(new CompatibilityIssue
                        {
                            Severity = IssueSeverity.Warning,
                            Category = IssueCategory.MissingValues,
                            Message = $"Label column '{labelColumn}' has {missingPct:F1}% missing values ({labelColumnAnalysis.NullCount} rows).",
                            Suggestion = "Use --drop-missing-labels flag to handle missing label values."
                        });
                    }
                }

                // Check label-task compatibility
                if (!string.IsNullOrEmpty(taskType))
                {
                    ValidateLabelTaskCompatibility(labelColumnAnalysis, taskType, issues);
                }
            }
        }

        // Check 3: Excessive missing values
        var highMissingColumns = report.Columns
            .Where(c => c.MissingPercentage > 50)
            .ToList();

        if (highMissingColumns.Count > 0)
        {
            foreach (var col in highMissingColumns)
            {
                issues.Add(new CompatibilityIssue
                {
                    Severity = IssueSeverity.Warning,
                    Category = IssueCategory.MissingValues,
                    Message = $"Column '{col.Name}' has {col.MissingPercentage:F1}% missing values.",
                    Suggestion = "Consider dropping this column or using imputation strategies."
                });
            }
        }

        // Check 4: All features are same type
        var numericCols = report.Columns.Count(c => c.InferredType == DataType.Numeric);
        var categoricalCols = report.Columns.Count(c => c.InferredType is DataType.Categorical or DataType.Text);

        if (numericCols == 0 && categoricalCols == report.ColumnCount)
        {
            issues.Add(new CompatibilityIssue
            {
                Severity = IssueSeverity.Info,
                Category = IssueCategory.Schema,
                Message = "All columns are categorical. ML.NET will encode them automatically.",
                Suggestion = "Consider one-hot encoding for high-cardinality categorical features."
            });
        }

        // Check 5: High cardinality categorical columns
        var highCardinalityColumns = report.Columns
            .Where(c => c.InferredType == DataType.Categorical && c.UniqueCount > 100)
            .ToList();

        foreach (var col in highCardinalityColumns)
        {
            issues.Add(new CompatibilityIssue
            {
                Severity = IssueSeverity.Warning,
                Category = IssueCategory.Schema,
                Message = $"Column '{col.Name}' has high cardinality ({col.UniqueCount} unique values).",
                Suggestion = "Consider feature hashing or grouping rare categories."
            });
        }

        // Check 6: Date/time columns
        var dateColumns = report.Columns.Where(c => c.InferredType == DataType.DateTime).ToList();
        if (dateColumns.Count > 0)
        {
            issues.Add(new CompatibilityIssue
            {
                Severity = IssueSeverity.Info,
                Category = IssueCategory.Schema,
                Message = $"Dataset contains {dateColumns.Count} datetime column(s): {string.Join(", ", dateColumns.Select(c => c.Name))}.",
                Suggestion = "Consider extracting date features (year, month, day, dayOfWeek) for better ML performance."
            });
        }

        return new CompatibilityResult
        {
            IsCompatible = !issues.Any(i => i.Severity == IssueSeverity.Critical),
            Issues = issues,
            Summary = BuildSummary(report, issues)
        };
    }

    private void ValidateLabelTaskCompatibility(
        ColumnAnalysis labelColumn,
        string taskType,
        List<CompatibilityIssue> issues)
    {
        var normalizedTask = taskType.ToLowerInvariant().Replace("-", "").Replace("_", "");

        switch (normalizedTask)
        {
            case "binaryclassification":
                if (labelColumn.UniqueCount > 2)
                {
                    issues.Add(new CompatibilityIssue
                    {
                        Severity = IssueSeverity.Warning,
                        Category = IssueCategory.TaskMismatch,
                        Message = $"Label column has {labelColumn.UniqueCount} unique values, but binary classification expects 2 classes.",
                        Suggestion = "Consider using multiclass-classification or binarizing the label column."
                    });
                }
                break;

            case "multiclassclassification":
                if (labelColumn.UniqueCount == 2)
                {
                    issues.Add(new CompatibilityIssue
                    {
                        Severity = IssueSeverity.Info,
                        Category = IssueCategory.TaskMismatch,
                        Message = "Label column has only 2 unique values. Binary classification might be more efficient.",
                        Suggestion = "Consider using binary-classification for better performance."
                    });
                }
                else if (labelColumn.UniqueCount > 100)
                {
                    issues.Add(new CompatibilityIssue
                    {
                        Severity = IssueSeverity.Warning,
                        Category = IssueCategory.TaskMismatch,
                        Message = $"Label column has {labelColumn.UniqueCount} classes, which may be too many for effective classification.",
                        Suggestion = "Consider grouping similar classes or using regression if appropriate."
                    });
                }
                break;

            case "regression":
                if (labelColumn.InferredType != DataType.Numeric)
                {
                    issues.Add(new CompatibilityIssue
                    {
                        Severity = IssueSeverity.Critical,
                        Category = IssueCategory.TaskMismatch,
                        Message = $"Label column '{labelColumn.Name}' is not numeric, but regression requires numeric targets.",
                        Suggestion = "Convert the label to numeric or use classification task type."
                    });
                }
                break;
        }
    }

    private string BuildSummary(DataAnalysisReport report, List<CompatibilityIssue> issues)
    {
        var criticalCount = issues.Count(i => i.Severity == IssueSeverity.Critical);
        var warningCount = issues.Count(i => i.Severity == IssueSeverity.Warning);
        var infoCount = issues.Count(i => i.Severity == IssueSeverity.Info);

        if (criticalCount > 0)
        {
            return $"Dataset NOT compatible for ML training. {criticalCount} critical issue(s) must be resolved.";
        }

        if (warningCount > 0)
        {
            return $"Dataset compatible with {warningCount} warning(s). Review suggestions for optimal results.";
        }

        if (infoCount > 0)
        {
            return $"Dataset ready for ML training. {infoCount} suggestion(s) for optimization.";
        }

        return "Dataset fully compatible for ML training.";
    }
}

/// <summary>
/// Result of dataset compatibility check.
/// </summary>
public sealed class CompatibilityResult
{
    /// <summary>
    /// Whether the dataset is compatible for ML training (no critical issues).
    /// </summary>
    public bool IsCompatible { get; init; }

    /// <summary>
    /// List of detected issues.
    /// </summary>
    public List<CompatibilityIssue> Issues { get; init; } = [];

    /// <summary>
    /// Human-readable summary.
    /// </summary>
    public string Summary { get; init; } = string.Empty;

    /// <summary>
    /// Gets critical issues that block training.
    /// </summary>
    public IEnumerable<CompatibilityIssue> CriticalIssues =>
        Issues.Where(i => i.Severity == IssueSeverity.Critical);

    /// <summary>
    /// Gets warnings that may affect training quality.
    /// </summary>
    public IEnumerable<CompatibilityIssue> Warnings =>
        Issues.Where(i => i.Severity == IssueSeverity.Warning);
}

/// <summary>
/// Individual compatibility issue.
/// </summary>
public sealed class CompatibilityIssue
{
    /// <summary>
    /// Issue severity level.
    /// </summary>
    public IssueSeverity Severity { get; init; }

    /// <summary>
    /// Issue category.
    /// </summary>
    public IssueCategory Category { get; init; }

    /// <summary>
    /// Issue description.
    /// </summary>
    public string Message { get; init; } = string.Empty;

    /// <summary>
    /// Suggested remediation.
    /// </summary>
    public string Suggestion { get; init; } = string.Empty;
}

/// <summary>
/// Issue severity levels.
/// </summary>
public enum IssueSeverity
{
    /// <summary>Informational - optimization suggestion.</summary>
    Info,
    /// <summary>Warning - may affect model quality.</summary>
    Warning,
    /// <summary>Critical - blocks training.</summary>
    Critical
}

/// <summary>
/// Issue categories.
/// </summary>
public enum IssueCategory
{
    /// <summary>Data size issues.</summary>
    DataSize,
    /// <summary>Schema/structure issues.</summary>
    Schema,
    /// <summary>Missing value issues.</summary>
    MissingValues,
    /// <summary>Task type mismatch issues.</summary>
    TaskMismatch,
    /// <summary>Encoding issues.</summary>
    Encoding
}
