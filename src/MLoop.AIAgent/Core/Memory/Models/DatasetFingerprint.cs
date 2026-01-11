using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Data.Analysis;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Core.Memory.Models;

/// <summary>
/// Fingerprint of a dataset for pattern matching.
/// Captures structural characteristics for similarity comparison.
/// </summary>
public sealed class DatasetFingerprint
{
    /// <summary>
    /// List of column names in the dataset.
    /// </summary>
    public List<string> ColumnNames { get; set; } = [];

    /// <summary>
    /// Mapping of column names to their data types.
    /// </summary>
    public Dictionary<string, string> ColumnTypes { get; set; } = [];

    /// <summary>
    /// Number of rows in the dataset.
    /// </summary>
    public long RowCount { get; set; }

    /// <summary>
    /// Approximate row count category (Small, Medium, Large, VeryLarge).
    /// </summary>
    public string SizeCategory { get; set; } = "Medium";

    /// <summary>
    /// Domain keywords extracted from column names.
    /// </summary>
    public List<string> DomainKeywords { get; set; } = [];

    /// <summary>
    /// Detected task type (classification, regression, etc.).
    /// </summary>
    public string? DetectedTaskType { get; set; }

    /// <summary>
    /// Label column name if detected.
    /// </summary>
    public string? LabelColumn { get; set; }

    /// <summary>
    /// Percentage of numeric columns (0.0 to 1.0).
    /// </summary>
    public double NumericRatio { get; set; }

    /// <summary>
    /// Percentage of categorical columns (0.0 to 1.0).
    /// </summary>
    public double CategoricalRatio { get; set; }

    /// <summary>
    /// Percentage of missing values across all columns (0.0 to 1.0).
    /// </summary>
    public double MissingValueRatio { get; set; }

    /// <summary>
    /// Hash of the fingerprint for quick comparison.
    /// </summary>
    public string? Hash { get; set; }

    /// <summary>
    /// Creates a fingerprint from a DataFrame.
    /// </summary>
    public static DatasetFingerprint FromDataFrame(DataFrame df, string? labelColumn = null)
    {
        var fingerprint = new DatasetFingerprint
        {
            ColumnNames = df.Columns.Select(c => c.Name).ToList(),
            RowCount = df.Rows.Count,
            LabelColumn = labelColumn
        };

        int numericCount = 0;
        int categoricalCount = 0;
        long totalMissing = 0;
        long totalCells = df.Rows.Count * df.Columns.Count;

        foreach (var col in df.Columns)
        {
            var typeName = col.DataType.Name;
            fingerprint.ColumnTypes[col.Name] = typeName;

            if (IsNumericType(typeName))
                numericCount++;
            else
                categoricalCount++;

            totalMissing += col.NullCount;
        }

        fingerprint.NumericRatio = df.Columns.Count > 0
            ? (double)numericCount / df.Columns.Count
            : 0;
        fingerprint.CategoricalRatio = df.Columns.Count > 0
            ? (double)categoricalCount / df.Columns.Count
            : 0;
        fingerprint.MissingValueRatio = totalCells > 0
            ? (double)totalMissing / totalCells
            : 0;

        fingerprint.SizeCategory = df.Rows.Count switch
        {
            < 1_000 => "Small",
            < 100_000 => "Medium",
            < 1_000_000 => "Large",
            _ => "VeryLarge"
        };

        fingerprint.DomainKeywords = ExtractDomainKeywords(fingerprint.ColumnNames);
        fingerprint.Hash = fingerprint.ComputeHash();

        return fingerprint;
    }

    /// <summary>
    /// Creates a fingerprint from a DataAnalysisReport.
    /// Used to bridge analysis results with memory-based pattern matching.
    /// </summary>
    public static DatasetFingerprint FromAnalysisReport(DataAnalysisReport report, string? labelColumn = null)
    {
        var fingerprint = new DatasetFingerprint
        {
            ColumnNames = report.Columns.Select(c => c.Name).ToList(),
            RowCount = report.RowCount,
            LabelColumn = labelColumn ?? report.RecommendedTarget?.ColumnName
        };

        int numericCount = 0;
        int categoricalCount = 0;
        long totalMissing = 0;
        long totalCells = report.RowCount * report.ColumnCount;

        foreach (var col in report.Columns)
        {
            // Map DataType enum to string type name
            var typeName = col.InferredType switch
            {
                DataType.Numeric => "Double",
                DataType.DateTime => "DateTime",
                DataType.Boolean => "Boolean",
                DataType.Categorical => "String",
                DataType.Text => "String",
                _ => "Object"
            };
            fingerprint.ColumnTypes[col.Name] = typeName;

            if (col.InferredType == DataType.Numeric)
                numericCount++;
            else if (col.InferredType is DataType.Categorical or DataType.Text or DataType.Boolean)
                categoricalCount++;

            totalMissing += col.NullCount;
        }

        fingerprint.NumericRatio = report.ColumnCount > 0
            ? (double)numericCount / report.ColumnCount
            : 0;
        fingerprint.CategoricalRatio = report.ColumnCount > 0
            ? (double)categoricalCount / report.ColumnCount
            : 0;
        fingerprint.MissingValueRatio = totalCells > 0
            ? (double)totalMissing / totalCells
            : 0;

        fingerprint.SizeCategory = report.RowCount switch
        {
            < 1_000 => "Small",
            < 100_000 => "Medium",
            < 1_000_000 => "Large",
            _ => "VeryLarge"
        };

        // Infer task type from recommended target if available
        if (report.RecommendedTarget != null && report.RecommendedTarget.ProblemType != MLProblemType.Unknown)
        {
            fingerprint.DetectedTaskType = report.RecommendedTarget.ProblemType.ToString().ToLowerInvariant();
        }

        fingerprint.DomainKeywords = ExtractDomainKeywords(fingerprint.ColumnNames);
        fingerprint.Hash = fingerprint.ComputeHash();

        return fingerprint;
    }

    /// <summary>
    /// Generates a human-readable description of the fingerprint.
    /// Used for semantic search embedding.
    /// </summary>
    public string Describe()
    {
        var sb = new StringBuilder();
        sb.Append($"Dataset with {RowCount} rows ({SizeCategory}), ");
        sb.Append($"{ColumnNames.Count} columns ");
        sb.Append($"({NumericRatio:P0} numeric, {CategoricalRatio:P0} categorical). ");

        if (MissingValueRatio > 0.01)
            sb.Append($"Missing values: {MissingValueRatio:P1}. ");

        if (!string.IsNullOrEmpty(DetectedTaskType))
            sb.Append($"Task: {DetectedTaskType}. ");

        if (!string.IsNullOrEmpty(LabelColumn))
            sb.Append($"Label: {LabelColumn}. ");

        if (DomainKeywords.Count > 0)
            sb.Append($"Domain: {string.Join(", ", DomainKeywords.Take(5))}. ");

        return sb.ToString();
    }

    private string ComputeHash()
    {
        var data = JsonSerializer.Serialize(new
        {
            ColumnNames = string.Join(",", ColumnNames.OrderBy(x => x)),
            ColumnTypes = string.Join(",", ColumnTypes.OrderBy(x => x.Key).Select(x => $"{x.Key}:{x.Value}")),
            SizeCategory,
            NumericRatio = Math.Round(NumericRatio, 2),
            CategoricalRatio = Math.Round(CategoricalRatio, 2)
        });

        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(data));
        return Convert.ToHexString(bytes)[..16];
    }

    private static bool IsNumericType(string typeName)
    {
        return typeName is "Double" or "Single" or "Int32" or "Int64"
            or "Int16" or "Byte" or "Decimal" or "Float";
    }

    private static List<string> ExtractDomainKeywords(List<string> columnNames)
    {
        var keywords = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var commonWords = new HashSet<string> { "id", "name", "date", "time", "value", "count", "type" };

        foreach (var name in columnNames)
        {
            // Split by common separators
            var parts = name.Split(['_', '-', ' '], StringSplitOptions.RemoveEmptyEntries);
            foreach (var part in parts)
            {
                var lower = part.ToLowerInvariant();
                if (lower.Length > 2 && !commonWords.Contains(lower))
                    keywords.Add(lower);
            }
        }

        return keywords.Take(10).ToList();
    }
}
