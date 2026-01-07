using System.Text;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.DataQuality;

/// <summary>
/// Analyzes CSV datasets for quality issues and suggests FilePrepper transformations.
/// </summary>
public class DataQualityAnalyzer
{
    private readonly ILogger _logger;
    private readonly CsvConfiguration _csvConfig;

    /// <summary>
    /// Initializes a new instance of DataQualityAnalyzer.
    /// </summary>
    /// <param name="logger">Logger for progress reporting</param>
    public DataQualityAnalyzer(ILogger logger)
    {
        _logger = logger;
        _csvConfig = new CsvConfiguration(System.Globalization.CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true,
            MissingFieldFound = null,
            BadDataFound = null
        };
    }

    /// <summary>
    /// Analyzes a CSV file for data quality issues.
    /// </summary>
    /// <param name="csvPath">Path to CSV file</param>
    /// <param name="labelColumn">Optional label column name for target-specific analysis</param>
    /// <returns>List of detected quality issues</returns>
    public async Task<List<DataQualityIssue>> AnalyzeAsync(string csvPath, string? labelColumn = null)
    {
        var issues = new List<DataQualityIssue>();

        if (!File.Exists(csvPath))
        {
            throw new FileNotFoundException($"CSV file not found: {csvPath}");
        }

        _logger.Info("📊 Analyzing data quality...");

        // Check encoding
        await CheckEncodingAsync(csvPath, issues);

        // Load data with ML.NET for analysis using TextLoader
        var mlContext = new MLContext(seed: 0);
        IDataView dataView;

        try
        {
            // Use TextLoader for schema inference
            var textLoader = mlContext.Data.CreateTextLoader(
                new TextLoader.Options
                {
                    Separators = new[] { ',' },
                    HasHeader = true,
                    AllowQuoting = true,
                    AllowSparse = false
                });

            dataView = textLoader.Load(csvPath);
        }
        catch (Exception ex)
        {
            issues.Add(new DataQualityIssue
            {
                Type = DataQualityIssueType.EncodingIssue,
                Severity = IssueSeverity.Critical,
                Description = $"Failed to load CSV: {ex.Message}",
                SuggestedFix = "Use FilePrepper CSVCleaner to fix encoding and format issues"
            });
            return issues;
        }

        // Get schema
        var schema = dataView.Schema;
        var columnCount = schema.Count;

        if (columnCount == 0)
        {
            issues.Add(new DataQualityIssue
            {
                Type = DataQualityIssueType.TypeInconsistency,
                Severity = IssueSeverity.Critical,
                Description = "No columns found in CSV",
                SuggestedFix = "Check CSV file format and structure"
            });
            return issues;
        }

        _logger.Info($"  Found {columnCount} columns");

        // Analyze each column
        foreach (var column in schema)
        {
            await AnalyzeColumnAsync(dataView, column, labelColumn, issues);
        }

        // Check for duplicate rows
        await CheckDuplicatesAsync(csvPath, issues);

        // Check class balance for label column
        if (!string.IsNullOrEmpty(labelColumn) && schema.Any(c => c.Name == labelColumn))
        {
            await CheckClassBalanceAsync(dataView, labelColumn, issues);
        }

        _logger.Info($"  Detected {issues.Count} quality issue(s)");

        return issues;
    }

    private async Task CheckEncodingAsync(string csvPath, List<DataQualityIssue> issues)
    {
        // Read first 1KB to detect encoding
        var buffer = new byte[1024];
        int bytesRead;

        using (var fs = new FileStream(csvPath, FileMode.Open, FileAccess.Read))
        {
            bytesRead = await fs.ReadAsync(buffer, 0, buffer.Length);
        }

        // Check for UTF-8 BOM
        bool hasUtf8Bom = bytesRead >= 3 &&
                          buffer[0] == 0xEF &&
                          buffer[1] == 0xBB &&
                          buffer[2] == 0xBF;

        // Simple UTF-8 validation
        bool isUtf8 = true;
        try
        {
            var _ = Encoding.UTF8.GetString(buffer, 0, bytesRead);
        }
        catch
        {
            isUtf8 = false;
        }

        if (!isUtf8 && !hasUtf8Bom)
        {
            issues.Add(new DataQualityIssue
            {
                Type = DataQualityIssueType.EncodingIssue,
                Severity = IssueSeverity.High,
                Description = "File encoding appears to be non-UTF-8",
                SuggestedFix = "Use FilePrepper to convert encoding to UTF-8:\n" +
                              "pipeline.ConvertEncoding(Encoding.UTF8)"
            });
        }
    }

    private async Task AnalyzeColumnAsync(
        IDataView dataView,
        DataViewSchema.Column column,
        string? labelColumn,
        List<DataQualityIssue> issues)
    {
        var columnName = column.Name;

        // Check for constant columns (zero variance)
        if (await IsConstantColumnAsync(dataView, columnName))
        {
            issues.Add(new DataQualityIssue
            {
                Type = DataQualityIssueType.ConstantColumn,
                Severity = columnName == labelColumn ? IssueSeverity.Critical : IssueSeverity.Medium,
                ColumnName = columnName,
                Description = $"Column has constant value (zero variance)",
                SuggestedFix = columnName == labelColumn
                    ? "Label column must have varying values for training"
                    : "Consider removing this column as it provides no information"
            });
        }

        // Note: Detailed missing value, outlier, and type checking would require
        // reading actual data values, which could be expensive for large files.
        // For now, we provide a basic implementation that can be extended.
    }

    private async Task<bool> IsConstantColumnAsync(IDataView dataView, string columnName)
    {
        // Simple check: Get first 100 rows and check uniqueness
        var preview = dataView.Preview(maxRows: 100);
        var columnIndex = preview.Schema.Select((s, i) => new { s.Name, Index = i })
                                 .FirstOrDefault(x => x.Name == columnName)?.Index;

        if (!columnIndex.HasValue)
            return false;

        var values = new HashSet<string>();

        foreach (var row in preview.RowView)
        {
            var value = row.Values[columnIndex.Value].Value?.ToString() ?? "";
            values.Add(value);

            if (values.Count > 1)
                return false;  // Has variation
        }

        return values.Count <= 1;  // Constant if only 0 or 1 unique value
    }

    private async Task CheckDuplicatesAsync(string csvPath, List<DataQualityIssue> issues)
    {
        var lineHashes = new HashSet<string>();
        int totalRows = 0;
        int duplicateCount = 0;

        using var reader = new StreamReader(csvPath);
        using var csv = new CsvReader(reader, _csvConfig);

        // Skip header
        await csv.ReadAsync();
        csv.ReadHeader();

        // Sample first 1000 rows for performance
        while (await csv.ReadAsync() && totalRows < 1000)
        {
            var rowData = string.Join(",", csv.Parser.Record ?? Array.Empty<string>());
            totalRows++;

            if (!lineHashes.Add(rowData))
            {
                duplicateCount++;
            }
        }

        if (duplicateCount > 0)
        {
            var duplicatePercent = (duplicateCount * 100.0 / totalRows);

            issues.Add(new DataQualityIssue
            {
                Type = DataQualityIssueType.DuplicateRows,
                Severity = duplicatePercent > 10 ? IssueSeverity.High :
                           duplicatePercent > 5 ? IssueSeverity.Medium : IssueSeverity.Low,
                Description = $"Found {duplicateCount} duplicate rows ({duplicatePercent:F1}% of sampled data)",
                SuggestedFix = "Use FilePrepper to remove duplicates:\n" +
                              "pipeline.DropDuplicates()",
                Metadata = new Dictionary<string, object>
                {
                    ["DuplicateCount"] = duplicateCount,
                    ["SampledRows"] = totalRows,
                    ["DuplicatePercent"] = duplicatePercent
                }
            });
        }
    }

    private async Task CheckClassBalanceAsync(IDataView dataView, string labelColumn, List<DataQualityIssue> issues)
    {
        var preview = dataView.Preview(maxRows: 1000);
        var labelIndex = preview.Schema.Select((s, i) => new { s.Name, Index = i })
                                .FirstOrDefault(x => x.Name == labelColumn)?.Index;

        if (!labelIndex.HasValue)
            return;

        var labelCounts = new Dictionary<string, int>();

        foreach (var row in preview.RowView)
        {
            var label = row.Values[labelIndex.Value].Value?.ToString() ?? "";
            labelCounts.TryGetValue(label, out var count);
            labelCounts[label] = count + 1;
        }

        if (labelCounts.Count >= 2)
        {
            var minCount = labelCounts.Values.Min();
            var maxCount = labelCounts.Values.Max();
            var imbalanceRatio = maxCount / (double)minCount;

            if (imbalanceRatio > 3)  // More than 3:1 ratio
            {
                issues.Add(new DataQualityIssue
                {
                    Type = DataQualityIssueType.ClassImbalance,
                    Severity = imbalanceRatio > 10 ? IssueSeverity.High : IssueSeverity.Medium,
                    ColumnName = labelColumn,
                    Description = $"Class imbalance detected: {imbalanceRatio:F1}:1 ratio",
                    SuggestedFix = "Consider:\n" +
                                  "1. Collecting more data for minority class\n" +
                                  "2. Using data augmentation\n" +
                                  "3. Adjusting ML.NET training parameters for imbalanced data",
                    Metadata = new Dictionary<string, object>
                    {
                        ["ClassCounts"] = labelCounts,
                        ["ImbalanceRatio"] = imbalanceRatio
                    }
                });
            }
        }

        await Task.CompletedTask;
    }

    /// <summary>
    /// Checks if a dataset likely needs preprocessing.
    /// </summary>
    /// <param name="csvPath">Path to CSV file</param>
    /// <param name="labelColumn">Optional label column</param>
    /// <returns>True if preprocessing is recommended</returns>
    public async Task<bool> NeedsPreprocessingAsync(string csvPath, string? labelColumn = null)
    {
        var issues = await AnalyzeAsync(csvPath, labelColumn);

        // Recommend preprocessing if any High or Critical issues
        return issues.Any(i => i.Severity == IssueSeverity.Critical ||
                              i.Severity == IssueSeverity.High);
    }
}
