using MLoop.AIAgent.Core.Models;
using System.Text;

namespace MLoop.AIAgent.Core;

/// <summary>
/// Generates C# preprocessing scripts based on data analysis
/// </summary>
public class PreprocessingScriptGenerator
{
    /// <summary>
    /// Generate preprocessing scripts from analysis report
    /// </summary>
    public PreprocessingScriptGenerationResult GenerateScripts(
        DataAnalysisReport analysisReport,
        string? outputDirectory = null)
    {
        var scripts = new List<PreprocessingScriptInfo>();
        int sequence = 1;

        // 1. Remove duplicate rows
        if (analysisReport.QualityIssues.DuplicateRowCount > 0)
        {
            scripts.Add(GenerateRemoveDuplicatesScript(sequence++, analysisReport));
        }

        // 2. Remove constant columns
        if (analysisReport.QualityIssues.ConstantColumns.Count > 0)
        {
            scripts.Add(GenerateRemoveConstantColumnsScript(
                sequence++,
                analysisReport.QualityIssues.ConstantColumns));
        }

        // 3. Handle missing values
        if (analysisReport.QualityIssues.ColumnsWithMissingValues.Count > 0)
        {
            scripts.Add(GenerateHandleMissingValuesScript(
                sequence++,
                analysisReport));
        }

        // 4. Handle outliers (if recommended)
        if (analysisReport.QualityIssues.ColumnsWithOutliers.Count > 0)
        {
            scripts.Add(GenerateHandleOutliersScript(
                sequence++,
                analysisReport.QualityIssues.ColumnsWithOutliers));
        }

        // 5. Encode categorical features (exclude target)
        var categoricalFeatures = analysisReport.Columns
            .Where(c => (c.InferredType == DataType.Categorical ||
                        c.InferredType == DataType.Boolean) &&
                       c.Name != analysisReport.RecommendedTarget?.ColumnName)
            .ToList();

        if (categoricalFeatures.Count > 0)
        {
            scripts.Add(GenerateEncodeCategoricalScript(
                sequence++,
                categoricalFeatures,
                analysisReport.QualityIssues.HighCardinalityColumns));
        }

        // 6. Normalize numeric features (exclude target if regression)
        var numericFeatures = analysisReport.Columns
            .Where(c => c.InferredType == DataType.Numeric &&
                       c.Name != analysisReport.RecommendedTarget?.ColumnName)
            .ToList();

        if (numericFeatures.Count > 0)
        {
            scripts.Add(GenerateNormalizeNumericScript(sequence++, numericFeatures));
        }

        var summary = GenerateSummary(scripts, analysisReport);

        return new PreprocessingScriptGenerationResult
        {
            Scripts = scripts,
            OutputDirectory = outputDirectory ?? ".",
            Summary = summary
        };
    }

    private PreprocessingScriptInfo GenerateRemoveDuplicatesScript(
        int sequence,
        DataAnalysisReport report)
    {
        var code = $$"""
using MLoop.Extensibility;

namespace MLoop.Preprocessing;

/// <summary>
/// Removes duplicate rows from the dataset
/// Generated: {{DateTime.Now:yyyy-MM-dd HH:mm:ss}}
/// </summary>
public class RemoveDuplicatesScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        context.Logger.Information("Removing duplicate rows...");

        var data = await context.Csv.ReadAsync(context.InputPath);
        var originalCount = data.Count;

        // Remove duplicates based on all columns
        var uniqueData = data
            .GroupBy(row => string.Join("|", row.Values))
            .Select(g => g.First())
            .ToList();

        var removedCount = originalCount - uniqueData.Count;
        context.Logger.Information($"Removed {removedCount} duplicate rows ({removedCount * 100.0 / originalCount:F1}%)");

        var outputPath = Path.Combine(context.OutputDirectory, "{{sequence:D2}}_remove_duplicates.csv");
        await context.Csv.WriteAsync(outputPath, uniqueData);

        return outputPath;
    }
}
""";

        return new PreprocessingScriptInfo
        {
            Sequence = sequence,
            Name = "remove_duplicates",
            FileName = $"{sequence:D2}_remove_duplicates.cs",
            Description = $"Remove {report.QualityIssues.DuplicateRowCount} duplicate rows",
            SourceCode = code
        };
    }

    private PreprocessingScriptInfo GenerateRemoveConstantColumnsScript(
        int sequence,
        List<string> constantColumns)
    {
        var columnsArray = string.Join("\", \"", constantColumns);

        var code = $$"""
using MLoop.Extensibility;

namespace MLoop.Preprocessing;

/// <summary>
/// Removes constant-value columns that provide no information
/// Generated: {{DateTime.Now:yyyy-MM-dd HH:mm:ss}}
/// </summary>
public class RemoveConstantColumnsScript : IPreprocessingScript
{
    private static readonly string[] ConstantColumns = { "{{columnsArray}}" };

    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        context.Logger.Information($"Removing {ConstantColumns.Length} constant columns...");

        var data = await context.Csv.ReadAsync(context.InputPath);

        // Remove constant columns
        foreach (var row in data)
        {
            foreach (var column in ConstantColumns)
            {
                row.Remove(column);
            }
        }

        context.Logger.Information($"Removed columns: {string.Join(", ", ConstantColumns)}");

        var outputPath = Path.Combine(context.OutputDirectory, "{{sequence:D2}}_remove_constant.csv");
        await context.Csv.WriteAsync(outputPath, data);

        return outputPath;
    }
}
""";

        return new PreprocessingScriptInfo
        {
            Sequence = sequence,
            Name = "remove_constant",
            FileName = $"{sequence:D2}_remove_constant.cs",
            Description = $"Remove {constantColumns.Count} constant columns: {string.Join(", ", constantColumns)}",
            SourceCode = code
        };
    }

    private PreprocessingScriptInfo GenerateHandleMissingValuesScript(
        int sequence,
        DataAnalysisReport report)
    {
        var numericColumns = report.Columns
            .Where(c => c.InferredType == DataType.Numeric &&
                       report.QualityIssues.ColumnsWithMissingValues.Contains(c.Name))
            .Select(c => c.Name)
            .ToList();

        var categoricalColumns = report.Columns
            .Where(c => (c.InferredType == DataType.Categorical ||
                        c.InferredType == DataType.Boolean) &&
                       report.QualityIssues.ColumnsWithMissingValues.Contains(c.Name))
            .Select(c => c.Name)
            .ToList();

        var numericArray = numericColumns.Count > 0
            ? $"{{ \"{string.Join("\", \"", numericColumns)}\" }}"
            : "Array.Empty<string>()";

        var categoricalArray = categoricalColumns.Count > 0
            ? $"{{ \"{string.Join("\", \"", categoricalColumns)}\" }}"
            : "Array.Empty<string>()";

        var code = $$"""
using MLoop.Extensibility;

namespace MLoop.Preprocessing;

/// <summary>
/// Handles missing values using appropriate strategies for each data type
/// Generated: {{DateTime.Now:yyyy-MM-dd HH:mm:ss}}
/// </summary>
public class HandleMissingValuesScript : IPreprocessingScript
{
    private static readonly string[] NumericColumns = {{numericArray}};
    private static readonly string[] CategoricalColumns = {{categoricalArray}};

    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        context.Logger.Information("Handling missing values...");

        var data = await context.Csv.ReadAsync(context.InputPath);
        int imputedCount = 0;

        // Calculate medians for numeric columns
        var medians = CalculateMedians(data, NumericColumns);

        foreach (var row in data)
        {
            // Impute numeric columns with median
            foreach (var column in NumericColumns)
            {
                if (row.TryGetValue(column, out var value) && string.IsNullOrWhiteSpace(value))
                {
                    row[column] = medians[column];
                    imputedCount++;
                }
            }

            // Impute categorical columns with mode (most frequent)
            foreach (var column in CategoricalColumns)
            {
                if (row.TryGetValue(column, out var value) && string.IsNullOrWhiteSpace(value))
                {
                    row[column] = "Unknown";
                    imputedCount++;
                }
            }
        }

        context.Logger.Information($"Imputed {imputedCount} missing values");

        var outputPath = Path.Combine(context.OutputDirectory, "{{sequence:D2}}_handle_missing.csv");
        await context.Csv.WriteAsync(outputPath, data);

        return outputPath;
    }

    private Dictionary<string, string> CalculateMedians(
        List<Dictionary<string, string>> data,
        string[] columns)
    {
        var medians = new Dictionary<string, string>();

        foreach (var column in columns)
        {
            var values = data
                .Select(row => row.TryGetValue(column, out var val) ? val : "")
                .Where(v => !string.IsNullOrWhiteSpace(v))
                .Select(v => double.TryParse(v, out var d) ? d : (double?)null)
                .Where(v => v.HasValue)
                .Select(v => v!.Value)
                .OrderBy(v => v)
                .ToList();

            if (values.Count > 0)
            {
                var median = values.Count % 2 == 0
                    ? (values[values.Count / 2 - 1] + values[values.Count / 2]) / 2
                    : values[values.Count / 2];
                medians[column] = median.ToString("G17");
            }
        }

        return medians;
    }
}
""";

        return new PreprocessingScriptInfo
        {
            Sequence = sequence,
            Name = "handle_missing",
            FileName = $"{sequence:D2}_handle_missing.cs",
            Description = $"Impute {report.QualityIssues.ColumnsWithMissingValues.Count} columns with missing values",
            SourceCode = code
        };
    }

    private PreprocessingScriptInfo GenerateHandleOutliersScript(
        int sequence,
        List<string> outlierColumns)
    {
        var columnsArray = string.Join("\", \"", outlierColumns);

        var code = $$"""
using MLoop.Extensibility;

namespace MLoop.Preprocessing;

/// <summary>
/// Handles outliers using IQR-based capping strategy
/// Generated: {{DateTime.Now:yyyy-MM-dd HH:mm:ss}}
/// </summary>
public class HandleOutliersScript : IPreprocessingScript
{
    private static readonly string[] OutlierColumns = { "{{columnsArray}}" };

    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        context.Logger.Information("Handling outliers using IQR capping...");

        var data = await context.Csv.ReadAsync(context.InputPath);
        int cappedCount = 0;

        // Calculate IQR bounds for each column
        var bounds = CalculateIQRBounds(data, OutlierColumns);

        foreach (var row in data)
        {
            foreach (var column in OutlierColumns)
            {
                if (row.TryGetValue(column, out var value) &&
                    double.TryParse(value, out var numValue))
                {
                    var (lower, upper) = bounds[column];

                    if (numValue < lower)
                    {
                        row[column] = lower.ToString("G17");
                        cappedCount++;
                    }
                    else if (numValue > upper)
                    {
                        row[column] = upper.ToString("G17");
                        cappedCount++;
                    }
                }
            }
        }

        context.Logger.Information($"Capped {cappedCount} outlier values");

        var outputPath = Path.Combine(context.OutputDirectory, "{{sequence:D2}}_handle_outliers.csv");
        await context.Csv.WriteAsync(outputPath, data);

        return outputPath;
    }

    private Dictionary<string, (double Lower, double Upper)> CalculateIQRBounds(
        List<Dictionary<string, string>> data,
        string[] columns)
    {
        var bounds = new Dictionary<string, (double, double)>();

        foreach (var column in columns)
        {
            var values = data
                .Select(row => row.TryGetValue(column, out var val) ? val : "")
                .Where(v => !string.IsNullOrWhiteSpace(v))
                .Select(v => double.TryParse(v, out var d) ? d : (double?)null)
                .Where(v => v.HasValue)
                .Select(v => v!.Value)
                .OrderBy(v => v)
                .ToList();

            if (values.Count > 0)
            {
                var q1 = values[(int)(values.Count * 0.25)];
                var q3 = values[(int)(values.Count * 0.75)];
                var iqr = q3 - q1;
                var lower = q1 - 1.5 * iqr;
                var upper = q3 + 1.5 * iqr;
                bounds[column] = (lower, upper);
            }
        }

        return bounds;
    }
}
""";

        return new PreprocessingScriptInfo
        {
            Sequence = sequence,
            Name = "handle_outliers",
            FileName = $"{sequence:D2}_handle_outliers.cs",
            Description = $"Cap outliers in {outlierColumns.Count} numeric columns using IQR method",
            SourceCode = code
        };
    }

    private PreprocessingScriptInfo GenerateEncodeCategoricalScript(
        int sequence,
        List<ColumnAnalysis> categoricalColumns,
        List<string> highCardinalityColumns)
    {
        var oneHotColumns = categoricalColumns
            .Where(c => !highCardinalityColumns.Contains(c.Name) && c.UniqueCount <= 10)
            .Select(c => c.Name)
            .ToList();

        var labelEncodeColumns = categoricalColumns
            .Where(c => highCardinalityColumns.Contains(c.Name) || c.UniqueCount > 10)
            .Select(c => c.Name)
            .ToList();

        var oneHotArray = oneHotColumns.Count > 0
            ? $"{{ \"{string.Join("\", \"", oneHotColumns)}\" }}"
            : "Array.Empty<string>()";

        var labelEncodeArray = labelEncodeColumns.Count > 0
            ? $"{{ \"{string.Join("\", \"", labelEncodeColumns)}\" }}"
            : "Array.Empty<string>()";

        var code = $$"""
using MLoop.Extensibility;

namespace MLoop.Preprocessing;

/// <summary>
/// Encodes categorical features using one-hot or label encoding
/// Generated: {{DateTime.Now:yyyy-MM-dd HH:mm:ss}}
/// </summary>
public class EncodeCategoricalScript : IPreprocessingScript
{
    private static readonly string[] OneHotColumns = {{oneHotArray}};
    private static readonly string[] LabelEncodeColumns = {{labelEncodeArray}};

    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        context.Logger.Information("Encoding categorical features...");

        var data = await context.Csv.ReadAsync(context.InputPath);

        // One-hot encoding for low-cardinality columns
        if (OneHotColumns.Length > 0)
        {
            data = OneHotEncode(data, OneHotColumns, context);
        }

        // Label encoding for high-cardinality columns
        if (LabelEncodeColumns.Length > 0)
        {
            data = LabelEncode(data, LabelEncodeColumns, context);
        }

        var outputPath = Path.Combine(context.OutputDirectory, "{{sequence:D2}}_encode_categorical.csv");
        await context.Csv.WriteAsync(outputPath, data);

        return outputPath;
    }

    private List<Dictionary<string, string>> OneHotEncode(
        List<Dictionary<string, string>> data,
        string[] columns,
        PreprocessContext context)
    {
        foreach (var column in columns)
        {
            // Get unique values
            var uniqueValues = data
                .Select(row => row.TryGetValue(column, out var val) ? val : "")
                .Distinct()
                .Where(v => !string.IsNullOrWhiteSpace(v))
                .OrderBy(v => v)
                .ToList();

            context.Logger.Information($"One-hot encoding '{column}' ({uniqueValues.Count} categories)");

            // Create binary columns
            foreach (var row in data)
            {
                var value = row.TryGetValue(column, out var val) ? val : "";

                foreach (var uniqueValue in uniqueValues)
                {
                    var newColumn = $"{column}_{uniqueValue}";
                    row[newColumn] = value == uniqueValue ? "1" : "0";
                }

                // Remove original column
                row.Remove(column);
            }
        }

        return data;
    }

    private List<Dictionary<string, string>> LabelEncode(
        List<Dictionary<string, string>> data,
        string[] columns,
        PreprocessContext context)
    {
        foreach (var column in columns)
        {
            // Get unique values
            var uniqueValues = data
                .Select(row => row.TryGetValue(column, out var val) ? val : "")
                .Distinct()
                .Where(v => !string.IsNullOrWhiteSpace(v))
                .OrderBy(v => v)
                .ToList();

            var labelMap = uniqueValues
                .Select((value, index) => (value, index))
                .ToDictionary(x => x.value, x => x.index);

            context.Logger.Information($"Label encoding '{column}' ({uniqueValues.Count} categories)");

            // Apply label encoding
            foreach (var row in data)
            {
                if (row.TryGetValue(column, out var value) && labelMap.ContainsKey(value))
                {
                    row[column] = labelMap[value].ToString();
                }
            }
        }

        return data;
    }
}
""";

        return new PreprocessingScriptInfo
        {
            Sequence = sequence,
            Name = "encode_categorical",
            FileName = $"{sequence:D2}_encode_categorical.cs",
            Description = $"Encode {categoricalColumns.Count} categorical columns " +
                         $"(one-hot: {oneHotColumns.Count}, label: {labelEncodeColumns.Count})",
            SourceCode = code
        };
    }

    private PreprocessingScriptInfo GenerateNormalizeNumericScript(
        int sequence,
        List<ColumnAnalysis> numericColumns)
    {
        var columnsArray = string.Join("\", \"", numericColumns.Select(c => c.Name));

        var code = $$"""
using MLoop.Extensibility;

namespace MLoop.Preprocessing;

/// <summary>
/// Normalizes numeric features using min-max scaling to [0, 1] range
/// Generated: {{DateTime.Now:yyyy-MM-dd HH:mm:ss}}
/// </summary>
public class NormalizeNumericScript : IPreprocessingScript
{
    private static readonly string[] NumericColumns = { "{{columnsArray}}" };

    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        context.Logger.Information("Normalizing numeric features...");

        var data = await context.Csv.ReadAsync(context.InputPath);

        // Calculate min/max for each column
        var ranges = CalculateRanges(data, NumericColumns);

        int normalizedCount = 0;
        foreach (var row in data)
        {
            foreach (var column in NumericColumns)
            {
                if (row.TryGetValue(column, out var value) &&
                    double.TryParse(value, out var numValue))
                {
                    var (min, max) = ranges[column];
                    var range = max - min;

                    if (range > 0)
                    {
                        var normalized = (numValue - min) / range;
                        row[column] = normalized.ToString("G17");
                        normalizedCount++;
                    }
                }
            }
        }

        context.Logger.Information($"Normalized {normalizedCount} values across {NumericColumns.Length} columns");

        var outputPath = Path.Combine(context.OutputDirectory, "{{sequence:D2}}_normalize_numeric.csv");
        await context.Csv.WriteAsync(outputPath, data);

        return outputPath;
    }

    private Dictionary<string, (double Min, double Max)> CalculateRanges(
        List<Dictionary<string, string>> data,
        string[] columns)
    {
        var ranges = new Dictionary<string, (double, double)>();

        foreach (var column in columns)
        {
            var values = data
                .Select(row => row.TryGetValue(column, out var val) ? val : "")
                .Where(v => !string.IsNullOrWhiteSpace(v))
                .Select(v => double.TryParse(v, out var d) ? d : (double?)null)
                .Where(v => v.HasValue)
                .Select(v => v!.Value)
                .ToList();

            if (values.Count > 0)
            {
                ranges[column] = (values.Min(), values.Max());
            }
        }

        return ranges;
    }
}
""";

        return new PreprocessingScriptInfo
        {
            Sequence = sequence,
            Name = "normalize_numeric",
            FileName = $"{sequence:D2}_normalize_numeric.cs",
            Description = $"Normalize {numericColumns.Count} numeric columns using min-max scaling",
            SourceCode = code
        };
    }

    private string GenerateSummary(
        List<PreprocessingScriptInfo> scripts,
        DataAnalysisReport report)
    {
        var summary = new StringBuilder();
        summary.AppendLine($"Generated {scripts.Count} preprocessing scripts:");
        summary.AppendLine();

        foreach (var script in scripts)
        {
            summary.AppendLine($"{script.Sequence}. {script.FileName}");
            summary.AppendLine($"   {script.Description}");
        }

        summary.AppendLine();
        summary.AppendLine($"Target column: {report.RecommendedTarget?.ColumnName ?? "Not identified"}");
        summary.AppendLine($"Problem type: {report.RecommendedTarget?.ProblemType}");

        return summary.ToString();
    }

    /// <summary>
    /// Save generated scripts to disk
    /// </summary>
    public async Task SaveScriptsAsync(
        PreprocessingScriptGenerationResult result,
        string outputDirectory)
    {
        Directory.CreateDirectory(outputDirectory);

        foreach (var script in result.Scripts)
        {
            var filePath = Path.Combine(outputDirectory, script.FileName);
            await File.WriteAllTextAsync(filePath, script.SourceCode);
        }
    }
}
