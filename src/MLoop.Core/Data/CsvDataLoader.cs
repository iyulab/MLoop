using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.Core.Contracts;
using MLoop.Core.Models;

namespace MLoop.Core.Data;

/// <summary>
/// CSV data loader implementation using ML.NET
/// </summary>
public class CsvDataLoader : IDataProvider
{
    private readonly MLContext _mlContext;

    public CsvDataLoader(MLContext mlContext)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
    }

    public IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Data file not found: {filePath}");
        }

        // Ensure UTF-8 BOM for ML.NET compatibility (ML.NET's InferColumns relies on BOM detection)
        string mlnetCompatiblePath = EnsureUtf8Bom(filePath);

        // Infer columns from the file
        var columnInference = _mlContext.Auto().InferColumns(
            mlnetCompatiblePath,
            labelColumnName: labelColumn,
            separatorChar: ',');

        // BUG-15: Fix InferColumns misdetecting multiclass label as Boolean.
        // When label column only has 0/1 in early rows, InferColumns infers Boolean,
        // but fails when encountering values like 2. Override Boolean label to String
        // so MapValueToKey can handle any discrete class values.
        // BUG-17: Skip this conversion for binary-classification — ML.NET binary
        // classification pipeline expects Boolean labels and will fail with String.
        var isBinaryTask = string.Equals(taskType, "binary-classification", StringComparison.OrdinalIgnoreCase)
                        || string.Equals(taskType, "BinaryClassification", StringComparison.OrdinalIgnoreCase);

        if (!isBinaryTask && !string.IsNullOrEmpty(labelColumn) && columnInference.TextLoaderOptions.Columns != null)
        {
            foreach (var col in columnInference.TextLoaderOptions.Columns)
            {
                if (col.Name != null &&
                    col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase) &&
                    col.DataKind == DataKind.Boolean)
                {
                    col.DataKind = DataKind.String;
                }
            }
        }

        // Auto-detect datetime columns and exclude them from features.
        // ML.NET treats datetime strings as text and applies FeaturizeText,
        // creating thousands of character n-gram features that are meaningless.
        ExcludeDateTimeColumns(columnInference, mlnetCompatiblePath, labelColumn);

        // Create text loader with inferred schema
        // Ensure RFC 4180 compliance: handle commas inside quoted fields
        columnInference.TextLoaderOptions.AllowQuoting = true;
        var loader = _mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        var dataView = loader.Load(mlnetCompatiblePath);

        // Note: Do NOT delete temp file here - ML.NET may lazily load data
        // Temp files will be cleaned up by OS temp directory cleanup

        // Validate label column if specified
        if (!string.IsNullOrEmpty(labelColumn))
        {
            if (!ValidateLabelColumn(dataView, labelColumn))
            {
                throw new InvalidOperationException(
                    $"Label column '{labelColumn}' not found in the data. " +
                    $"Available columns: {string.Join(", ", GetColumnNames(dataView))}");
            }
        }

        return dataView;
    }

    public bool ValidateLabelColumn(IDataView data, string labelColumn)
    {
        var schema = data.Schema;
        return schema.Any(col => col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase));
    }

    public DataSchema GetSchema(IDataView data)
    {
        var schema = data.Schema;
        var columns = new List<ColumnInfo>();

        foreach (var column in schema)
        {
            columns.Add(new ColumnInfo
            {
                Name = column.Name,
                Type = DataTypeHelper.GetFriendlyTypeName(column.Type),
                IsLabel = false
            });
        }

        // Get approximate row count
        var rowCount = GetRowCount(data);

        return new DataSchema
        {
            Columns = columns,
            RowCount = rowCount
        };
    }

    public (IDataView trainSet, IDataView testSet) SplitData(
        IDataView data,
        double testFraction = 0.2)
    {
        // If testFraction is 0, use all data for both train and test (AutoML will use cross-validation)
        if (testFraction <= 0)
        {
            return (data, data);
        }

        if (testFraction >= 1)
        {
            throw new ArgumentException(
                "Test fraction must be less than 1",
                nameof(testFraction));
        }

        var split = _mlContext.Data.TrainTestSplit(
            data,
            testFraction: testFraction,
            seed: 42); // Fixed seed for reproducibility

        return (split.TrainSet, split.TestSet);
    }

    private IEnumerable<string> GetColumnNames(IDataView data)
    {
        return data.Schema.Select(col => col.Name);
    }

    private int GetRowCount(IDataView data)
    {
        // Try to get row count efficiently
        if (data is IDataView view)
        {
            long? count = view.GetRowCount();
            if (count.HasValue)
            {
                return (int)count.Value;
            }
        }

        // Fallback: count manually (expensive)
        int rowCount = 0;
        using (var cursor = data.GetRowCursor(data.Schema))
        {
            while (cursor.MoveNext())
            {
                rowCount++;
            }
        }

        return rowCount;
    }

    /// <summary>
    /// Detects datetime-like columns and moves them to IgnoredColumnNames.
    /// Prevents ML.NET from applying FeaturizeText to datetime strings,
    /// which would create thousands of useless character n-gram features.
    /// </summary>
    private static void ExcludeDateTimeColumns(
        ColumnInferenceResults columnInference,
        string filePath,
        string? labelColumn)
    {
        var textColumns = columnInference.ColumnInformation.TextColumnNames;
        if (textColumns == null || textColumns.Count == 0) return;

        // Sample first few data rows to detect datetime values
        var dateTimeColumns = new List<string>();
        var sampled = SampleColumnValues(filePath, textColumns, maxRows: 10);

        foreach (var colName in textColumns.ToList())
        {
            // Skip label column
            if (!string.IsNullOrEmpty(labelColumn) &&
                colName.Equals(labelColumn, StringComparison.OrdinalIgnoreCase))
                continue;

            bool isDateTime = false;

            // Heuristic 1: Column name patterns
            var lowerName = colName.ToLowerInvariant();
            if (lowerName.Contains("datetime") || lowerName.Contains("timestamp") ||
                lowerName == "date" || lowerName == "time" ||
                lowerName.EndsWith("_date") || lowerName.EndsWith("_time") ||
                lowerName.StartsWith("date_") || lowerName.StartsWith("time_"))
            {
                isDateTime = true;
            }

            // Heuristic 2: Value-based detection (more reliable)
            if (!isDateTime && sampled.TryGetValue(colName, out var values) && values.Count > 0)
            {
                var parsedCount = values.Count(v =>
                    !string.IsNullOrWhiteSpace(v) &&
                    DateTime.TryParse(v, System.Globalization.CultureInfo.InvariantCulture,
                        System.Globalization.DateTimeStyles.None, out _));

                var nonEmptyCount = values.Count(v => !string.IsNullOrWhiteSpace(v));
                if (nonEmptyCount > 0 && parsedCount >= nonEmptyCount * 0.8)
                {
                    isDateTime = true;
                }
            }

            if (isDateTime)
            {
                dateTimeColumns.Add(colName);
            }
        }

        // Move detected datetime columns to ignored
        foreach (var col in dateTimeColumns)
        {
            textColumns.Remove(col);
            columnInference.ColumnInformation.IgnoredColumnNames.Add(col);
            Console.WriteLine($"[Info] DateTime column '{col}' excluded from features (use FilePrepper to extract date features if needed)");
        }
    }

    /// <summary>
    /// Samples values from specific columns by reading the CSV header + first N rows.
    /// </summary>
    private static Dictionary<string, List<string>> SampleColumnValues(
        string filePath, ICollection<string> columnNames, int maxRows)
    {
        var result = new Dictionary<string, List<string>>();
        foreach (var col in columnNames)
            result[col] = new List<string>();

        using var reader = new StreamReader(filePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
        var headerLine = reader.ReadLine();
        if (headerLine == null) return result;

        // Parse header to find column indices
        var headers = ParseCsvLine(headerLine);
        var colIndices = new Dictionary<string, int>();
        for (int i = 0; i < headers.Length; i++)
        {
            if (columnNames.Contains(headers[i]))
                colIndices[headers[i]] = i;
        }

        // Read sample rows
        int rowsRead = 0;
        string? line;
        while (rowsRead < maxRows && (line = reader.ReadLine()) != null)
        {
            var fields = ParseCsvLine(line);
            foreach (var (colName, idx) in colIndices)
            {
                if (idx < fields.Length)
                    result[colName].Add(fields[idx]);
            }
            rowsRead++;
        }

        return result;
    }

    /// <summary>
    /// Simple CSV line parser that handles quoted fields.
    /// </summary>
    private static string[] ParseCsvLine(string line)
    {
        var fields = new List<string>();
        bool inQuote = false;
        var current = new System.Text.StringBuilder();

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (c == '"')
            {
                if (inQuote && i + 1 < line.Length && line[i + 1] == '"')
                {
                    current.Append('"');
                    i++;
                }
                else
                {
                    inQuote = !inQuote;
                }
            }
            else if (c == ',' && !inQuote)
            {
                fields.Add(current.ToString().Trim());
                current.Clear();
            }
            else
            {
                current.Append(c);
            }
        }
        fields.Add(current.ToString().Trim());
        return fields.ToArray();
    }

    /// <summary>
    /// Ensures the CSV file has UTF-8 BOM for ML.NET compatibility.
    /// ML.NET's InferColumns doesn't have encoding parameters and relies on BOM detection.
    /// Detects encoding (UTF-8, CP949, EUC-KR) and converts to UTF-8 with BOM if needed.
    /// </summary>
    private static string EnsureUtf8Bom(string filePath)
    {
        var (convertedPath, detection) = EncodingDetector.ConvertToUtf8WithBom(filePath);

        if (detection.WasConverted && detection.EncodingName != "UTF-8")
        {
            Console.WriteLine($"[Info] Converted {detection.EncodingName} → UTF-8: {Path.GetFileName(filePath)}");
        }

        return convertedPath;
    }
}
