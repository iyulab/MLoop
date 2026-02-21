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

        // Flatten multi-line quoted headers (ML.NET doesn't support them)
        mlnetCompatiblePath = FlattenMultiLineHeaders(mlnetCompatiblePath);

        // Remove unnamed/index columns (e.g., pandas default index, "Unnamed: 0")
        mlnetCompatiblePath = RemoveIndexColumns(mlnetCompatiblePath);

        // Warn if CSV appears to have no header row
        WarnIfHeaderless(mlnetCompatiblePath);

        // Pre-InferColumns: Remove DateTime columns from CSV.
        // ML.NET treats datetime strings as text and applies FeaturizeText,
        // creating tens of thousands of character n-gram features.
        // Removing from CSV ensures InferColumns never sees them.
        mlnetCompatiblePath = RemoveDateTimeColumns(mlnetCompatiblePath, labelColumn);

        // Pre-InferColumns: Remove sparse columns (>90% missing) from CSV.
        // ML.NET may combine sparse columns into a "Features" vector, preventing
        // post-InferColumns detection. Pre-removing prevents OOM from FeaturizeText.
        mlnetCompatiblePath = RemoveSparseColumns(mlnetCompatiblePath, labelColumn);

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

        // For binary classification with String label, convert to Boolean.
        // ML.NET binary classification AutoML requires Boolean labels.
        // String labels like "OK"/"NG", "Pass"/"Fail" are common in manufacturing data.
        if (isBinaryTask && !string.IsNullOrEmpty(labelColumn))
        {
            var labelSchema = dataView.Schema.GetColumnOrNull(labelColumn);
            if (labelSchema.HasValue && labelSchema.Value.Type is TextDataViewType)
            {
                dataView = ConvertStringLabelToBoolean(dataView, labelColumn);
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
    /// <summary>
    /// Pre-InferColumns step: removes DateTime columns from the CSV file.
    /// Returns a new file path without the DateTime columns, or the original path if none detected.
    /// </summary>
    public static string RemoveDateTimeColumns(
        string filePath,
        string? labelColumn)
    {
        try
        {
            string? headerLine;
            string[] headers;
            using (var reader = new StreamReader(filePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                headerLine = reader.ReadLine();
                if (headerLine == null) return filePath;
                headers = ParseCsvLine(headerLine);
            }

            if (headers.Length <= 1) return filePath;

            // Sample values to detect DateTime columns
            const int sampleRows = 10;
            var columnValues = new Dictionary<string, List<string>>();
            foreach (var h in headers)
                columnValues[h] = new List<string>();

            using (var reader = new StreamReader(filePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                reader.ReadLine(); // skip header
                int count = 0;
                string? line;
                while ((line = reader.ReadLine()) != null && count < sampleRows)
                {
                    count++;
                    var fields = ParseCsvLine(line);
                    for (int i = 0; i < headers.Length && i < fields.Length; i++)
                    {
                        if (!string.IsNullOrWhiteSpace(fields[i]))
                            columnValues[headers[i]].Add(fields[i]);
                    }
                }
            }

            // Detect DateTime columns
            var dateTimeIndices = new HashSet<int>();
            for (int i = 0; i < headers.Length; i++)
            {
                // Don't remove label column
                if (!string.IsNullOrEmpty(labelColumn) &&
                    headers[i].Equals(labelColumn, StringComparison.OrdinalIgnoreCase))
                    continue;

                columnValues.TryGetValue(headers[i], out var values);
                if (DateTimeDetector.IsDateTimeColumn(headers[i], values))
                {
                    dateTimeIndices.Add(i);
                }
            }

            if (dateTimeIndices.Count == 0) return filePath;

            // Create new CSV without DateTime columns
            var tempPath = Path.Combine(Path.GetTempPath(), $"mloop_nodt_{Guid.NewGuid():N}{Path.GetExtension(filePath)}");
            var keepIndices = Enumerable.Range(0, headers.Length).Except(dateTimeIndices).ToArray();

            var allLines = File.ReadAllLines(filePath, System.Text.Encoding.UTF8);
            using (var writer = new StreamWriter(tempPath, false, new System.Text.UTF8Encoding(true)))
            {
                foreach (var line in allLines)
                {
                    var fields = ParseCsvLine(line);
                    var kept = keepIndices
                        .Where(idx => idx < fields.Length)
                        .Select(idx => fields[idx].Contains(',') || fields[idx].Contains('"')
                            ? $"\"{fields[idx].Replace("\"", "\"\"")}\""
                            : fields[idx]);
                    writer.WriteLine(string.Join(",", kept));
                }
            }

            foreach (var idx in dateTimeIndices)
            {
                Console.WriteLine($"[Info] DateTime column '{headers[idx]}' excluded from features (use FilePrepper to extract date features if needed)");
            }

            return tempPath;
        }
        catch
        {
            return filePath; // Non-critical, continue with original
        }
    }

    /// <summary>
    /// Pre-InferColumns step: removes columns with >threshold missing values from the CSV file.
    /// Returns a new file path without the sparse columns, or the original path if none are sparse.
    /// </summary>
    public static string RemoveSparseColumns(
        string filePath,
        string? labelColumn,
        double threshold = 0.90)
    {
        try
        {
            const int sampleRows = 200;

            string? headerLine;
            string[] headers;
            using (var reader = new StreamReader(filePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                headerLine = reader.ReadLine();
                if (headerLine == null) return filePath;
                headers = ParseCsvLine(headerLine);
            }

            if (headers.Length <= 1) return filePath;

            // Count missing values per column by sampling
            var missingCounts = new int[headers.Length];
            int totalRows = 0;

            using (var reader = new StreamReader(filePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                reader.ReadLine(); // skip header
                string? line;
                while ((line = reader.ReadLine()) != null && totalRows < sampleRows)
                {
                    totalRows++;
                    var fields = ParseCsvLine(line);
                    for (int i = 0; i < headers.Length; i++)
                    {
                        if (i >= fields.Length || string.IsNullOrWhiteSpace(fields[i]))
                        {
                            missingCounts[i]++;
                        }
                    }
                }
            }

            if (totalRows == 0) return filePath;

            // Identify sparse columns (skip label column)
            var sparseIndices = new HashSet<int>();
            for (int i = 0; i < headers.Length; i++)
            {
                double missingRate = (double)missingCounts[i] / totalRows;
                if (missingRate >= threshold)
                {
                    // Don't remove label column
                    if (!string.IsNullOrEmpty(labelColumn) &&
                        headers[i].Equals(labelColumn, StringComparison.OrdinalIgnoreCase))
                        continue;

                    sparseIndices.Add(i);
                }
            }

            if (sparseIndices.Count == 0) return filePath;

            // Create new CSV without sparse columns
            var tempPath = Path.Combine(Path.GetTempPath(), $"mloop_nosparse_{Guid.NewGuid():N}{Path.GetExtension(filePath)}");
            var keepIndices = Enumerable.Range(0, headers.Length).Except(sparseIndices).ToArray();

            var allLines = File.ReadAllLines(filePath, System.Text.Encoding.UTF8);
            using (var writer = new StreamWriter(tempPath, false, new System.Text.UTF8Encoding(true)))
            {
                foreach (var line in allLines)
                {
                    var fields = ParseCsvLine(line);
                    var kept = keepIndices
                        .Where(idx => idx < fields.Length)
                        .Select(idx => fields[idx].Contains(',') || fields[idx].Contains('"')
                            ? $"\"{fields[idx].Replace("\"", "\"\"")}\""
                            : fields[idx]);
                    writer.WriteLine(string.Join(",", kept));
                }
            }

            var removedNames = sparseIndices.Select(i => headers[i]);
            foreach (var name in removedNames)
            {
                Console.WriteLine($"[Warning] Sparse column '{name}' excluded (>{threshold:P0} missing values)");
            }

            return tempPath;
        }
        catch
        {
            return filePath; // Non-critical, continue with original
        }
    }

    /// <summary>
    /// Simple CSV line parser that handles quoted fields.
    /// </summary>
    /// <summary>
    /// Converts a String label column to Boolean for binary classification.
    /// Maps unique label values alphabetically: first → false, second → true.
    /// e.g., "NG"/"OK" → NG=false, OK=true; "Fail"/"Pass" → Fail=false, Pass=true
    /// </summary>
    private IDataView ConvertStringLabelToBoolean(IDataView dataView, string labelColumn)
    {
        // Collect unique label values (stop at 3 to detect non-binary)
        var uniqueValues = new HashSet<string>();
        var labelCol = dataView.Schema[labelColumn];

        using (var cursor = dataView.GetRowCursor(new[] { labelCol }))
        {
            var getter = cursor.GetGetter<ReadOnlyMemory<char>>(labelCol);
            while (cursor.MoveNext() && uniqueValues.Count <= 3)
            {
                ReadOnlyMemory<char> val = default;
                getter(ref val);
                var str = val.ToString().Trim();
                if (!string.IsNullOrEmpty(str))
                {
                    uniqueValues.Add(str);
                }
            }
        }

        if (uniqueValues.Count != 2)
        {
            return dataView; // Not binary, return as-is
        }

        // Sort alphabetically: first → negative (false), second → positive (true)
        var sorted = uniqueValues.OrderBy(v => v, StringComparer.OrdinalIgnoreCase).ToArray();
        Console.WriteLine($"[Info] Converting label: '{sorted[0]}' → False, '{sorted[1]}' → True");

        // Build lookup IDataView for MapValue transform
        var lookupData = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new LabelMapping { Key = sorted[0], Value = false },
            new LabelMapping { Key = sorted[1], Value = true }
        });

        var pipeline = _mlContext.Transforms.Conversion.MapValue(
            labelColumn,
            lookupData,
            lookupData.Schema["Key"],
            lookupData.Schema["Value"],
            labelColumn);

        return pipeline.Fit(dataView).Transform(dataView);
    }

    private sealed class LabelMapping
    {
        public string Key { get; set; } = "";
        public bool Value { get; set; }
    }

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

    /// <summary>
    /// Detects and removes unnamed/index columns from CSV files.
    /// Common patterns: empty column name (pandas default index), "Unnamed: 0", "Unnamed: N".
    /// These columns are auto-generated row numbers that should not be used as features.
    /// Returns original path if no index columns found, or a temp file path otherwise.
    /// </summary>

    /// <summary>
    /// Warns if the CSV file appears to have no header row (first row looks like data).
    /// Detection heuristic: all fields in the first row are numeric.
    /// </summary>
    private static void WarnIfHeaderless(string filePath)
    {
        if (IsLikelyHeaderless(filePath))
        {
            Console.WriteLine("[Warning] Possible headerless CSV detected: first row appears to be data (all numeric values).");
            Console.WriteLine("[Warning] ML.NET will treat the first row as column names, which may cause incorrect results.");
            Console.WriteLine("[Info] Solution: Add a header row with column names (e.g., Feature1,Feature2,...,Label).");
        }
    }

    /// <summary>
    /// Determines if a CSV file likely has no header row.
    /// Returns true if all fields in the first row are numeric (int/float/double).
    /// </summary>
    public static bool IsLikelyHeaderless(string filePath)
    {
        try
        {
            string? firstLine;
            string? secondLine;
            using (var reader = new StreamReader(filePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine();
                secondLine = reader.ReadLine();
            }

            if (string.IsNullOrEmpty(firstLine)) return false;

            var fields = ParseCsvLine(firstLine);
            if (fields.Length < 2) return false; // Too few fields to judge

            // Check if ALL fields in the first row are numeric
            var allNumeric = fields.All(f =>
                !string.IsNullOrWhiteSpace(f) && double.TryParse(f.Trim(), System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.InvariantCulture, out _));

            if (!allNumeric) return false;

            // Additional check: if second row exists and has same pattern, more confident
            if (string.IsNullOrEmpty(secondLine)) return true; // Only one row, but all numeric = suspicious

            var secondFields = ParseCsvLine(secondLine);

            // If both rows have same field count and both all-numeric, very likely headerless
            return secondFields.Length == fields.Length;
        }
        catch
        {
            return false;
        }
    }

    public static string RemoveIndexColumns(string filePath)
    {
        try
        {
            string? firstLine;
            using (var reader = new StreamReader(filePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine();
            }

            if (string.IsNullOrEmpty(firstLine)) return filePath;

            var headers = ParseCsvLine(firstLine);
            var indexColumns = new List<int>();

            for (int i = 0; i < headers.Length; i++)
            {
                if (IsLikelyIndexColumn(headers[i]))
                {
                    indexColumns.Add(i);
                }
            }

            if (indexColumns.Count == 0) return filePath;

            // Create temp file without the index columns
            var tempPath = Path.Combine(Path.GetTempPath(), $"mloop_noidx_{Guid.NewGuid():N}{Path.GetExtension(filePath)}");
            var keepIndices = Enumerable.Range(0, headers.Length).Except(indexColumns).ToArray();

            var allLines = File.ReadAllLines(filePath, System.Text.Encoding.UTF8);
            using (var writer = new StreamWriter(tempPath, false, new System.Text.UTF8Encoding(true)))
            {
                foreach (var line in allLines)
                {
                    var fields = ParseCsvLine(line);
                    var kept = keepIndices
                        .Where(idx => idx < fields.Length)
                        .Select(idx => fields[idx].Contains(',') || fields[idx].Contains('"')
                            ? $"\"{fields[idx].Replace("\"", "\"\"")}\""
                            : fields[idx]);
                    writer.WriteLine(string.Join(",", kept));
                }
            }

            var removedNames = indexColumns.Select(i => string.IsNullOrWhiteSpace(headers[i]) ? "(empty)" : headers[i]);
            Console.WriteLine($"[Info] Removed index column(s): {string.Join(", ", removedNames)}");
            return tempPath;
        }
        catch
        {
            return filePath; // Non-critical, continue with original
        }
    }

    /// <summary>
    /// Determines if a column name is likely an auto-generated index column.
    /// Matches: empty/whitespace names, "Unnamed: N" (pandas), "Unnamed".
    /// Does NOT match common feature names like "id", "index" as they may be intentional.
    /// </summary>
    public static bool IsLikelyIndexColumn(string columnName)
    {
        // Pattern 1: Empty or whitespace-only name (pandas df.to_csv() with index=True default)
        if (string.IsNullOrWhiteSpace(columnName))
            return true;

        // Pattern 2: Pandas "Unnamed: 0", "Unnamed: 1", etc.
        if (columnName.StartsWith("Unnamed:", StringComparison.OrdinalIgnoreCase) ||
            columnName.Equals("Unnamed", StringComparison.OrdinalIgnoreCase))
            return true;

        return false;
    }

    /// <summary>
    /// Detects and normalizes multi-line quoted headers in CSV files.
    /// ML.NET's TextLoader and InferColumns do not support multi-line quoted headers.
    /// If newlines are found within quoted header fields, they are replaced with spaces.
    /// Returns the original path if no multi-line header is detected, or a temp file path otherwise.
    /// </summary>
    public static string FlattenMultiLineHeaders(string filePath)
    {
        // Quick check: read first line and see if quotes are unbalanced
        string? firstLine;
        using (var reader = new StreamReader(filePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
        {
            firstLine = reader.ReadLine();
        }

        if (string.IsNullOrEmpty(firstLine)) return filePath;

        // Count quotes - if even number, header is single-line (no unterminated quoted field)
        int quoteCount = firstLine.Count(c => c == '"');
        if (quoteCount % 2 == 0)
        {
            // BUG-R2-07: Detect possible multi-row header pattern
            // Row1 has many duplicate values = likely category headers, not real column names
            DetectMultiRowHeaderPattern(filePath, firstLine);
            return filePath;
        }

        // Multi-line header detected - need to flatten
        var tempPath = Path.Combine(Path.GetTempPath(), $"mloop_flat_{Guid.NewGuid():N}{Path.GetExtension(filePath)}");

        using (var reader = new StreamReader(filePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
        using (var writer = new StreamWriter(tempPath, false, new System.Text.UTF8Encoding(true)))
        {
            // Read and flatten the header (may span multiple physical lines)
            var headerBuilder = new System.Text.StringBuilder();
            bool inQuote = false;

            while (true)
            {
                var line = reader.ReadLine();
                if (line == null) break;

                if (headerBuilder.Length > 0) headerBuilder.Append(' ');
                headerBuilder.Append(line);

                foreach (char c in line)
                {
                    if (c == '"') inQuote = !inQuote;
                }

                if (!inQuote) break;
            }

            writer.WriteLine(headerBuilder.ToString());

            // Copy remaining data lines as-is
            string? dataLine;
            while ((dataLine = reader.ReadLine()) != null)
            {
                writer.WriteLine(dataLine);
            }
        }

        Console.WriteLine($"[Info] Flattened multi-line CSV headers: {Path.GetFileName(filePath)}");
        return tempPath;
    }

    /// <summary>
    /// Detects multi-row header pattern where Row 1 contains category names
    /// and Row 2 contains actual column names (common in Excel exports).
    /// Warns the user if detected.
    /// </summary>
    private static void DetectMultiRowHeaderPattern(string filePath, string firstLine)
    {
        try
        {
            var row1Fields = ParseCsvLine(firstLine);
            if (row1Fields.Length < 3) return;

            var uniqueRow1 = new HashSet<string>(row1Fields, StringComparer.OrdinalIgnoreCase);

            // If Row1 has very few unique values relative to total columns,
            // it's likely a category header row (e.g., "Process, Process, Sensor, Sensor, Defects")
            var uniqueRatio = (double)uniqueRow1.Count / row1Fields.Length;
            if (uniqueRatio >= 0.5) return; // More than half unique — probably a real header

            // Read Row 2 to compare
            using var reader = new StreamReader(filePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            reader.ReadLine(); // skip Row 1
            var secondLine = reader.ReadLine();
            if (string.IsNullOrEmpty(secondLine)) return;

            var row2Fields = ParseCsvLine(secondLine);
            var uniqueRow2 = new HashSet<string>(row2Fields, StringComparer.OrdinalIgnoreCase);

            // If Row2 has significantly more unique values than Row1, confirm multi-row header
            if (uniqueRow2.Count > uniqueRow1.Count * 2)
            {
                Console.WriteLine($"[Warning] Possible multi-row header detected: Row 1 has only {uniqueRow1.Count} unique values across {row1Fields.Length} columns.");
                Console.WriteLine($"[Warning] Row 1 may be category headers (e.g., '{string.Join("', '", uniqueRow1.Take(3))}').");
                Console.WriteLine($"[Warning] If columns appear incorrect, preprocess the CSV to use Row 2 as the header.");
            }
        }
        catch
        {
            // Non-critical detection — ignore errors
        }
    }
}
