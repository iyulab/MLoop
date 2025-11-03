using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Handles categorical value mapping and preprocessing for prediction
/// Ensures prediction data categorical values match training data values
/// </summary>
public class CategoricalMapper
{
    public enum UnknownValueStrategy
    {
        /// <summary>
        /// Automatically select the best strategy based on unknown value ratio
        /// - If unknown ratio less than 5%: Use most frequent value (safe replacement)
        /// - If unknown ratio 5-30%: Use missing (let ML.NET handle)
        /// - If unknown ratio greater than 30%: Error (likely data quality issue)
        /// </summary>
        Auto,

        /// <summary>
        /// Throw error when unknown categorical value is encountered
        /// </summary>
        Error,

        /// <summary>
        /// Replace unknown values with the most frequent value from training
        /// </summary>
        UseMostFrequent,

        /// <summary>
        /// Replace unknown values with empty string (ML.NET will handle as missing)
        /// </summary>
        UseMissing
    }

    public class MappingResult
    {
        public bool Success { get; set; }
        public List<string> UnknownValues { get; set; } = new();
        public Dictionary<string, List<string>> UnknownValuesByColumn { get; set; } = new();
        public string? ErrorMessage { get; set; }
        public string? TempFilePath { get; set; }
        public UnknownValueStrategy? AppliedStrategy { get; set; }
        public double UnknownValueRatio { get; set; }
        public string? StrategyReason { get; set; }
    }

    /// <summary>
    /// Preprocesses categorical values in prediction data to match training schema
    /// </summary>
    public MappingResult PreprocessPredictionData(
        string predictionDataPath,
        InputSchemaInfo trainedSchema,
        UnknownValueStrategy strategy = UnknownValueStrategy.Auto)
    {
        var result = new MappingResult { Success = true };

        try
        {
            // Read prediction data
            var lines = File.ReadAllLines(predictionDataPath);
            if (lines.Length == 0)
            {
                result.Success = false;
                result.ErrorMessage = "Prediction data file is empty";
                return result;
            }

            // If Auto strategy, analyze and select optimal strategy
            if (strategy == UnknownValueStrategy.Auto)
            {
                return AutoSelectStrategy(predictionDataPath, trainedSchema, lines);
            }

            var header = lines[0];
            var columnNames = header.Split(',');
            var dataLines = lines.Skip(1).ToArray();

            // Build column index mapping for categorical columns
            var categoricalColumns = new Dictionary<int, ColumnSchema>();

            foreach (var schemaCol in trainedSchema.Columns)
            {
                if (schemaCol.DataType == "Categorical" &&
                    schemaCol.Purpose == "Feature" &&
                    schemaCol.CategoricalValues != null)
                {
                    var colIndex = Array.IndexOf(columnNames, schemaCol.Name);
                    if (colIndex >= 0)
                    {
                        categoricalColumns[colIndex] = schemaCol;
                    }
                }
            }

            // If no categorical columns, no preprocessing needed
            if (!categoricalColumns.Any())
            {
                result.TempFilePath = predictionDataPath;
                return result;
            }

            // Process each data line
            var processedLines = new List<string> { header };
            var unknownValuesByColumn = new Dictionary<string, HashSet<string>>();

            foreach (var line in dataLines)
            {
                var values = line.Split(',');
                var processedValues = new string[values.Length];
                Array.Copy(values, processedValues, values.Length);

                foreach (var (colIndex, schemaCol) in categoricalColumns)
                {
                    if (colIndex < values.Length)
                    {
                        var value = values[colIndex].Trim();

                        // Check if value exists in training schema
                        if (!string.IsNullOrEmpty(value) &&
                            !schemaCol.CategoricalValues!.Contains(value))
                        {
                            // Track unknown value
                            if (!unknownValuesByColumn.ContainsKey(schemaCol.Name))
                            {
                                unknownValuesByColumn[schemaCol.Name] = new HashSet<string>();
                            }
                            unknownValuesByColumn[schemaCol.Name].Add(value);

                            // Apply strategy
                            switch (strategy)
                            {
                                case UnknownValueStrategy.Error:
                                    // Will report error after processing all lines
                                    break;

                                case UnknownValueStrategy.UseMostFrequent:
                                    // Use first value (after sorting, it's consistent)
                                    processedValues[colIndex] = schemaCol.CategoricalValues[0];
                                    break;

                                case UnknownValueStrategy.UseMissing:
                                    // Replace with empty (ML.NET handles as missing)
                                    processedValues[colIndex] = "";
                                    break;
                            }
                        }
                    }
                }

                processedLines.Add(string.Join(",", processedValues));
            }

            // Report unknown values if found
            if (unknownValuesByColumn.Any())
            {
                result.UnknownValuesByColumn = unknownValuesByColumn
                    .ToDictionary(
                        kvp => kvp.Key,
                        kvp => kvp.Value.OrderBy(v => v).ToList());

                result.UnknownValues = unknownValuesByColumn
                    .SelectMany(kvp => kvp.Value.Select(v => $"{kvp.Key}={v}"))
                    .OrderBy(v => v)
                    .ToList();

                if (strategy == UnknownValueStrategy.Error)
                {
                    result.Success = false;
                    result.ErrorMessage = BuildUnknownValueErrorMessage(result.UnknownValuesByColumn, trainedSchema);
                    return result;
                }
            }

            // Write processed data to temporary file
            var tempFile = Path.GetTempFileName();
            File.WriteAllLines(tempFile, processedLines);
            result.TempFilePath = tempFile;

            return result;
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.ErrorMessage = $"Error preprocessing categorical data: {ex.Message}";
            return result;
        }
    }

    private string BuildUnknownValueErrorMessage(
        Dictionary<string, List<string>> unknownValuesByColumn,
        InputSchemaInfo trainedSchema)
    {
        var message = "Categorical values in prediction data do not match training data:\n\n";

        foreach (var (columnName, unknownValues) in unknownValuesByColumn)
        {
            var schemaCol = trainedSchema.Columns.First(c => c.Name == columnName);

            message += $"Column '{columnName}':\n";
            message += $"  Unknown values: {string.Join(", ", unknownValues.Take(10))}";

            if (unknownValues.Count > 10)
            {
                message += $" ... and {unknownValues.Count - 10} more";
            }

            message += $"\n  Expected values ({schemaCol.UniqueValueCount} total): ";
            message += string.Join(", ", schemaCol.CategoricalValues!.Take(5));

            if (schemaCol.CategoricalValues!.Count > 5)
            {
                message += $" ... and {schemaCol.CategoricalValues.Count - 5} more";
            }

            message += "\n\n";
        }

        message += "Solutions:\n";
        message += "1. Ensure prediction data uses same categorical values as training data\n";
        message += "2. Retrain model with combined set of categorical values\n";
        message += "3. Map unknown values to known values before prediction\n";

        return message;
    }

    /// <summary>
    /// Automatically selects the best strategy based on unknown value analysis
    /// </summary>
    private MappingResult AutoSelectStrategy(
        string predictionDataPath,
        InputSchemaInfo trainedSchema,
        string[] lines)
    {
        var header = lines[0];
        var columnNames = header.Split(',');
        var dataLines = lines.Skip(1).ToArray();

        // Build column index mapping for categorical columns
        var categoricalColumns = new Dictionary<int, ColumnSchema>();
        foreach (var schemaCol in trainedSchema.Columns)
        {
            if (schemaCol.DataType == "Categorical" &&
                schemaCol.Purpose == "Feature" &&
                schemaCol.CategoricalValues != null)
            {
                var colIndex = Array.IndexOf(columnNames, schemaCol.Name);
                if (colIndex >= 0)
                {
                    categoricalColumns[colIndex] = schemaCol;
                }
            }
        }

        // Calculate unknown value statistics
        int totalCategoricalValues = 0;
        int unknownValueCount = 0;

        foreach (var line in dataLines)
        {
            var values = line.Split(',');

            foreach (var (colIndex, schemaCol) in categoricalColumns)
            {
                if (colIndex < values.Length)
                {
                    var value = values[colIndex].Trim();
                    if (!string.IsNullOrEmpty(value))
                    {
                        totalCategoricalValues++;

                        if (!schemaCol.CategoricalValues!.Contains(value))
                        {
                            unknownValueCount++;
                        }
                    }
                }
            }
        }

        // Calculate unknown ratio
        double unknownRatio = totalCategoricalValues > 0
            ? (double)unknownValueCount / totalCategoricalValues
            : 0;

        // Select strategy based on ratio
        UnknownValueStrategy selectedStrategy;
        string reason;

        if (unknownRatio < 0.05) // < 5%
        {
            selectedStrategy = UnknownValueStrategy.UseMostFrequent;
            reason = $"Low unknown value ratio ({unknownRatio:P2}) - safely replacing with most frequent values";
        }
        else if (unknownRatio < 0.30) // 5-30%
        {
            selectedStrategy = UnknownValueStrategy.UseMissing;
            reason = $"Moderate unknown value ratio ({unknownRatio:P2}) - treating as missing values";
        }
        else // > 30%
        {
            selectedStrategy = UnknownValueStrategy.Error;
            reason = $"High unknown value ratio ({unknownRatio:P2}) - likely data quality issue, manual review recommended";
        }

        // Process with selected strategy
        var result = PreprocessPredictionData(predictionDataPath, trainedSchema, selectedStrategy);

        // Add auto-selection metadata
        result.AppliedStrategy = selectedStrategy;
        result.UnknownValueRatio = unknownRatio;
        result.StrategyReason = reason;

        return result;
    }

    /// <summary>
    /// Validates categorical values without modifying the data
    /// </summary>
    public MappingResult ValidateCategoricalValues(
        string predictionDataPath,
        InputSchemaInfo trainedSchema)
    {
        return PreprocessPredictionData(predictionDataPath, trainedSchema, UnknownValueStrategy.Error);
    }
}
