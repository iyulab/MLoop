using Microsoft.ML;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Validates data quality before training to catch common issues
/// </summary>
public class DataQualityValidator
{
    private readonly MLContext _mlContext;

    public DataQualityValidator(MLContext mlContext)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
    }

    /// <summary>
    /// Validates data quality before training (label column + dataset size)
    /// </summary>
    public DataQualityResult ValidateTrainingData(string dataFile, string labelColumn)
    {
        // First validate label column
        var result = ValidateLabelColumn(dataFile, labelColumn);
        if (!result.IsValid)
        {
            return result; // Critical error, don't check dataset size
        }

        // Then check dataset size
        CheckDatasetSize(dataFile, result);

        return result;
    }

    /// <summary>
    /// Validates label column before training
    /// </summary>
    private DataQualityResult ValidateLabelColumn(string dataFile, string labelColumn)
    {
        var result = new DataQualityResult { IsValid = true };

        try
        {
            // Read the CSV file with UTF-8 encoding
            string? firstLine;
            using (var reader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine();
            }

            if (string.IsNullOrEmpty(firstLine))
            {
                result.IsValid = false;
                result.ErrorMessage = "Data file is empty";
                return result;
            }

            var columnNames = firstLine.Split(',');
            var labelColumnIndex = Array.IndexOf(columnNames, labelColumn);

            if (labelColumnIndex == -1)
            {
                result.IsValid = false;
                result.ErrorMessage = $"Label column '{labelColumn}' not found in data";
                return result;
            }

            // Read all data lines with UTF-8 encoding
            var allLines = File.ReadAllLines(dataFile, System.Text.Encoding.UTF8);
            var dataLines = allLines.Skip(1).ToArray(); // Skip header

            if (dataLines.Length == 0)
            {
                result.IsValid = false;
                result.ErrorMessage = "No data rows found";
                return result;
            }

            // Extract label column values
            var labelValues = new List<double>();
            var parseErrors = 0;

            foreach (var line in dataLines)
            {
                var values = line.Split(',');
                if (labelColumnIndex < values.Length)
                {
                    if (double.TryParse(values[labelColumnIndex], out var value))
                    {
                        labelValues.Add(value);
                    }
                    else
                    {
                        parseErrors++;
                    }
                }
            }

            if (parseErrors > 0)
            {
                result.Warnings.Add($"⚠ {parseErrors} values in label column '{labelColumn}' could not be parsed as numbers");
            }

            if (labelValues.Count == 0)
            {
                result.IsValid = false;
                result.ErrorMessage = $"No valid numeric values found in label column '{labelColumn}'";
                return result;
            }

            // Check 1: All same value (no variance)
            var uniqueValues = labelValues.Distinct().ToList();
            if (uniqueValues.Count == 1)
            {
                result.IsValid = false;
                result.ErrorMessage = $"Label column '{labelColumn}' contains all identical values ({uniqueValues[0]})";
                result.ErrorMessageEn = $"Cannot train model with constant label - this indicates a data quality issue";

                // Check specifically for all zeros
                if (uniqueValues[0] == 0.0)
                {
                    result.Suggestions.Add("❌ All values are 0.0 - check data collection or preprocessing logic");
                    result.Suggestions.Add("💡 This often indicates sensor malfunction or incomplete data collection");
                }

                // Suggest alternative columns with variation
                var alternatives = FindColumnsWithVariation(dataFile, columnNames, dataLines);
                if (alternatives.Any())
                {
                    result.Suggestions.Add($"✅ Alternative columns with variation: {string.Join(", ", alternatives.Take(3))}");
                }

                return result;
            }

            // Check 2: Very low variance (nearly constant)
            var mean = labelValues.Average();
            var variance = labelValues.Select(v => Math.Pow(v - mean, 2)).Average();
            var stdDev = Math.Sqrt(variance);
            var coefficientOfVariation = Math.Abs(mean) > 0.0001 ? stdDev / Math.Abs(mean) : 0;

            if (coefficientOfVariation < 0.01 && uniqueValues.Count > 1) // CV < 1%
            {
                result.Warnings.Add($"⚠ Label column '{labelColumn}' has very low variance (CV={coefficientOfVariation:P2})");
                result.Warnings.Add("💡 Model performance may be poor with nearly constant label values");
            }

            return result;
        }
        catch (Exception ex)
        {
            result.IsValid = false;
            result.ErrorMessage = $"Error validating data quality: {ex.Message}";
            return result;
        }
    }

    /// <summary>
    /// Finds columns that have variation (not constant)
    /// </summary>
    private static List<string> FindColumnsWithVariation(string dataFile, string[] columnNames, string[] dataLines)
    {
        var columnsWithVariation = new List<(string Name, int UniqueCount, double Min, double Max)>();

        for (int colIndex = 0; colIndex < columnNames.Length; colIndex++)
        {
            var columnName = columnNames[colIndex];
            var values = new List<double>();

            foreach (var line in dataLines)
            {
                var cells = line.Split(',');
                if (colIndex < cells.Length && double.TryParse(cells[colIndex], out var value))
                {
                    values.Add(value);
                }
            }

            if (values.Count > 0)
            {
                var uniqueCount = values.Distinct().Count();
                if (uniqueCount > 1) // Has variation
                {
                    var min = values.Min();
                    var max = values.Max();
                    columnsWithVariation.Add((columnName, uniqueCount, min, max));
                }
            }
        }

        // Return columns sorted by variation (unique count)
        return columnsWithVariation
            .OrderByDescending(c => c.UniqueCount)
            .Select(c => $"{c.Name} (min={c.Min:F1}, max={c.Max:F1})")
            .ToList();
    }

    /// <summary>
    /// Checks if dataset has sufficient samples for the number of features
    /// </summary>
    private static void CheckDatasetSize(string dataFile, DataQualityResult result)
    {
        try
        {
            // Read file with UTF-8 encoding
            string? firstLine;
            using (var reader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine();
            }

            if (string.IsNullOrEmpty(firstLine))
            {
                return; // Already handled by label validation
            }

            var featureCount = firstLine.Split(',').Length - 1; // Exclude label column
            var allLines = File.ReadAllLines(dataFile, System.Text.Encoding.UTF8);
            var sampleCount = allLines.Length - 1; // Exclude header

            if (sampleCount <= 0)
            {
                return; // Already handled by label validation
            }

            // Statistical rule of thumb: minimum 10× features
            var recommendedMin = featureCount * 10;

            if (sampleCount < recommendedMin)
            {
                var ratio = (double)sampleCount / recommendedMin;
                result.Warnings.Add(
                    $"⚠ Small dataset detected: {sampleCount} samples for {featureCount} features");
                result.Warnings.Add(
                    $"   Statistical rule of thumb: minimum {recommendedMin} samples (10× features)");
                result.Warnings.Add(
                    $"   Current dataset is {ratio:P0} of recommended minimum");
                result.Warnings.Add(
                    $"   Model performance may be poor due to insufficient data");
                result.Warnings.Add("");
                result.Suggestions.Add("💡 Consider:");
                result.Suggestions.Add($"   1. Collect more data (target: ≥{recommendedMin} samples)");
                result.Suggestions.Add("   2. Reduce feature count through feature selection");
                result.Suggestions.Add("   3. Use simpler models or increase regularization");
            }
            else if (sampleCount < recommendedMin * 2)
            {
                result.Warnings.Add(
                    $"⚠ Borderline dataset size: {sampleCount} samples for {featureCount} features");
                result.Warnings.Add(
                    $"   Recommended: ≥{recommendedMin * 2} samples for robust model");
                result.Warnings.Add(
                    $"   Performance may improve with more data");
            }
        }
        catch
        {
            // Non-critical check, ignore errors
        }
    }
}

/// <summary>
/// Result of data quality validation
/// </summary>
public class DataQualityResult
{
    public bool IsValid { get; set; }
    public string? ErrorMessage { get; set; }
    public string? ErrorMessageEn { get; set; }
    public List<string> Warnings { get; set; } = new();
    public List<string> Suggestions { get; set; } = new();
}
