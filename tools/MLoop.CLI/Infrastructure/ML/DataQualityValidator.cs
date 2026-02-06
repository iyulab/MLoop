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
    public DataQualityResult ValidateTrainingData(string dataFile, string labelColumn, string? taskType = null)
    {
        // First validate label column
        var result = ValidateLabelColumn(dataFile, labelColumn, taskType);
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
    private DataQualityResult ValidateLabelColumn(string dataFile, string labelColumn, string? taskType = null)
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

            var columnNames = CsvFieldParser.ParseFields(firstLine);
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

            // Determine if this is a classification task (text labels allowed)
            var isClassificationTask = taskType?.ToLowerInvariant() switch
            {
                "binary-classification" => true,
                "multiclass-classification" => true,
                _ => false
            };

            // Extract label column values - support both numeric and text labels
            var numericLabelValues = new List<double>();
            var textLabelValues = new List<string>();
            var parseErrors = 0;

            foreach (var line in dataLines)
            {
                var values = CsvFieldParser.ParseFields(line);
                if (labelColumnIndex < values.Length)
                {
                    var rawValue = values[labelColumnIndex].Trim();
                    if (double.TryParse(rawValue, out var numericValue))
                    {
                        numericLabelValues.Add(numericValue);
                        textLabelValues.Add(rawValue);
                    }
                    else if (!string.IsNullOrWhiteSpace(rawValue))
                    {
                        // Text label (valid for classification)
                        textLabelValues.Add(rawValue);
                        parseErrors++;
                    }
                }
            }

            // For classification tasks, text labels are valid
            if (isClassificationTask && textLabelValues.Count > 0)
            {
                var uniqueClasses = textLabelValues.Distinct().ToList();
                result.UniqueClassCount = uniqueClasses.Count;

                // All text labels - valid for classification
                if (numericLabelValues.Count == 0)
                {
                    // Check 1: Only one unique class
                    if (uniqueClasses.Count == 1)
                    {
                        result.IsValid = false;
                        result.ErrorMessage = $"Label column '{labelColumn}' contains only one class: '{uniqueClasses[0]}'";
                        result.ErrorMessageEn = "Cannot train classifier with only one class";
                        return result;
                    }

                    // Check 2: Binary classification with more than 2 classes
                    if (taskType == "binary-classification" && uniqueClasses.Count > 2)
                    {
                        result.Warnings.Add($"⚠ Found {uniqueClasses.Count} classes but task is binary-classification");
                        result.Suggestions.Add("💡 Consider using --task multiclass-classification");
                    }

                    // Check 3: Class imbalance detection
                    CheckClassImbalance(textLabelValues, uniqueClasses, result);

                    // Valid text labels for classification
                    return result;
                }
            }

            // For numeric classification labels, also check class imbalance
            if (isClassificationTask && numericLabelValues.Count > 0)
            {
                var uniqueNumericClasses = numericLabelValues.Select(v => v.ToString()).Distinct().ToList();
                var allLabelStrings = numericLabelValues.Select(v => v.ToString()).ToList();
                CheckClassImbalance(allLabelStrings, uniqueNumericClasses, result);
            }

            // For regression or mixed labels, require numeric values
            if (parseErrors > 0 && !isClassificationTask)
            {
                result.Warnings.Add($"⚠ {parseErrors} values in label column '{labelColumn}' could not be parsed as numbers");
            }

            if (numericLabelValues.Count == 0 && !isClassificationTask)
            {
                result.IsValid = false;
                result.ErrorMessage = $"No valid numeric values found in label column '{labelColumn}'";
                result.Suggestions.Add("💡 For text labels, use --task binary-classification or --task multiclass-classification");
                return result;
            }

            // For numeric labels, check variance
            if (numericLabelValues.Count > 0)
            {
                // Check 1: All same value (no variance)
                var uniqueValues = numericLabelValues.Distinct().ToList();
                result.UniqueClassCount = uniqueValues.Count;

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

                // Check 2: Very low variance (nearly constant) - only for regression
                if (!isClassificationTask)
                {
                    var mean = numericLabelValues.Average();
                    var variance = numericLabelValues.Select(v => Math.Pow(v - mean, 2)).Average();
                    var stdDev = Math.Sqrt(variance);
                    var coefficientOfVariation = Math.Abs(mean) > 0.0001 ? stdDev / Math.Abs(mean) : 0;

                    if (coefficientOfVariation < 0.01 && uniqueValues.Count > 1) // CV < 1%
                    {
                        result.Warnings.Add($"⚠ Label column '{labelColumn}' has very low variance (CV={coefficientOfVariation:P2})");
                        result.Warnings.Add("💡 Model performance may be poor with nearly constant label values");
                    }
                }
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
    /// Checks for class imbalance in classification tasks
    /// </summary>
    private static void CheckClassImbalance(List<string> labelValues, List<string> uniqueClasses, DataQualityResult result)
    {
        if (uniqueClasses.Count < 2)
        {
            return; // Already handled by single-class check
        }

        // Count each class
        var classCounts = uniqueClasses
            .Select(c => new { Class = c, Count = labelValues.Count(v => v == c) })
            .OrderByDescending(x => x.Count)
            .ToList();

        var majorityClass = classCounts.First();
        var minorityClass = classCounts.Last();
        var totalSamples = labelValues.Count;

        // Calculate imbalance ratio
        var imbalanceRatio = minorityClass.Count > 0
            ? (double)majorityClass.Count / minorityClass.Count
            : double.MaxValue;

        // Store class distribution info
        result.ClassDistribution = classCounts.ToDictionary(x => x.Class, x => x.Count);
        result.ImbalanceRatio = imbalanceRatio;

        // Warning levels based on imbalance severity
        const double MODERATE_THRESHOLD = 5.0;   // 5:1
        const double HIGH_THRESHOLD = 10.0;      // 10:1
        const double EXTREME_THRESHOLD = 20.0;   // 20:1
        const double CRITICAL_THRESHOLD = 50.0;  // 50:1 - likely to fail CV

        if (imbalanceRatio >= CRITICAL_THRESHOLD)
        {
            // Critical imbalance - likely to fail cross-validation
            result.Warnings.Add($"🔴 Critical class imbalance detected: {imbalanceRatio:F1}:1 ratio");
            result.Warnings.Add($"   Majority class '{majorityClass.Class}': {majorityClass.Count} samples ({(double)majorityClass.Count / totalSamples:P1})");
            result.Warnings.Add($"   Minority class '{minorityClass.Class}': {minorityClass.Count} samples ({(double)minorityClass.Count / totalSamples:P1})");
            result.Warnings.Add("");
            result.Warnings.Add("⚠️  AutoML may fail with 'AUC not defined' error due to insufficient minority class samples in cross-validation folds");
            result.Warnings.Add("");
            result.Suggestions.Add("🔧 RECOMMENDED WORKAROUNDS:");
            result.Suggestions.Add("   1. Oversample minority class before training:");
            result.Suggestions.Add("      - Simple replication: Duplicate minority samples 10-20x");
            result.Suggestions.Add("      - SMOTE: Generate synthetic samples (requires external tool)");
            result.Suggestions.Add("   2. Use a shorter training time with --time 30 to reduce CV folds");
            result.Suggestions.Add("   3. Consider treating this as anomaly detection instead of classification");
        }
        else if (imbalanceRatio >= EXTREME_THRESHOLD)
        {
            result.Warnings.Add($"🟠 Extreme class imbalance detected: {imbalanceRatio:F1}:1 ratio");
            result.Warnings.Add($"   Minority class '{minorityClass.Class}' has only {minorityClass.Count} samples ({(double)minorityClass.Count / totalSamples:P1})");
            result.Warnings.Add("   This may cause training instability or poor minority class recall");
            result.Warnings.Add("");
            result.Suggestions.Add("💡 Consider oversampling minority class or using class weights");
        }
        else if (imbalanceRatio >= HIGH_THRESHOLD)
        {
            result.Warnings.Add($"🟡 High class imbalance: {imbalanceRatio:F1}:1 ratio");
            result.Warnings.Add($"   Minority class has {minorityClass.Count} samples ({(double)minorityClass.Count / totalSamples:P1})");
            result.Suggestions.Add("💡 Use F1-score or recall instead of accuracy: --metric f1_score");
        }
        else if (imbalanceRatio >= MODERATE_THRESHOLD)
        {
            result.Warnings.Add($"⚠ Moderate class imbalance: {imbalanceRatio:F1}:1 ratio");
            result.Suggestions.Add("💡 Consider using F1-score for evaluation: --metric f1_score");
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
                var cells = CsvFieldParser.ParseFields(line);
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

            var featureCount = CsvFieldParser.ParseFields(firstLine).Length - 1; // Exclude label column
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

    /// <summary>
    /// Number of unique classes/values in label column (for classification tasks)
    /// </summary>
    public int UniqueClassCount { get; set; }

    /// <summary>
    /// Class distribution for classification tasks (class name -> sample count)
    /// </summary>
    public Dictionary<string, int>? ClassDistribution { get; set; }

    /// <summary>
    /// Ratio of majority to minority class (e.g., 10.0 means 10:1 imbalance)
    /// </summary>
    public double ImbalanceRatio { get; set; }
}
