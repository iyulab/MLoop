using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Prediction;

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
        // Skip label validation for unsupervised tasks
        if (string.IsNullOrEmpty(labelColumn))
            return new DataQualityResult { IsValid = true };

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
        // Flatten multiline quoted fields before line-by-line processing
        dataFile = CsvDataLoader.FlattenMultiLineQuotedFields(dataFile);

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
                "text-classification" => true,
                "image-classification" => true,
                _ => false
            };

            // Unsupervised tasks don't require label validation
            var isUnsupervisedTask = taskType?.ToLowerInvariant() switch
            {
                "anomaly-detection" => true,
                "clustering" => true,
                "time-series-anomaly" => true,
                _ => false
            };

            if (isUnsupervisedTask)
            {
                // Skip label-related warnings for unsupervised tasks
                return result;
            }

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

                    // Check 4: insufficient samples in some class
                    CheckPerClassMinimumSamples(textLabelValues, uniqueClasses, taskType, result);

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
                CheckPerClassMinimumSamples(allLabelStrings, uniqueNumericClasses, taskType, result);
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
    /// Checks whether every class has enough samples for the train/test split and cross-validation
    /// to be meaningful. Rejects the provably untrainable case and warns about the merely risky ones.
    /// </summary>
    /// <remarks>
    /// This used to be multiclass-only, which excluded exactly the shape that fails hardest in
    /// practice: an extremely imbalanced binary set (e.g. 3 positives in 357 rows). The per-class
    /// sample count matters identically in both — the class count never made it a different problem.
    /// </remarks>
    private static void CheckPerClassMinimumSamples(
        List<string> labelValues, List<string> uniqueClasses, string? taskType, DataQualityResult result)
    {
        var isClassification = taskType?.ToLowerInvariant()
            is "binary-classification" or "multiclass-classification";
        if (!isClassification || uniqueClasses.Count < 2)
            return;

        var classCounts = uniqueClasses
            .Select(c => new { Class = c, Count = labelValues.Count(v => v == c) })
            .OrderBy(x => x.Count)
            .ToList();

        var totalSamples = labelValues.Count;
        var classCount = uniqueClasses.Count;
        var smallestClass = classCounts.First();

        // Below this a class cannot exist in the train and test partitions at the same time: a
        // stratified split has to keep at least one row in train, so a single-sample class leaves the
        // test partition without it and every metric that needs the class is undefined. That follows
        // from the split arithmetic rather than being a tuned heuristic. Measured at 2 samples the run
        // completes and reports honest (degenerate) metrics for the promotion gate to block, so the
        // floor is exactly 2 — see below for when a starved class is fatal rather than merely lossy.
        const int UNTRAINABLE_PER_CLASS = 2;
        // Per-class minimum: need at least 5 samples per class for any CV fold to work
        const int MINIMUM_PER_CLASS = 5;
        // Recommended minimum: 15 samples per class for stable cross-validation
        const int RECOMMENDED_PER_CLASS = 15;

        var untrainableClasses = classCounts.Where(c => c.Count < UNTRAINABLE_PER_CLASS).ToList();
        if (untrainableClasses.Count > 0)
        {
            // Losing a class is only fatal when it leaves nothing to classify. Measured on both
            // shapes: 356 negatives + 1 positive dies inside ML.NET after burning the full time
            // budget, because the one class that mattered cannot be evaluated — while 80/80/1 across
            // three classes trains fine and produces a model that is genuinely useful for the two
            // healthy classes. Rejecting the second case would remove working capability, so the
            // rejection is scoped to "fewer than two classes survive", not "some class is starved".
            var viableClassCount = classCount - untrainableClasses.Count;
            var starved = string.Join(", ", untrainableClasses.Select(c => $"'{c.Class}' ({c.Count} sample)"));

            if (viableClassCount < 2)
            {
                result.IsValid = false;
                // Drop the imbalance advice collected just before this. It suggests oversampling and
                // shorter time limits, which are answers to "training will be unstable" — not to
                // "there is nothing left to tell apart". Two competing explanations for one rejection
                // is worse than one correct explanation.
                result.Suggestions.Clear();
                result.Warnings.Clear();
                result.ErrorMessage =
                    $"Only {viableClassCount} class has enough samples to train on: " +
                    $"{starved} {(untrainableClasses.Count > 1 ? "have" : "has")} " +
                    $"fewer than {UNTRAINABLE_PER_CLASS} samples";
                result.ErrorMessageEn =
                    "A class needs at least one row in the training set and one in the test set. With a " +
                    "single sample it can only be in one of them, so the model cannot be evaluated on it — " +
                    "and here that leaves no second class to classify against.";
                result.Suggestions.Add("Class distribution:");
                foreach (var c in classCounts)
                {
                    result.Suggestions.Add($"   '{c.Class}': {c.Count} samples ({(double)c.Count / totalSamples:P1})");
                }
                result.Suggestions.Add("");
                result.Suggestions.Add("🔧 Options:");
                result.Suggestions.Add($"   1. Collect more samples for the rare class (target: ≥{RECOMMENDED_PER_CLASS})");
                result.Suggestions.Add("   2. If the rare class is what you want to find, use --task anomaly-detection");
                return;
            }

            result.Warnings.Add(
                $"🔴 {starved} — too few samples to appear in both the train and test sets");
            result.Warnings.Add(
                $"   Training continues on the remaining {viableClassCount} classes, but the model will not " +
                "learn or be evaluated on the class(es) above");
            result.Suggestions.Add($"💡 Collect more samples for the rare class (target: ≥{RECOMMENDED_PER_CLASS}) or drop it from the data");
        }

        var classesBelow5 = classCounts.Where(c => c.Count < MINIMUM_PER_CLASS).ToList();
        var classesBelow15 = classCounts.Where(c => c.Count < RECOMMENDED_PER_CLASS).ToList();

        if (classesBelow5.Count > 0)
        {
            result.Warnings.Add($"🔴 Training at risk: {classesBelow5.Count} of {classCount} classes have fewer than {MINIMUM_PER_CLASS} samples");
            foreach (var c in classesBelow5)
            {
                result.Warnings.Add($"   Class '{c.Class}': {c.Count} samples");
            }
            result.Warnings.Add($"   Total dataset: {totalSamples} rows, {classCount} classes");
            result.Warnings.Add("");
            result.Warnings.Add("   Cross-validation folds will likely have empty classes, causing AutoML to fail");
            result.Warnings.Add("");
            result.Suggestions.Add("🔧 For datasets with rare classes:");
            result.Suggestions.Add($"   1. Collect more data (target: ≥{RECOMMENDED_PER_CLASS} samples per class, ≥{classCount * RECOMMENDED_PER_CLASS} total)");
            result.Suggestions.Add("   2. Merge similar classes to reduce class count");
            result.Suggestions.Add("   3. Remove classes with very few samples");
            result.Suggestions.Add("   4. Use --time 30 to limit training complexity");
        }
        else if (classesBelow15.Count > 0)
        {
            result.Warnings.Add($"🟡 Training may be unstable: {classesBelow15.Count} of {classCount} classes have fewer than {RECOMMENDED_PER_CLASS} samples");
            result.Warnings.Add($"   Smallest class '{smallestClass.Class}': {smallestClass.Count} samples");
            result.Warnings.Add($"   Total dataset: {totalSamples} rows, {classCount} classes");
            result.Suggestions.Add($"💡 Target: ≥{RECOMMENDED_PER_CLASS} samples per class for stable training");
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

            // Absolute minimum row count check
            const int CRITICAL_MINIMUM = 10;
            const int RECOMMENDED_MINIMUM = 30;

            if (sampleCount < CRITICAL_MINIMUM)
            {
                result.Warnings.Add($"🔴 Critically small dataset: only {sampleCount} rows");
                result.Warnings.Add($"   Minimum recommended: {CRITICAL_MINIMUM} rows for any ML task");
                result.Warnings.Add("   Train/Test split may produce empty test sets, causing metric calculation failures");
                result.Warnings.Add("");
                result.Suggestions.Add($"💡 Collect at least {RECOMMENDED_MINIMUM}+ rows before training");
            }
            else if (sampleCount < RECOMMENDED_MINIMUM)
            {
                result.Warnings.Add($"🟡 Very small dataset: {sampleCount} rows");
                result.Warnings.Add($"   Recommended minimum: {RECOMMENDED_MINIMUM} rows for reliable results");
                result.Warnings.Add("   Model performance may be unreliable with limited data");
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
