using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Diagnostics;

/// <summary>
/// Analyzes class distribution for classification tasks and provides balance assessments.
/// Part of T4.5 - Class Distribution Analysis feature.
/// </summary>
public class ClassDistributionAnalyzer
{
    private readonly ICsvHelper _csvHelper;

    public ClassDistributionAnalyzer(ICsvHelper csvHelper)
    {
        _csvHelper = csvHelper ?? throw new ArgumentNullException(nameof(csvHelper));
    }

    /// <summary>
    /// Analyzes the distribution of classes in the label column.
    /// </summary>
    /// <param name="csvPath">Path to CSV file</param>
    /// <param name="labelColumn">Name of the label column</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Class distribution analysis result</returns>
    public async Task<ClassDistributionResult> AnalyzeAsync(
        string csvPath,
        string labelColumn,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(csvPath))
        {
            throw new FileNotFoundException($"CSV file not found: {csvPath}");
        }

        // Streamed: only the label column's counts are kept, never the rows.
        var classCounts = new Dictionary<string, int>();
        var totalRows = 0;
        var labelIndex = -1;
        await foreach (var row in _csvHelper.ReadRowsAsync(csvPath, headers => labelIndex = IndexOf(headers, labelColumn),
            cancellationToken: cancellationToken).ConfigureAwait(false))
        {
            totalRows++;
            if (labelIndex < 0)
            {
                continue;
            }

            var value = row[labelIndex];
            if (string.IsNullOrWhiteSpace(value))
            {
                value = "(empty)";
            }

            classCounts.TryGetValue(value, out var count);
            classCounts[value] = count + 1;
        }

        if (totalRows == 0)
        {
            return new ClassDistributionResult
            {
                LabelColumn = labelColumn,
                TotalRows = 0,
                Error = "Empty dataset"
            };
        }

        if (labelIndex < 0)
        {
            return new ClassDistributionResult
            {
                LabelColumn = labelColumn,
                TotalRows = totalRows,
                Error = $"Label column '{labelColumn}' not found"
            };
        }

        classCounts = FoldNumericSpellings(classCounts);

        // Calculate distribution metrics
        var classCount = classCounts.Count;
        var classDistribution = classCounts
            .OrderByDescending(kv => kv.Value)
            .ToDictionary(kv => kv.Key, kv => kv.Value);

        var minCount = classCounts.Values.Min();
        var maxCount = classCounts.Values.Max();
        var avgCount = classCounts.Values.Average();

        // Calculate imbalance ratio (max / min)
        var imbalanceRatio = minCount > 0 ? (double)maxCount / minCount : double.MaxValue;

        // Determine balance level
        var balanceLevel = DetermineBalanceLevel(imbalanceRatio, classCount);

        // Generate result
        var result = new ClassDistributionResult
        {
            LabelColumn = labelColumn,
            TotalRows = totalRows,
            ClassCount = classCount,
            ClassDistribution = classDistribution,
            MinClassCount = minCount,
            MaxClassCount = maxCount,
            AverageClassCount = avgCount,
            ImbalanceRatio = imbalanceRatio,
            BalanceLevel = balanceLevel
        };

        // Generate visualization
        result.DistributionVisualization = GenerateVisualization(classDistribution, totalRows);

        // Add warnings and suggestions based on analysis
        AddWarningsAndSuggestions(result);

        return result;
    }

    private static int IndexOf(IReadOnlyList<string> headers, string name)
    {
        for (var i = 0; i < headers.Count; i++)
        {
            if (headers[i] == name)
            {
                return i;
            }
        }
        return -1;
    }

    private static ClassBalanceLevel DetermineBalanceLevel(double imbalanceRatio, int classCount)
    {
        // Single class: not trainable, report as severely imbalanced
        if (classCount <= 1)
        {
            return ClassBalanceLevel.SeverelyImbalanced;
        }

        // For binary classification
        if (classCount == 2)
        {
            if (imbalanceRatio <= 1.5) return ClassBalanceLevel.Balanced;
            if (imbalanceRatio <= 3) return ClassBalanceLevel.SlightlyImbalanced;
            if (imbalanceRatio <= 10) return ClassBalanceLevel.ModeratelyImbalanced;
            if (imbalanceRatio <= 50) return ClassBalanceLevel.HighlyImbalanced;
            return ClassBalanceLevel.SeverelyImbalanced;
        }

        // For multiclass
        if (imbalanceRatio <= 2) return ClassBalanceLevel.Balanced;
        if (imbalanceRatio <= 5) return ClassBalanceLevel.SlightlyImbalanced;
        if (imbalanceRatio <= 20) return ClassBalanceLevel.ModeratelyImbalanced;
        if (imbalanceRatio <= 100) return ClassBalanceLevel.HighlyImbalanced;
        return ClassBalanceLevel.SeverelyImbalanced;
    }

    /// <summary>
    /// A label written as numbers is read as numbers: <c>3</c> and <c>3.00</c> are one class, as the trainers see them. Counted as
    /// text they were two, so a label that holds one value (a column of 3s, some written <c>3.00</c>) passed the one-class refusal and
    /// the search ran for minutes before it stopped on "all identical values". Each class keeps the spelling it was most often written
    /// with. A label with a value that is not a number stays as text.
    /// </summary>
    internal static Dictionary<string, int> FoldNumericSpellings(Dictionary<string, int> counts)
    {
        var numeric = new Dictionary<string, double>();
        foreach (var key in counts.Keys)
        {
            if (key == "(empty)")
                continue;
            if (!double.TryParse(key, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var value)
                || !double.IsFinite(value))
                return counts;
            numeric[key] = value;
        }

        var folded = new Dictionary<string, int>();
        foreach (var group in numeric.GroupBy(kv => kv.Value == 0 ? 0 : kv.Value))
        {
            var spellings = group.Select(kv => kv.Key).ToList();
            folded[spellings.OrderByDescending(k => counts[k]).First()] = spellings.Sum(k => counts[k]);
        }
        if (counts.TryGetValue("(empty)", out var empty))
            folded["(empty)"] = empty;
        return folded;
    }

    /// <summary>Most classes the picture draws: a label of thousands of values shows its largest classes and counts the rest.</summary>
    internal const int MaxClassesDrawn = 20;

    internal static string GenerateVisualization(Dictionary<string, int> distribution, int total)
    {
        const int barWidth = 30;
        var lines = new List<string>();
        var drawn = distribution.Take(MaxClassesDrawn).ToList();
        var maxLabelLength = Math.Min(drawn.Max(kv => kv.Key.Length), 20); // Cap label display length
        var largest = distribution.Values.Max();   // once: inside the loop it made the picture cost classes squared

        foreach (var (className, count) in drawn)
        {
            var percentage = (double)count / total * 100;
            var barLength = (int)Math.Round((double)count / largest * barWidth);
            var bar = new string('█', barLength);
            var displayName = className.Length > 20 ? className[..17] + "..." : className;

            lines.Add($"  {displayName.PadRight(maxLabelLength)} │ {bar.PadRight(barWidth)} {count,6} ({percentage,5:F1}%)");
        }
        if (distribution.Count > drawn.Count)
        {
            var rest = distribution.Skip(drawn.Count).Select(kv => kv.Value).ToList();
            lines.Add($"  … and {rest.Count:N0} more classes ({rest.Min():N0} to {rest.Max():N0} rows each)");
        }

        return string.Join(Environment.NewLine, lines);
    }

    private static void AddWarningsAndSuggestions(ClassDistributionResult result)
    {
        // Check for single-class dataset (not trainable for classification)
        if (result.ClassCount <= 1)
        {
            result.Summary = $"Only {result.ClassCount} class found — cannot train a classifier";
            result.Warnings.Add("Label column contains only one unique value");
            result.Suggestions.Add("Check if the correct label column is specified");
            result.Suggestions.Add("This dataset may only contain one category (e.g., all 'normal' samples)");
            return;
        }

        // Check for empty class
        if (result.ClassDistribution.ContainsKey("(empty)"))
        {
            var emptyCount = result.ClassDistribution["(empty)"];
            result.Warnings.Add($"{emptyCount} rows have empty/missing labels");
            result.Suggestions.Add("Use --drop-missing-labels to remove rows with empty labels");
        }

        // Add warnings based on balance level
        switch (result.BalanceLevel)
        {
            case ClassBalanceLevel.Balanced:
                result.Summary = "Class distribution is well-balanced";
                break;

            case ClassBalanceLevel.SlightlyImbalanced:
                result.Summary = $"Slight class imbalance detected (ratio: {result.ImbalanceRatio:F1}:1)";
                result.Suggestions.Add("Model should handle this naturally - monitor minority class performance");
                break;

            case ClassBalanceLevel.ModeratelyImbalanced:
                result.Summary = $"Moderate class imbalance (ratio: {result.ImbalanceRatio:F1}:1)";
                result.Warnings.Add("Class imbalance may affect minority class prediction");
                result.Suggestions.Add("Consider using class weights or evaluation metrics like F1-score");
                result.SuggestedStrategies.Add(SamplingStrategy.ClassWeights);
                break;

            case ClassBalanceLevel.HighlyImbalanced:
                result.Summary = $"High class imbalance (ratio: {result.ImbalanceRatio:F1}:1)";
                result.Warnings.Add("Significant imbalance - minority classes may be poorly predicted");
                result.Suggestions.Add("Consider SMOTE oversampling for minority classes");
                result.Suggestions.Add("Use precision/recall and F1-score instead of accuracy");
                result.SuggestedStrategies.Add(SamplingStrategy.SMOTE);
                result.SuggestedStrategies.Add(SamplingStrategy.ClassWeights);
                break;

            case ClassBalanceLevel.SeverelyImbalanced:
                result.Summary = $"Severe class imbalance (ratio: {result.ImbalanceRatio:F1}:1)";
                result.Warnings.Add("Extreme imbalance - model may ignore minority classes entirely");
                result.Suggestions.Add("Consider combining undersampling majority with oversampling minority");
                result.Suggestions.Add("Evaluate using PR-AUC instead of ROC-AUC");
                result.Suggestions.Add("Consider collecting more minority class samples");
                result.SuggestedStrategies.Add(SamplingStrategy.CombinedSampling);
                result.SuggestedStrategies.Add(SamplingStrategy.SMOTE);
                break;
        }

        // Check for rare classes
        var rareClasses = result.ClassDistribution
            .Where(kv => kv.Value < 10 && kv.Key != "(empty)")
            .ToList();

        if (rareClasses.Count > 0)
        {
            result.Warnings.Add($"{rareClasses.Count} class(es) have fewer than 10 samples");
            foreach (var rc in rareClasses.Take(3))
            {
                result.Warnings.Add($"  - '{rc.Key}' has only {rc.Value} sample(s)");
            }
        }

        // Check for single-sample classes
        var singleSampleClasses = result.ClassDistribution
            .Where(kv => kv.Value == 1 && kv.Key != "(empty)")
            .ToList();

        if (singleSampleClasses.Count > 0)
        {
            result.Warnings.Add($"{singleSampleClasses.Count} class(es) have only 1 sample - cannot be used for training");
        }
    }
}

/// <summary>
/// Result of class distribution analysis.
/// </summary>
public class ClassDistributionResult
{
    /// <summary>
    /// Name of the label column analyzed.
    /// </summary>
    public string LabelColumn { get; set; } = "";

    /// <summary>
    /// Total number of rows in dataset.
    /// </summary>
    public int TotalRows { get; set; }

    /// <summary>
    /// Number of unique classes.
    /// </summary>
    public int ClassCount { get; set; }

    /// <summary>
    /// Distribution of classes (class name -> count), ordered by count descending.
    /// </summary>
    public Dictionary<string, int> ClassDistribution { get; set; } = new();

    /// <summary>
    /// Minimum class count.
    /// </summary>
    public int MinClassCount { get; set; }

    /// <summary>
    /// Maximum class count.
    /// </summary>
    public int MaxClassCount { get; set; }

    /// <summary>
    /// Average class count.
    /// </summary>
    public double AverageClassCount { get; set; }

    /// <summary>
    /// Ratio of largest to smallest class.
    /// </summary>
    public double ImbalanceRatio { get; set; }

    /// <summary>
    /// Overall balance level assessment.
    /// </summary>
    public ClassBalanceLevel BalanceLevel { get; set; }

    /// <summary>
    /// Visual representation of class distribution.
    /// </summary>
    public string DistributionVisualization { get; set; } = "";

    /// <summary>
    /// Summary of analysis.
    /// </summary>
    public string Summary { get; set; } = "";

    /// <summary>
    /// Warning messages.
    /// </summary>
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Suggested improvements.
    /// </summary>
    public List<string> Suggestions { get; set; } = new();

    /// <summary>
    /// Recommended sampling strategies.
    /// </summary>
    public List<SamplingStrategy> SuggestedStrategies { get; set; } = new();

    /// <summary>
    /// Error message if analysis failed.
    /// </summary>
    public string? Error { get; set; }

    /// <summary>
    /// Whether the distribution needs attention.
    /// </summary>
    public bool NeedsAttention => BalanceLevel >= ClassBalanceLevel.ModeratelyImbalanced;
}

/// <summary>
/// Class balance level classification.
/// </summary>
public enum ClassBalanceLevel
{
    Balanced = 0,
    SlightlyImbalanced = 1,
    ModeratelyImbalanced = 2,
    HighlyImbalanced = 3,
    SeverelyImbalanced = 4
}

/// <summary>
/// Suggested sampling strategies for handling class imbalance.
/// </summary>
public enum SamplingStrategy
{
    /// <summary>
    /// No sampling needed.
    /// </summary>
    None,

    /// <summary>
    /// Use class weights to penalize majority class errors more.
    /// </summary>
    ClassWeights,

    /// <summary>
    /// Synthetic Minority Oversampling Technique.
    /// </summary>
    SMOTE,

    /// <summary>
    /// Random undersampling of majority class.
    /// </summary>
    RandomUndersampling,

    /// <summary>
    /// Random oversampling of minority class.
    /// </summary>
    RandomOversampling,

    /// <summary>
    /// Combination of over and undersampling.
    /// </summary>
    CombinedSampling
}
