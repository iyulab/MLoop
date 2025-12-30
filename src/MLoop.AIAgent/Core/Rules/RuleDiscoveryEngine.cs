using System.Globalization;
using System.Text.RegularExpressions;

namespace MLoop.AIAgent.Core.Rules;

/// <summary>
/// Options for rule discovery.
/// </summary>
public class RuleDiscoveryOptions
{
    /// <summary>Minimum percentage of records affected to consider a pattern.</summary>
    public double MinAffectedPercentage { get; set; } = 0.01; // 1%

    /// <summary>Minimum confidence to auto-approve a rule.</summary>
    public double AutoApproveConfidence { get; set; } = 0.95;

    /// <summary>Whether to detect date format variations.</summary>
    public bool DetectDateFormats { get; set; } = true;

    /// <summary>Whether to detect missing value patterns.</summary>
    public bool DetectMissingValues { get; set; } = true;

    /// <summary>Whether to detect type inconsistencies.</summary>
    public bool DetectTypeInconsistencies { get; set; } = true;

    /// <summary>Whether to detect outliers.</summary>
    public bool DetectOutliers { get; set; } = true;

    /// <summary>Standard deviation threshold for outlier detection.</summary>
    public double OutlierStdDevThreshold { get; set; } = 3.0;

    /// <summary>Columns to ignore during analysis.</summary>
    public HashSet<string> IgnoreColumns { get; set; } = [];
}

/// <summary>
/// Engine for discovering preprocessing rules from data patterns.
/// </summary>
public partial class RuleDiscoveryEngine
{
    private readonly RuleDiscoveryOptions _options;

    // Common missing value indicators
    private static readonly HashSet<string> MissingValueIndicators = new(StringComparer.OrdinalIgnoreCase)
    {
        "", "null", "nil", "na", "n/a", "nan", "none", "-", "--", ".", "unknown", "undefined", "missing"
    };

    // Date format patterns to detect
    private static readonly (string Pattern, string Format, string Example)[] DatePatterns =
    [
        (@"^\d{4}-\d{2}-\d{2}$", "yyyy-MM-dd", "2024-01-15"),
        (@"^\d{2}/\d{2}/\d{4}$", "dd/MM/yyyy", "15/01/2024"),
        (@"^\d{2}/\d{2}/\d{4}$", "MM/dd/yyyy", "01/15/2024"),
        (@"^\d{2}-\d{2}-\d{4}$", "dd-MM-yyyy", "15-01-2024"),
        (@"^\d{4}/\d{2}/\d{2}$", "yyyy/MM/dd", "2024/01/15"),
        (@"^\d{8}$", "yyyyMMdd", "20240115"),
    ];

    public RuleDiscoveryEngine(RuleDiscoveryOptions? options = null)
    {
        _options = options ?? new RuleDiscoveryOptions();
    }

    /// <summary>
    /// Discovers preprocessing rules from sample data.
    /// </summary>
    /// <param name="sample">Sample data as list of row dictionaries.</param>
    /// <param name="stage">Current processing stage (1-5).</param>
    /// <returns>List of discovered preprocessing rules.</returns>
    public List<PreprocessingRule> DiscoverRules(
        List<Dictionary<string, string>> sample,
        int stage = 1)
    {
        var rules = new List<PreprocessingRule>();

        if (sample.Count == 0)
            return rules;

        var columns = sample[0].Keys.Where(c => !_options.IgnoreColumns.Contains(c)).ToList();

        foreach (var column in columns)
        {
            var columnValues = sample
                .Select(row => row.GetValueOrDefault(column, ""))
                .ToList();

            // Detect missing values
            if (_options.DetectMissingValues)
            {
                var missingRule = DetectMissingValuePattern(column, columnValues, sample.Count, stage);
                if (missingRule != null)
                    rules.Add(missingRule);
            }

            // Detect date format variations
            if (_options.DetectDateFormats)
            {
                var dateRule = DetectDateFormatPattern(column, columnValues, sample.Count, stage);
                if (dateRule != null)
                    rules.Add(dateRule);
            }

            // Detect type inconsistencies
            if (_options.DetectTypeInconsistencies)
            {
                var typeRule = DetectTypeInconsistency(column, columnValues, sample.Count, stage);
                if (typeRule != null)
                    rules.Add(typeRule);
            }

            // Detect outliers (for numeric columns)
            if (_options.DetectOutliers)
            {
                var outlierRule = DetectOutlierPattern(column, columnValues, sample.Count, stage);
                if (outlierRule != null)
                    rules.Add(outlierRule);
            }

            // Detect whitespace issues
            var whitespaceRule = DetectWhitespaceIssues(column, columnValues, sample.Count, stage);
            if (whitespaceRule != null)
                rules.Add(whitespaceRule);
        }

        // Auto-approve rules that don't need HITL and have high confidence
        foreach (var rule in rules.Where(r => r.IsAutoResolvable && r.Confidence >= _options.AutoApproveConfidence))
        {
            rule.IsApproved = true;
            rule.ApprovedBy = "System (Auto)";
        }

        return rules;
    }

    /// <summary>
    /// Validates existing rules against new sample data.
    /// </summary>
    public List<RuleValidationResult> ValidateRules(
        List<PreprocessingRule> rules,
        List<Dictionary<string, string>> newSample)
    {
        var results = new List<RuleValidationResult>();

        foreach (var rule in rules)
        {
            var result = ValidateRule(rule, newSample);
            results.Add(result);
        }

        return results;
    }

    private RuleValidationResult ValidateRule(
        PreprocessingRule rule,
        List<Dictionary<string, string>> newSample)
    {
        int matchCount = 0;
        int exceptionCount = 0;
        var newPatterns = new List<string>();

        foreach (var row in newSample)
        {
            foreach (var column in rule.Columns)
            {
                if (!row.TryGetValue(column, out var value))
                    continue;

                // Check if rule pattern matches
                if (!string.IsNullOrEmpty(rule.Pattern))
                {
                    if (Regex.IsMatch(value, rule.Pattern))
                        matchCount++;
                    else
                        exceptionCount++;
                }
                else
                {
                    matchCount++;
                }
            }
        }

        var total = matchCount + exceptionCount;
        var updatedConfidence = total > 0 ? (double)matchCount / total : rule.Confidence;

        return new RuleValidationResult
        {
            Rule = rule,
            IsValid = updatedConfidence >= 0.8,
            UpdatedConfidence = updatedConfidence,
            MatchCount = matchCount,
            ExceptionCount = exceptionCount,
            NewPatterns = newPatterns,
            Message = $"Matched {matchCount}/{total} records (Confidence: {updatedConfidence:P1})"
        };
    }

    private PreprocessingRule? DetectMissingValuePattern(
        string column,
        List<string> values,
        int totalSample,
        int stage)
    {
        var missingCount = values.Count(v => IsMissingValue(v));
        var percentage = (double)missingCount / totalSample;

        if (percentage < _options.MinAffectedPercentage)
            return null;

        // Determine the most common missing value indicator
        var missingPatterns = values
            .Where(IsMissingValue)
            .GroupBy(v => v.ToLowerInvariant())
            .OrderByDescending(g => g.Count())
            .Select(g => g.Key)
            .Take(3)
            .ToList();

        return new PreprocessingRule
        {
            Name = $"Handle missing values in '{column}'",
            Description = $"Found {missingCount} missing values ({percentage:P1}) with patterns: {string.Join(", ", missingPatterns)}",
            Type = RuleType.MissingValueStrategy,
            Columns = [column],
            Pattern = string.Join("|", missingPatterns.Select(Regex.Escape)),
            Transformation = "Replace with [mean/median/mode/default] - REQUIRES DECISION",
            RequiresHITL = true,
            Confidence = 1.0 - percentage, // Higher missing = lower confidence
            MatchCount = missingCount,
            AffectedPercentage = percentage,
            Severity = percentage > 0.1 ? IssueSeverity.High : IssueSeverity.Medium,
            DiscoveredAtStage = stage
        };
    }

    private PreprocessingRule? DetectDateFormatPattern(
        string column,
        List<string> values,
        int totalSample,
        int stage)
    {
        var dateFormats = new Dictionary<string, int>();
        var nonEmptyValues = values.Where(v => !string.IsNullOrWhiteSpace(v)).ToList();

        foreach (var value in nonEmptyValues)
        {
            foreach (var (pattern, format, _) in DatePatterns)
            {
                if (Regex.IsMatch(value, pattern))
                {
                    dateFormats[format] = dateFormats.GetValueOrDefault(format) + 1;
                    break;
                }
            }
        }

        // Need at least 2 different formats to create a rule
        if (dateFormats.Count < 2)
            return null;

        var totalMatched = dateFormats.Values.Sum();
        var percentage = (double)totalMatched / totalSample;

        if (percentage < _options.MinAffectedPercentage)
            return null;

        var formatList = string.Join(", ", dateFormats.Select(kv => $"{kv.Key} ({kv.Value})"));

        return new PreprocessingRule
        {
            Name = $"Standardize date format in '{column}'",
            Description = $"Found {dateFormats.Count} date formats: {formatList}",
            Type = RuleType.DateFormatStandardization,
            Columns = [column],
            Transformation = "Convert all to ISO-8601 (yyyy-MM-dd)",
            RequiresHITL = false,
            Confidence = percentage,
            MatchCount = totalMatched,
            AffectedPercentage = percentage,
            Severity = IssueSeverity.Low,
            DiscoveredAtStage = stage
        };
    }

    private PreprocessingRule? DetectTypeInconsistency(
        string column,
        List<string> values,
        int totalSample,
        int stage)
    {
        var typeCategories = new Dictionary<string, int>
        {
            ["numeric"] = 0,
            ["text"] = 0,
            ["empty"] = 0
        };

        foreach (var value in values)
        {
            if (string.IsNullOrWhiteSpace(value))
                typeCategories["empty"]++;
            else if (double.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out _))
                typeCategories["numeric"]++;
            else
                typeCategories["text"]++;
        }

        var nonEmpty = typeCategories["numeric"] + typeCategories["text"];
        if (nonEmpty == 0)
            return null;

        // Check for mixed numeric/text
        var numericRatio = (double)typeCategories["numeric"] / nonEmpty;

        if (numericRatio > 0.1 && numericRatio < 0.9)
        {
            var mixedCount = Math.Min(typeCategories["numeric"], typeCategories["text"]);
            var percentage = (double)mixedCount / totalSample;

            if (percentage < _options.MinAffectedPercentage)
                return null;

            return new PreprocessingRule
            {
                Name = $"Resolve type inconsistency in '{column}'",
                Description = $"Column has mixed types: {typeCategories["numeric"]} numeric, {typeCategories["text"]} text values",
                Type = RuleType.UnknownCategoryMapping,
                Columns = [column],
                Transformation = "Determine target type and convert - REQUIRES DECISION",
                RequiresHITL = true,
                Confidence = Math.Max(numericRatio, 1 - numericRatio),
                MatchCount = mixedCount,
                AffectedPercentage = percentage,
                Severity = IssueSeverity.High,
                DiscoveredAtStage = stage
            };
        }

        return null;
    }

    private PreprocessingRule? DetectOutlierPattern(
        string column,
        List<string> values,
        int totalSample,
        int stage)
    {
        var numericValues = values
            .Where(v => double.TryParse(v, NumberStyles.Any, CultureInfo.InvariantCulture, out _))
            .Select(v => double.Parse(v, CultureInfo.InvariantCulture))
            .ToList();

        if (numericValues.Count < 10)
            return null;

        var mean = numericValues.Average();
        var stdDev = Math.Sqrt(numericValues.Select(v => Math.Pow(v - mean, 2)).Average());

        if (stdDev < 0.001)
            return null;

        var threshold = _options.OutlierStdDevThreshold;
        var outliers = numericValues
            .Where(v => Math.Abs(v - mean) > threshold * stdDev)
            .ToList();

        if (outliers.Count == 0)
            return null;

        var percentage = (double)outliers.Count / totalSample;

        if (percentage < _options.MinAffectedPercentage)
            return null;

        return new PreprocessingRule
        {
            Name = $"Handle outliers in '{column}'",
            Description = $"Found {outliers.Count} outliers ({percentage:P1}) beyond {threshold}σ. Range: [{outliers.Min():F2}, {outliers.Max():F2}]",
            Type = RuleType.OutlierHandling,
            Columns = [column],
            Transformation = "Handle outliers (remove/cap/keep) - REQUIRES DECISION",
            RequiresHITL = true,
            Confidence = 1.0 - percentage,
            MatchCount = outliers.Count,
            AffectedPercentage = percentage,
            Severity = percentage > 0.05 ? IssueSeverity.High : IssueSeverity.Medium,
            DiscoveredAtStage = stage
        };
    }

    private PreprocessingRule? DetectWhitespaceIssues(
        string column,
        List<string> values,
        int totalSample,
        int stage)
    {
        var issueCount = values.Count(v =>
            !string.IsNullOrEmpty(v) &&
            (v != v.Trim() || WhitespaceRegex().IsMatch(v)));

        var percentage = (double)issueCount / totalSample;

        if (percentage < _options.MinAffectedPercentage)
            return null;

        return new PreprocessingRule
        {
            Name = $"Normalize whitespace in '{column}'",
            Description = $"Found {issueCount} values with whitespace issues (leading/trailing/multiple spaces)",
            Type = RuleType.WhitespaceNormalization,
            Columns = [column],
            Transformation = "Trim and collapse multiple spaces",
            RequiresHITL = false,
            Confidence = 1.0,
            MatchCount = issueCount,
            AffectedPercentage = percentage,
            Severity = IssueSeverity.Low,
            DiscoveredAtStage = stage
        };
    }

    private static bool IsMissingValue(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return true;
        return MissingValueIndicators.Contains(value.Trim());
    }

    [GeneratedRegex(@"\s{2,}")]
    private static partial Regex WhitespaceRegex();
}
