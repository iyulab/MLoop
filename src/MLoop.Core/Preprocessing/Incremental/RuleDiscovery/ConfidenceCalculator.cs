using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery;

/// <summary>
/// Calculates statistical confidence scores for preprocessing rules.
/// Measures how consistently and reliably a rule applies across samples.
/// </summary>
public sealed class ConfidenceCalculator
{
    /// <summary>
    /// Calculate confidence score for a rule across two samples.
    /// </summary>
    /// <param name="rule">Rule to evaluate.</param>
    /// <param name="previousSample">Previous sample (Stage N-1).</param>
    /// <param name="currentSample">Current sample (Stage N).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Confidence score with detailed metrics.</returns>
    public async Task<ConfidenceScore> CalculateAsync(
        PreprocessingRule rule,
        DataFrame previousSample,
        DataFrame currentSample,
        CancellationToken cancellationToken = default)
    {
        // Calculate each component
        var consistency = await CalculateConsistencyAsync(
            rule,
            previousSample,
            currentSample,
            cancellationToken);

        var coverage = CalculateCoverage(rule, currentSample);

        var stability = CalculateStability(rule, previousSample, currentSample);

        // Calculate overall score (weighted average)
        var overall = (consistency * 0.5) + (coverage * 0.3) + (stability * 0.2);

        return new ConfidenceScore
        {
            Consistency = consistency,
            Coverage = coverage,
            Stability = stability,
            Overall = overall,
            ExceptionCount = 0, // Will be updated if rule application fails
            TotalAttempts = 1   // Number of times rule was evaluated
        };
    }

    /// <summary>
    /// Consistency: Rule applies consistently to affected data (0-1).
    /// Measures: How often the rule successfully applies to data it should affect.
    /// </summary>
    private async Task<double> CalculateConsistencyAsync(
        PreprocessingRule rule,
        DataFrame previousSample,
        DataFrame currentSample,
        CancellationToken cancellationToken)
    {
        await Task.CompletedTask; // Async for future extensions
        cancellationToken.ThrowIfCancellationRequested();

        // Get the column(s) the rule applies to
        var columnName = rule.ColumnNames.First();

        if (!currentSample.Columns.Any(c => c.Name == columnName))
        {
            return 0.0; // Column doesn't exist
        }

        var column = currentSample.Columns[columnName];

        // Simulate rule application and count successes
        var applicableRows = CountApplicableRows(rule, column);
        var successfulApplications = SimulateRuleApplication(rule, column);

        if (applicableRows == 0)
        {
            return 1.0; // No rows to apply to = perfect consistency
        }

        return (double)successfulApplications / applicableRows;
    }

    /// <summary>
    /// Coverage: Percentage of data covered by this rule (0-1).
    /// Measures: What proportion of the dataset this rule affects.
    /// </summary>
    private static double CalculateCoverage(
        PreprocessingRule rule,
        DataFrame currentSample)
    {
        var columnName = rule.ColumnNames.First();

        if (!currentSample.Columns.Any(c => c.Name == columnName))
        {
            return 0.0;
        }

        var column = currentSample.Columns[columnName];
        var applicableRows = CountApplicableRows(rule, column);

        return (double)applicableRows / column.Length;
    }

    /// <summary>
    /// Stability: Rule definition unchanged across samples (0-1).
    /// Measures: How similar the rule's applicability is between samples.
    /// </summary>
    private static double CalculateStability(
        PreprocessingRule rule,
        DataFrame previousSample,
        DataFrame currentSample)
    {
        var columnName = rule.ColumnNames.First();

        // Check if column exists in both samples
        var prevColumn = previousSample.Columns.FirstOrDefault(c => c.Name == columnName);
        var currColumn = currentSample.Columns.FirstOrDefault(c => c.Name == columnName);

        if (prevColumn == null || currColumn == null)
        {
            return 0.0; // Column missing in one sample
        }

        // Calculate applicability in both samples
        var prevApplicableRatio = (double)CountApplicableRows(rule, prevColumn) / prevColumn.Length;
        var currApplicableRatio = (double)CountApplicableRows(rule, currColumn) / currColumn.Length;

        // Stability = 1 - absolute difference in ratios
        var difference = Math.Abs(prevApplicableRatio - currApplicableRatio);
        return 1.0 - difference;
    }

    /// <summary>
    /// Count rows where the rule is applicable based on pattern type.
    /// </summary>
    private static int CountApplicableRows(
        PreprocessingRule rule,
        DataFrameColumn column)
    {
        int count = 0;

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i]?.ToString();

            // Pattern-specific applicability logic
            bool isApplicable = rule.PatternType switch
            {
                PatternType.MissingValue => IsValueMissing(value),
                PatternType.WhitespaceIssue => HasWhitespaceIssue(value),
                PatternType.TypeInconsistency => true, // All rows potentially affected
                PatternType.FormatVariation => true,   // All rows potentially affected
                PatternType.OutlierAnomaly => IsNumericColumn(column),
                PatternType.CategoryVariation => !string.IsNullOrWhiteSpace(value),
                PatternType.EncodingIssue => HasEncodingIssue(value),
                PatternType.BusinessRule => true,
                _ => false
            };

            if (isApplicable)
                count++;
        }

        return count;
    }

    /// <summary>
    /// Simulate rule application and count successful applications.
    /// In real implementation, this would actually apply the rule.
    /// </summary>
    private static int SimulateRuleApplication(
        PreprocessingRule rule,
        DataFrameColumn column)
    {
        // For now, assume high success rate for well-defined rules
        // In real implementation, this would try to apply the rule and count successes
        var applicableRows = CountApplicableRows(rule, column);

        // Simulate success rate based on rule type
        var successRate = rule.Type switch
        {
            PreprocessingRuleType.WhitespaceNormalization => 0.99,
            PreprocessingRuleType.EncodingNormalization => 0.95,
            PreprocessingRuleType.DateFormatStandardization => 0.90,
            PreprocessingRuleType.NumericFormatStandardization => 0.90,
            PreprocessingRuleType.MissingValueStrategy => 0.85,
            PreprocessingRuleType.TypeConversion => 0.80,
            PreprocessingRuleType.OutlierHandling => 0.85,
            PreprocessingRuleType.CategoryMapping => 0.90,
            PreprocessingRuleType.BusinessLogicDecision => 0.75,
            _ => 0.80
        };

        return (int)(applicableRows * successRate);
    }

    // Helper methods for pattern detection
    private static bool IsValueMissing(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return true;

        var normalized = value.Trim().ToUpperInvariant();
        return normalized is "NULL" or "NA" or "N/A" or "NAN" or "NONE" or "-" or "?";
    }

    private static bool HasWhitespaceIssue(string? value)
    {
        if (string.IsNullOrEmpty(value))
            return false;

        return char.IsWhiteSpace(value[0]) ||
               char.IsWhiteSpace(value[^1]) ||
               value.Contains("  ") ||
               value.Contains('\t') ||
               value.Contains('\n');
    }

    private static bool HasEncodingIssue(string? value)
    {
        if (string.IsNullOrEmpty(value))
            return false;

        // Simple heuristic: check for replacement character
        return value.Contains('\uFFFD');
    }

    private static bool IsNumericColumn(DataFrameColumn column)
    {
        var dataType = column.DataType;
        return dataType == typeof(int) ||
               dataType == typeof(long) ||
               dataType == typeof(float) ||
               dataType == typeof(double) ||
               dataType == typeof(decimal);
    }
}
