using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;

/// <summary>
/// Detects whitespace issues like leading/trailing spaces, multiple spaces, and tabs.
/// </summary>
public sealed class WhitespaceDetector : IPatternDetector
{
    public PatternType PatternType => PatternType.WhitespaceIssue;

    public Task<IReadOnlyList<DetectedPattern>> DetectAsync(
        DataFrameColumn column,
        string columnName,
        CancellationToken cancellationToken = default)
    {
        var patterns = new List<DetectedPattern>();

        if (!ColumnTypeHelper.IsStringColumn(column))
        {
            return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
        }

        int leadingSpaces = 0;
        int trailingSpaces = 0;
        int multipleSpaces = 0;
        int tabsOrNewlines = 0;
        var examples = new List<string>();

        for (long i = 0; i < column.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var value = DetectorHelpers.GetStringValue(column, i);
            if (string.IsNullOrWhiteSpace(value))
                continue;

            bool hasIssue = false;

            // Check for leading spaces
            if (value.Length > 0 && char.IsWhiteSpace(value[0]))
            {
                leadingSpaces++;
                hasIssue = true;
            }

            // Check for trailing spaces
            if (value.Length > 0 && char.IsWhiteSpace(value[^1]))
            {
                trailingSpaces++;
                hasIssue = true;
            }

            // Check for multiple consecutive spaces
            if (value.Contains("  ")) // Two or more spaces
            {
                multipleSpaces++;
                hasIssue = true;
            }

            // Check for tabs or newlines
            if (value.Contains('\t') || value.Contains('\n') || value.Contains('\r'))
            {
                tabsOrNewlines++;
                hasIssue = true;
            }

            if (hasIssue && examples.Count < 5)
            {
                examples.Add($"\"{value}\""); // Quote to show spaces
            }
        }

        var totalIssues = leadingSpaces + trailingSpaces + multipleSpaces + tabsOrNewlines;

        if (totalIssues == 0)
        {
            return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
        }

        var affectedPercentage = DetectorHelpers.CalculatePercentage(
            totalIssues,
            column.Length);

        // Whitespace issues are usually low severity unless widespread
        var severity = affectedPercentage switch
        {
            >= 0.50 => Severity.Medium,
            >= 0.20 => Severity.Low,
            _ => Severity.Low
        };

        var pattern = new DetectedPattern
        {
            Type = PatternType.WhitespaceIssue,
            ColumnName = columnName,
            Description = BuildDescription(leadingSpaces, trailingSpaces, multipleSpaces, tabsOrNewlines),
            Severity = severity,
            Occurrences = totalIssues,
            TotalRows = (int)column.Length,
            Confidence = 1.0, // Deterministic detection
            Examples = examples,
            SuggestedFix = "Trim leading/trailing spaces, collapse multiple spaces to single space"
        };

        patterns.Add(pattern);

        return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
    }

    public bool IsApplicable(DataFrameColumn column)
    {
        return ColumnTypeHelper.IsStringColumn(column);
    }

    private static string BuildDescription(
        int leadingSpaces,
        int trailingSpaces,
        int multipleSpaces,
        int tabsOrNewlines)
    {
        var parts = new List<string>();

        if (leadingSpaces > 0)
            parts.Add($"{leadingSpaces} leading spaces");

        if (trailingSpaces > 0)
            parts.Add($"{trailingSpaces} trailing spaces");

        if (multipleSpaces > 0)
            parts.Add($"{multipleSpaces} multiple spaces");

        if (tabsOrNewlines > 0)
            parts.Add($"{tabsOrNewlines} tabs/newlines");

        return $"Whitespace issues: {string.Join(", ", parts)}";
    }
}
