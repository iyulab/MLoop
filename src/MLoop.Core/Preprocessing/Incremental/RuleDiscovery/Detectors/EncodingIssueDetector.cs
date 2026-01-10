using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;
using System.Text;
using System.Text.RegularExpressions;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;

/// <summary>
/// Detects character encoding issues like UTF-8/Latin-1 conflicts and mojibake.
/// Identifies corrupted text that needs encoding correction.
/// </summary>
public sealed partial class EncodingIssueDetector : IPatternDetector
{
    public PatternType PatternType => PatternType.EncodingIssue;

    // Common mojibake patterns (UTF-8 bytes interpreted as Windows-1252)
    [GeneratedRegex(@"[\xC3\xA2\x80\x9C\x93\x94\x98\x99]")]
    private static partial Regex MojibakeRegex();

    // Replacement character (U+FFFD)
    [GeneratedRegex(@"\uFFFD")]
    private static partial Regex ReplacementCharRegex();

    // Multiple consecutive non-ASCII characters (potential corruption)
    [GeneratedRegex(@"[^\x00-\x7F]{3,}")]
    private static partial Regex NonAsciiClusterRegex();

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

        int mojibakeCount = 0;
        int replacementCharCount = 0;
        int corruptionCount = 0;
        var examples = new List<string>();

        for (long i = 0; i < column.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var value = DetectorHelpers.GetStringValue(column, i);
            if (string.IsNullOrWhiteSpace(value))
                continue;

            bool hasIssue = false;

            // Check for mojibake patterns
            if (MojibakeRegex().IsMatch(value))
            {
                mojibakeCount++;
                hasIssue = true;
            }

            // Check for replacement characters
            if (ReplacementCharRegex().IsMatch(value))
            {
                replacementCharCount++;
                hasIssue = true;
            }

            // Check for suspicious non-ASCII clusters
            if (HasSuspiciousEncoding(value))
            {
                corruptionCount++;
                hasIssue = true;
            }

            if (hasIssue && examples.Count < 5)
            {
                examples.Add(TruncateForDisplay(value, 50));
            }
        }

        var totalIssues = mojibakeCount + replacementCharCount + corruptionCount;

        if (totalIssues == 0)
        {
            return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
        }

        var affectedPercentage = DetectorHelpers.CalculatePercentage(
            totalIssues,
            column.Length);

        var severity = DetectorHelpers.DetermineSeverity(affectedPercentage);

        var pattern = new DetectedPattern
        {
            Type = PatternType.EncodingIssue,
            ColumnName = columnName,
            Description = BuildDescription(mojibakeCount, replacementCharCount, corruptionCount),
            Severity = severity,
            Occurrences = totalIssues,
            TotalRows = (int)column.Length,
            Confidence = 0.85, // Encoding detection can be uncertain
            Examples = examples,
            SuggestedFix = DetermineSuggestedFix(severity)
        };

        patterns.Add(pattern);

        return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
    }

    public bool IsApplicable(DataFrameColumn column)
    {
        return ColumnTypeHelper.IsStringColumn(column);
    }

    /// <summary>
    /// Check if string has suspicious encoding patterns.
    /// </summary>
    private static bool HasSuspiciousEncoding(string value)
    {
        // Check for multiple consecutive non-ASCII characters
        // that look like encoding corruption
        if (NonAsciiClusterRegex().IsMatch(value))
        {
            // Additional check: if it's valid UTF-8 but looks corrupted
            var bytes = Encoding.UTF8.GetBytes(value);
            var asLatin1 = Encoding.Latin1.GetString(bytes);

            // If Latin1 interpretation looks more readable, likely encoding issue
            if (asLatin1.Length > 0 && IsMoreReadable(asLatin1, value))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Heuristic to check if one string is more readable than another.
    /// </summary>
    private static bool IsMoreReadable(string candidate, string original)
    {
        // Count ASCII letters
        var candidateAscii = candidate.Count(c => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
        var originalAscii = original.Count(c => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));

        // More ASCII letters suggests more readable
        return candidateAscii > originalAscii * 1.5;
    }

    private static string TruncateForDisplay(string value, int maxLength)
    {
        if (value.Length <= maxLength)
            return value;

        return value[..maxLength] + "...";
    }

    private static string BuildDescription(
        int mojibakeCount,
        int replacementCharCount,
        int corruptionCount)
    {
        var parts = new List<string>();

        if (mojibakeCount > 0)
            parts.Add($"{mojibakeCount} mojibake");

        if (replacementCharCount > 0)
            parts.Add($"{replacementCharCount} replacement chars");

        if (corruptionCount > 0)
            parts.Add($"{corruptionCount} encoding corruptions");

        return $"Character encoding issues: {string.Join(", ", parts)}";
    }

    private static string DetermineSuggestedFix(Severity severity)
    {
        return severity switch
        {
            Severity.Critical => "Re-import data with correct encoding (UTF-8 recommended)",
            Severity.High => "Detect source encoding and convert to UTF-8",
            Severity.Medium => "Try encoding detection and conversion, verify results",
            Severity.Low => "Document encoding and consider conversion if needed",
            _ => "Review encoding and decide based on data source"
        };
    }
}
