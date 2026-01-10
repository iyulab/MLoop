using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;
using System.Globalization;
using System.Text.RegularExpressions;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;

/// <summary>
/// Detects format variations in date, number, and boolean representations.
/// Identifies inconsistent formatting that should be standardized.
/// </summary>
public sealed partial class FormatVariationDetector : IPatternDetector
{
    public PatternType PatternType => PatternType.FormatVariation;

    [GeneratedRegex(@"^\d{4}-\d{2}-\d{2}$")]
    private static partial Regex IsoDateRegex();

    [GeneratedRegex(@"^\d{1,2}/\d{1,2}/\d{2,4}$")]
    private static partial Regex UsDateRegex();

    [GeneratedRegex(@"^\d{1,2}\.\d{1,2}\.\d{2,4}$")]
    private static partial Regex EuDateRegex();

    public Task<IReadOnlyList<DetectedPattern>> DetectAsync(
        DataFrameColumn column,
        string columnName,
        CancellationToken cancellationToken = default)
    {
        var patterns = new List<DetectedPattern>();

        // Only applicable to string columns
        if (!ColumnTypeHelper.IsStringColumn(column))
        {
            return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
        }

        var inferredType = ColumnTypeHelper.InferColumnType(column);

        // Check for date format variations
        if (inferredType == ColumnTypeHelper.InferredType.DateTime)
        {
            var datePattern = DetectDateFormatVariations(column, columnName, cancellationToken);
            if (datePattern != null)
                patterns.Add(datePattern);
        }

        // Check for number format variations
        if (inferredType == ColumnTypeHelper.InferredType.Numeric)
        {
            var numberPattern = DetectNumberFormatVariations(column, columnName, cancellationToken);
            if (numberPattern != null)
                patterns.Add(numberPattern);
        }

        // Check for boolean format variations
        if (inferredType == ColumnTypeHelper.InferredType.Boolean)
        {
            var boolPattern = DetectBooleanFormatVariations(column, columnName, cancellationToken);
            if (boolPattern != null)
                patterns.Add(boolPattern);
        }

        return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
    }

    public bool IsApplicable(DataFrameColumn column)
    {
        return ColumnTypeHelper.IsStringColumn(column);
    }

    private DetectedPattern? DetectDateFormatVariations(
        DataFrameColumn column,
        string columnName,
        CancellationToken cancellationToken)
    {
        var formats = new Dictionary<string, int>
        {
            ["ISO-8601"] = 0,
            ["US"] = 0,
            ["EU"] = 0,
            ["Other"] = 0
        };

        var examples = new List<string>();

        for (long i = 0; i < column.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var value = DetectorHelpers.GetStringValue(column, i);
            if (string.IsNullOrWhiteSpace(value))
                continue;

            if (IsoDateRegex().IsMatch(value))
            {
                formats["ISO-8601"]++;
                if (examples.Count < 2)
                    examples.Add($"ISO: {value}");
            }
            else if (UsDateRegex().IsMatch(value))
            {
                formats["US"]++;
                if (examples.Count < 2)
                    examples.Add($"US: {value}");
            }
            else if (EuDateRegex().IsMatch(value))
            {
                formats["EU"]++;
                if (examples.Count < 2)
                    examples.Add($"EU: {value}");
            }
            else if (ColumnTypeHelper.TryParseDateTime(value, out _))
            {
                formats["Other"]++;
                if (examples.Count < 2)
                    examples.Add($"Other: {value}");
            }
        }

        // Multiple formats detected?
        var formatsWithValues = formats.Where(kvp => kvp.Value > 0).ToList();
        if (formatsWithValues.Count <= 1)
            return null;

        var totalDates = formats.Values.Sum();
        var affectedPercentage = DetectorHelpers.CalculatePercentage(
            totalDates - formatsWithValues.Max(kvp => kvp.Value),
            column.Length);

        return new DetectedPattern
        {
            Type = PatternType.FormatVariation,
            ColumnName = columnName,
            Description = $"Date format variations: {string.Join(", ", formatsWithValues.Select(f => $"{f.Key} ({f.Value})"))}",
            Severity = Severity.Medium,
            Occurrences = totalDates - formatsWithValues.Max(kvp => kvp.Value),
            TotalRows = (int)column.Length,
            Confidence = 0.90,
            Examples = examples,
            SuggestedFix = "Convert all dates to ISO-8601 (YYYY-MM-DD) format"
        };
    }

    private DetectedPattern? DetectNumberFormatVariations(
        DataFrameColumn column,
        string columnName,
        CancellationToken cancellationToken)
    {
        int withCommas = 0;
        int withSpaces = 0;
        int withDotDecimal = 0;
        int withCommaDecimal = 0;
        var examples = new List<string>();

        for (long i = 0; i < column.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var value = DetectorHelpers.GetStringValue(column, i);
            if (string.IsNullOrWhiteSpace(value))
                continue;

            if (value.Contains(',') && value.Contains('.'))
            {
                // European: 1.234,56 or US: 1,234.56
                if (value.LastIndexOf(',') > value.LastIndexOf('.'))
                {
                    withCommaDecimal++;
                    if (examples.Count < 3)
                        examples.Add($"Comma decimal: {value}");
                }
                else
                {
                    withCommas++;
                    if (examples.Count < 3)
                        examples.Add($"Comma separator: {value}");
                }
            }
            else if (value.Contains(','))
            {
                if (value.Count(c => c == ',') == 1 && value.IndexOf(',') > value.Length - 4)
                {
                    withCommaDecimal++;
                }
                else
                {
                    withCommas++;
                }
            }
            else if (value.Contains(' '))
            {
                withSpaces++;
                if (examples.Count < 3)
                    examples.Add($"Space separator: {value}");
            }
            else if (value.Contains('.'))
            {
                withDotDecimal++;
            }
        }

        // Multiple formats detected?
        var formatCounts = new[] { withCommas, withSpaces, withCommaDecimal }.Where(c => c > 0).Count();
        if (formatCounts <= 1)
            return null;

        var totalNumbers = withCommas + withSpaces + withDotDecimal + withCommaDecimal;
        var affectedPercentage = DetectorHelpers.CalculatePercentage(
            Math.Min(withCommas + withSpaces, withCommaDecimal),
            column.Length);

        return new DetectedPattern
        {
            Type = PatternType.FormatVariation,
            ColumnName = columnName,
            Description = $"Number format variations: {withCommas} comma-separated, {withSpaces} space-separated, {withCommaDecimal} comma-decimal",
            Severity = Severity.Low,
            Occurrences = Math.Min(withCommas + withSpaces, withCommaDecimal),
            TotalRows = (int)column.Length,
            Confidence = 0.85,
            Examples = examples,
            SuggestedFix = "Remove thousand separators, standardize decimal point to dot (.)"
        };
    }

    private DetectedPattern? DetectBooleanFormatVariations(
        DataFrameColumn column,
        string columnName,
        CancellationToken cancellationToken)
    {
        var representations = new Dictionary<string, int>();
        var examples = new List<string>();

        for (long i = 0; i < column.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var value = DetectorHelpers.GetStringValue(column, i);
            if (string.IsNullOrWhiteSpace(value))
                continue;

            if (ColumnTypeHelper.TryParseBoolean(value, out _))
            {
                var normalized = value.Trim().ToUpperInvariant();
                representations.TryGetValue(normalized, out var count);
                representations[normalized] = count + 1;

                if (examples.Count < 5 && !examples.Contains(value))
                {
                    examples.Add(value);
                }
            }
        }

        // Multiple representations detected?
        if (representations.Count <= 2) // true/false pair is normal
            return null;

        var totalBooleans = representations.Values.Sum();
        var affectedPercentage = DetectorHelpers.CalculatePercentage(
            totalBooleans - representations.Values.Max(),
            column.Length);

        return new DetectedPattern
        {
            Type = PatternType.FormatVariation,
            ColumnName = columnName,
            Description = $"Boolean format variations: {string.Join(", ", representations.Keys.Take(5))}",
            Severity = Severity.Low,
            Occurrences = totalBooleans - representations.Values.Max(),
            TotalRows = (int)column.Length,
            Confidence = 0.90,
            Examples = examples,
            SuggestedFix = "Convert all booleans to true/false or 1/0"
        };
    }
}
