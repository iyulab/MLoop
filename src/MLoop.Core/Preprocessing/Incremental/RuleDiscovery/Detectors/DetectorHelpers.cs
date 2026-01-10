using Microsoft.Data.Analysis;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;

/// <summary>
/// Helper utilities for pattern detectors.
/// </summary>
internal static class DetectorHelpers
{
    /// <summary>
    /// Safely get string value from DataFrameColumn, handling nulls.
    /// </summary>
    public static string? GetStringValue(DataFrameColumn column, long rowIndex)
    {
        var value = column[rowIndex];
        return value?.ToString();
    }

    /// <summary>
    /// Get all non-null string values from a column.
    /// </summary>
    public static IEnumerable<string> GetNonNullStrings(DataFrameColumn column)
    {
        for (long i = 0; i < column.Length; i++)
        {
            var value = GetStringValue(column, i);
            if (!string.IsNullOrEmpty(value))
            {
                yield return value;
            }
        }
    }

    /// <summary>
    /// Get random sample of non-null values for examples.
    /// </summary>
    public static List<string> GetRandomSample(DataFrameColumn column, int maxSamples = 5)
    {
        var nonNullValues = GetNonNullStrings(column).ToList();
        if (nonNullValues.Count <= maxSamples)
        {
            return nonNullValues;
        }

        var random = new Random(42); // Deterministic for consistency
        return nonNullValues
            .OrderBy(_ => random.Next())
            .Take(maxSamples)
            .ToList();
    }

    /// <summary>
    /// Count null or empty values in a column.
    /// </summary>
    public static int CountNullOrEmpty(DataFrameColumn column)
    {
        int count = 0;
        for (long i = 0; i < column.Length; i++)
        {
            var value = GetStringValue(column, i);
            if (string.IsNullOrEmpty(value))
            {
                count++;
            }
        }
        return count;
    }

    /// <summary>
    /// Check if a string value represents a missing value.
    /// </summary>
    public static bool IsMissingValue(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return true;

        var normalized = value.Trim().ToUpperInvariant();
        return normalized is "NULL" or "NA" or "N/A" or "NAN" or "NONE" or "-" or "?";
    }

    /// <summary>
    /// Calculate percentage (0-1) with safe division.
    /// </summary>
    public static double CalculatePercentage(int count, long total)
    {
        if (total == 0)
            return 0.0;

        return (double)count / total;
    }

    /// <summary>
    /// Determine severity based on affected percentage.
    /// </summary>
    public static Models.Severity DetermineSeverity(double affectedPercentage)
    {
        return affectedPercentage switch
        {
            >= 0.50 => Models.Severity.Critical,  // >50%
            >= 0.20 => Models.Severity.High,      // >20%
            >= 0.05 => Models.Severity.Medium,    // >5%
            _ => Models.Severity.Low              // 1-5%
        };
    }
}
