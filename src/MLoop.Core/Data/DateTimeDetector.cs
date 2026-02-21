using System.Globalization;

namespace MLoop.Core.Data;

/// <summary>
/// Shared heuristics for detecting DateTime columns.
/// Used by both CsvDataLoader (training data loading) and TrainingEngine (schema capture).
/// </summary>
public static class DateTimeDetector
{
    /// <summary>
    /// Checks if a column name is a strong DateTime indicator (no value check needed).
    /// Only exact/near-exact names like "datetime", "timestamp" qualify.
    /// </summary>
    public static bool IsStrongDateTimeName(string columnName)
    {
        var lowerName = columnName.ToLowerInvariant();
        return lowerName.Contains("datetime") || lowerName.Contains("timestamp");
    }

    /// <summary>
    /// Checks if a column name is a weak DateTime indicator (requires value confirmation).
    /// Names like "cycle_time", "spray_time" match but are often numeric durations.
    /// </summary>
    public static bool IsWeakDateTimeName(string columnName)
    {
        var lowerName = columnName.ToLowerInvariant();
        return lowerName == "date" || lowerName == "time" ||
               lowerName.EndsWith("_date") || lowerName.EndsWith("_time") ||
               lowerName.EndsWith("_dt") ||
               lowerName.StartsWith("date_") || lowerName.StartsWith("time_");
    }

    /// <summary>
    /// Checks if a column name matches any DateTime naming pattern (strong or weak).
    /// </summary>
    public static bool IsDateTimeColumnName(string columnName)
    {
        return IsStrongDateTimeName(columnName) || IsWeakDateTimeName(columnName);
    }

    /// <summary>
    /// Checks if sample values indicate a DateTime column (80% parse threshold).
    /// Values must contain typical DateTime separators (-, /, :, T) and be at least
    /// 6 characters to avoid false positives from short numeric values like "6.3"
    /// which DateTime.TryParse interprets as "June 3rd".
    /// </summary>
    public static bool IsDateTimeByValues(IEnumerable<string> sampleValues)
    {
        var parsedCount = 0;
        var nonEmptyCount = 0;

        foreach (var val in sampleValues)
        {
            if (string.IsNullOrWhiteSpace(val)) continue;
            nonEmptyCount++;

            // Reject values that are too short or lack DateTime separators
            // Real datetime strings look like: "2024-01-15", "01/15/2024", "09:30:00"
            // False positives look like: "6.3", "7.8", "2.134", "695.0"
            var trimmed = val.Trim();
            if (trimmed.Length < 6) continue;
            if (!trimmed.Contains('-') && !trimmed.Contains('/') &&
                !trimmed.Contains(':') && !trimmed.Contains('T'))
                continue;

            if (DateTime.TryParse(trimmed, CultureInfo.InvariantCulture,
                DateTimeStyles.None, out _))
            {
                parsedCount++;
            }
        }

        return nonEmptyCount > 0 && parsedCount >= nonEmptyCount * 0.8;
    }

    /// <summary>
    /// Detects if a column is DateTime using both name and value heuristics.
    /// Strong names (datetime, timestamp) are accepted without value check.
    /// Weak names (_time, _date) require value confirmation to avoid false positives
    /// on numeric duration/temperature columns like "Cycle_Time" or "Melting_Furnace_Temp".
    /// </summary>
    public static bool IsDateTimeColumn(string columnName, IEnumerable<string>? sampleValues = null)
    {
        // Strong name match: always DateTime (e.g., "Datetime", "created_timestamp")
        if (IsStrongDateTimeName(columnName))
            return true;

        bool hasValues = sampleValues != null;
        bool valuesAreDateTime = hasValues && IsDateTimeByValues(sampleValues!);

        // Weak name match: only if values confirm DateTime format
        if (IsWeakDateTimeName(columnName))
            return valuesAreDateTime;

        // No name match: rely on values alone
        return valuesAreDateTime;
    }
}
