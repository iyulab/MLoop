using System.Globalization;

namespace MLoop.Core.Data;

/// <summary>
/// Shared heuristics for detecting DateTime columns.
/// Used by both CsvDataLoader (training data loading) and TrainingEngine (schema capture).
/// </summary>
public static class DateTimeDetector
{
    /// <summary>
    /// Checks if a column name matches common DateTime naming patterns.
    /// </summary>
    public static bool IsDateTimeColumnName(string columnName)
    {
        var lowerName = columnName.ToLowerInvariant();
        return lowerName.Contains("datetime") || lowerName.Contains("timestamp") ||
               lowerName == "date" || lowerName == "time" ||
               lowerName.EndsWith("_date") || lowerName.EndsWith("_time") ||
               lowerName.StartsWith("date_") || lowerName.StartsWith("time_");
    }

    /// <summary>
    /// Checks if sample values indicate a DateTime column (80% parse threshold).
    /// </summary>
    public static bool IsDateTimeByValues(IEnumerable<string> sampleValues)
    {
        var parsedCount = 0;
        var nonEmptyCount = 0;

        foreach (var val in sampleValues)
        {
            if (string.IsNullOrWhiteSpace(val)) continue;
            nonEmptyCount++;
            if (DateTime.TryParse(val, CultureInfo.InvariantCulture,
                DateTimeStyles.None, out _))
            {
                parsedCount++;
            }
        }

        return nonEmptyCount > 0 && parsedCount >= nonEmptyCount * 0.8;
    }

    /// <summary>
    /// Detects if a column is DateTime using both name and value heuristics.
    /// </summary>
    public static bool IsDateTimeColumn(string columnName, IEnumerable<string>? sampleValues = null)
    {
        if (IsDateTimeColumnName(columnName))
            return true;

        if (sampleValues != null && IsDateTimeByValues(sampleValues))
            return true;

        return false;
    }
}
