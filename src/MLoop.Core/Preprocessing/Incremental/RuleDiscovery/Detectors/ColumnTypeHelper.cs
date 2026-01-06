using Microsoft.Data.Analysis;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;

/// <summary>
/// Utilities for detecting and working with column data types.
/// </summary>
internal static class ColumnTypeHelper
{
    /// <summary>
    /// Check if column is numeric type.
    /// </summary>
    public static bool IsNumericColumn(DataFrameColumn column)
    {
        var dataType = column.DataType;
        return dataType == typeof(int) ||
               dataType == typeof(long) ||
               dataType == typeof(float) ||
               dataType == typeof(double) ||
               dataType == typeof(decimal) ||
               dataType == typeof(short) ||
               dataType == typeof(byte);
    }

    /// <summary>
    /// Check if column is string type.
    /// </summary>
    public static bool IsStringColumn(DataFrameColumn column)
    {
        return column.DataType == typeof(string);
    }

    /// <summary>
    /// Check if column is DateTime type.
    /// </summary>
    public static bool IsDateTimeColumn(DataFrameColumn column)
    {
        return column.DataType == typeof(DateTime);
    }

    /// <summary>
    /// Check if column is boolean type.
    /// </summary>
    public static bool IsBooleanColumn(DataFrameColumn column)
    {
        return column.DataType == typeof(bool);
    }

    /// <summary>
    /// Try to parse a string as a numeric value.
    /// </summary>
    public static bool TryParseNumeric(string value, out double result)
    {
        // Remove common thousand separators
        var cleaned = value.Replace(",", "").Replace(" ", "").Trim();
        return double.TryParse(cleaned, out result);
    }

    /// <summary>
    /// Try to parse a string as a DateTime.
    /// </summary>
    public static bool TryParseDateTime(string value, out DateTime result)
    {
        return DateTime.TryParse(value, out result);
    }

    /// <summary>
    /// Try to parse a string as a boolean.
    /// </summary>
    public static bool TryParseBoolean(string value, out bool result)
    {
        var normalized = value.Trim().ToUpperInvariant();

        if (normalized is "TRUE" or "YES" or "Y" or "1" or "ON")
        {
            result = true;
            return true;
        }

        if (normalized is "FALSE" or "NO" or "N" or "0" or "OFF")
        {
            result = false;
            return true;
        }

        result = false;
        return false;
    }

    /// <summary>
    /// Detect inferred type from string column values.
    /// Samples up to 100 values for performance.
    /// </summary>
    public static InferredType InferColumnType(DataFrameColumn column)
    {
        if (!IsStringColumn(column))
        {
            return InferredType.TypedColumn;
        }

        int numericCount = 0;
        int dateTimeCount = 0;
        int booleanCount = 0;
        int totalNonNull = 0;

        var sampleSize = Math.Min(100, (int)column.Length);
        var step = column.Length / sampleSize;

        for (long i = 0; i < column.Length; i += step)
        {
            var value = DetectorHelpers.GetStringValue(column, i);
            if (string.IsNullOrWhiteSpace(value) || DetectorHelpers.IsMissingValue(value))
                continue;

            totalNonNull++;

            if (TryParseNumeric(value, out _))
                numericCount++;
            else if (TryParseDateTime(value, out _))
                dateTimeCount++;
            else if (TryParseBoolean(value, out _))
                booleanCount++;
        }

        if (totalNonNull == 0)
            return InferredType.Unknown;

        var numericRatio = (double)numericCount / totalNonNull;
        var dateTimeRatio = (double)dateTimeCount / totalNonNull;
        var booleanRatio = (double)booleanCount / totalNonNull;

        // Majority rule: >70% of values match a type
        if (numericRatio > 0.7)
            return InferredType.Numeric;
        if (dateTimeRatio > 0.7)
            return InferredType.DateTime;
        if (booleanRatio > 0.7)
            return InferredType.Boolean;

        // Mixed type if no clear majority
        if (numericRatio > 0.3 || dateTimeRatio > 0.3)
            return InferredType.Mixed;

        return InferredType.Text;
    }

    /// <summary>
    /// Inferred column type categories.
    /// </summary>
    public enum InferredType
    {
        Unknown,
        TypedColumn,  // Already has a non-string type
        Numeric,
        DateTime,
        Boolean,
        Text,
        Mixed         // Contains multiple types
    }
}
