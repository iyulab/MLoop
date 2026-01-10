using Microsoft.Data.Analysis;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;

/// <summary>
/// Statistical utilities for pattern detection.
/// </summary>
internal static class StatisticalHelper
{
    /// <summary>
    /// Calculate mean of numeric column values.
    /// </summary>
    public static double CalculateMean(DataFrameColumn column)
    {
        if (!ColumnTypeHelper.IsNumericColumn(column))
            throw new ArgumentException("Column must be numeric", nameof(column));

        double sum = 0;
        long count = 0;

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value != null)
            {
                sum += Convert.ToDouble(value);
                count++;
            }
        }

        return count > 0 ? sum / count : 0;
    }

    /// <summary>
    /// Calculate standard deviation of numeric column values.
    /// </summary>
    public static double CalculateStandardDeviation(DataFrameColumn column, double mean)
    {
        if (!ColumnTypeHelper.IsNumericColumn(column))
            throw new ArgumentException("Column must be numeric", nameof(column));

        double sumSquaredDiff = 0;
        long count = 0;

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value != null)
            {
                var diff = Convert.ToDouble(value) - mean;
                sumSquaredDiff += diff * diff;
                count++;
            }
        }

        return count > 1 ? Math.Sqrt(sumSquaredDiff / (count - 1)) : 0;
    }

    /// <summary>
    /// Calculate Z-score for a value.
    /// </summary>
    public static double CalculateZScore(double value, double mean, double stdDev)
    {
        if (stdDev == 0)
            return 0;

        return (value - mean) / stdDev;
    }

    /// <summary>
    /// Detect outliers using Z-score method (threshold: |z| > 3).
    /// </summary>
    public static List<OutlierInfo> DetectOutliersZScore(DataFrameColumn column, double threshold = 3.0)
    {
        if (!ColumnTypeHelper.IsNumericColumn(column))
            return new List<OutlierInfo>();

        var mean = CalculateMean(column);
        var stdDev = CalculateStandardDeviation(column, mean);

        if (stdDev == 0)
            return new List<OutlierInfo>(); // No variation, no outliers

        var outliers = new List<OutlierInfo>();

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value == null)
                continue;

            var numValue = Convert.ToDouble(value);
            var zScore = CalculateZScore(numValue, mean, stdDev);

            if (Math.Abs(zScore) > threshold)
            {
                outliers.Add(new OutlierInfo
                {
                    RowIndex = i,
                    Value = numValue,
                    ZScore = zScore
                });
            }
        }

        return outliers;
    }

    /// <summary>
    /// Calculate quartiles for IQR method.
    /// </summary>
    public static (double Q1, double Median, double Q3) CalculateQuartiles(DataFrameColumn column)
    {
        if (!ColumnTypeHelper.IsNumericColumn(column))
            throw new ArgumentException("Column must be numeric", nameof(column));

        var values = new List<double>();
        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value != null)
            {
                values.Add(Convert.ToDouble(value));
            }
        }

        if (values.Count == 0)
            return (0, 0, 0);

        values.Sort();

        var q1Index = (int)(values.Count * 0.25);
        var medianIndex = (int)(values.Count * 0.50);
        var q3Index = (int)(values.Count * 0.75);

        return (values[q1Index], values[medianIndex], values[q3Index]);
    }

    /// <summary>
    /// Detect outliers using IQR method.
    /// Outlier if: value &lt; Q1 - 1.5*IQR OR value &gt; Q3 + 1.5*IQR
    /// </summary>
    public static List<OutlierInfo> DetectOutliersIQR(DataFrameColumn column)
    {
        if (!ColumnTypeHelper.IsNumericColumn(column))
            return new List<OutlierInfo>();

        var (q1, _, q3) = CalculateQuartiles(column);
        var iqr = q3 - q1;

        if (iqr == 0)
            return new List<OutlierInfo>(); // No variation

        var lowerBound = q1 - 1.5 * iqr;
        var upperBound = q3 + 1.5 * iqr;

        var outliers = new List<OutlierInfo>();

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value == null)
                continue;

            var numValue = Convert.ToDouble(value);

            if (numValue < lowerBound || numValue > upperBound)
            {
                outliers.Add(new OutlierInfo
                {
                    RowIndex = i,
                    Value = numValue,
                    IsBelowLowerBound = numValue < lowerBound,
                    IsAboveUpperBound = numValue > upperBound
                });
            }
        }

        return outliers;
    }

    /// <summary>
    /// Information about a detected outlier.
    /// </summary>
    public sealed class OutlierInfo
    {
        public required long RowIndex { get; init; }
        public required double Value { get; init; }
        public double? ZScore { get; init; }
        public bool IsBelowLowerBound { get; init; }
        public bool IsAboveUpperBound { get; init; }
    }
}
