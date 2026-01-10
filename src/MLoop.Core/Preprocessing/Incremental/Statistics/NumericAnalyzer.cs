using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;

namespace MLoop.Core.Preprocessing.Incremental.Statistics;

/// <summary>
/// Analyzer for numeric columns providing comprehensive statistical metrics.
/// </summary>
public static class NumericAnalyzer
{
    /// <summary>
    /// Analyzes a numeric column and calculates comprehensive statistics.
    /// </summary>
    public static NumericStats Analyze(DataFrameColumn column, AnalysisConfiguration config)
    {
        ArgumentNullException.ThrowIfNull(column);
        ArgumentNullException.ThrowIfNull(config);

        // Extract numeric values (skip nulls)
        var values = ExtractNumericValues(column);

        if (values.Count == 0)
        {
            return CreateEmptyStats();
        }

        values.Sort();

        var mean = CalculateMean(values);
        var variance = CalculateVariance(values, mean);
        var stdDev = Math.Sqrt(variance);

        var q1 = CalculatePercentile(values, 0.25);
        var median = CalculatePercentile(values, 0.5);
        var q3 = CalculatePercentile(values, 0.75);

        // Calculate outliers
        int outlierCount = 0;
        if (config.OutlierMethod == OutlierDetectionMethod.IQR)
        {
            var iqr = q3 - q1;
            var lowerBound = q1 - config.IQRMultiplier * iqr;
            var upperBound = q3 + config.IQRMultiplier * iqr;
            outlierCount = values.Count(v => v < lowerBound || v > upperBound);
        }
        else if (config.OutlierMethod == OutlierDetectionMethod.ZScore)
        {
            outlierCount = values.Count(v => Math.Abs((v - mean) / stdDev) > config.ZScoreThreshold);
        }

        // Calculate advanced stats if enabled
        double? skewness = null;
        double? kurtosis = null;

        if (config.CalculateAdvancedStats && values.Count >= 3)
        {
            skewness = CalculateSkewness(values, mean, stdDev);
            kurtosis = CalculateKurtosis(values, mean, stdDev);
        }

        // Calculate additional percentiles
        var percentiles = new Dictionary<double, double>();
        foreach (var p in config.Percentiles)
        {
            percentiles[p] = CalculatePercentile(values, p);
        }

        return new NumericStats
        {
            Count = values.Count,
            Mean = mean,
            Median = median,
            Mode = TryCalculateMode(values),
            StandardDeviation = stdDev,
            Variance = variance,
            Min = values[0],
            Max = values[^1],
            Q1 = q1,
            Q3 = q3,
            Skewness = skewness,
            Kurtosis = kurtosis,
            OutlierCount = outlierCount,
            Sum = values.Sum(),
            Percentiles = percentiles
        };
    }

    /// <summary>
    /// Extracts numeric values from a DataFrame column, skipping nulls.
    /// </summary>
    private static List<double> ExtractNumericValues(DataFrameColumn column)
    {
        var values = new List<double>();

        for (int i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value != null)
            {
                values.Add(Convert.ToDouble(value));
            }
        }

        return values;
    }

    /// <summary>
    /// Calculates the arithmetic mean.
    /// </summary>
    private static double CalculateMean(List<double> values)
    {
        return values.Average();
    }

    /// <summary>
    /// Calculates the variance.
    /// </summary>
    private static double CalculateVariance(List<double> values, double mean)
    {
        if (values.Count < 2) return 0.0;

        return values.Sum(v => Math.Pow(v - mean, 2)) / (values.Count - 1);
    }

    /// <summary>
    /// Calculates a percentile using linear interpolation.
    /// </summary>
    private static double CalculatePercentile(List<double> sortedValues, double percentile)
    {
        if (sortedValues.Count == 0) return 0.0;
        if (sortedValues.Count == 1) return sortedValues[0];

        double position = percentile * (sortedValues.Count - 1);
        int lowerIndex = (int)Math.Floor(position);
        int upperIndex = (int)Math.Ceiling(position);

        if (lowerIndex == upperIndex)
        {
            return sortedValues[lowerIndex];
        }

        double lowerValue = sortedValues[lowerIndex];
        double upperValue = sortedValues[upperIndex];
        double fraction = position - lowerIndex;

        return lowerValue + (upperValue - lowerValue) * fraction;
    }

    /// <summary>
    /// Attempts to calculate the mode (most frequent value).
    /// </summary>
    private static double? TryCalculateMode(List<double> values)
    {
        if (values.Count == 0) return null;

        // For continuous data, mode might not be meaningful
        // Only return if there's a clear mode
        var groups = values.GroupBy(v => Math.Round(v, 2))
                          .OrderByDescending(g => g.Count())
                          .ToList();

        if (groups.Count > 0 && groups[0].Count() > 1)
        {
            return groups[0].Key;
        }

        return null;
    }

    /// <summary>
    /// Calculates skewness (measure of asymmetry).
    /// </summary>
    private static double CalculateSkewness(List<double> values, double mean, double stdDev)
    {
        if (stdDev == 0) return 0.0;

        var n = values.Count;
        var sum = values.Sum(v => Math.Pow((v - mean) / stdDev, 3));

        return (n / ((n - 1.0) * (n - 2.0))) * sum;
    }

    /// <summary>
    /// Calculates kurtosis (measure of tailedness).
    /// </summary>
    private static double CalculateKurtosis(List<double> values, double mean, double stdDev)
    {
        if (stdDev == 0) return 0.0;

        var n = values.Count;
        var sum = values.Sum(v => Math.Pow((v - mean) / stdDev, 4));

        var kurtosis = ((n * (n + 1)) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * sum;
        var adjustment = (3 * Math.Pow(n - 1, 2)) / ((n - 2.0) * (n - 3.0));

        return kurtosis - adjustment + 3; // Excess kurtosis + 3 for standard kurtosis
    }

    /// <summary>
    /// Creates empty stats for columns with no valid values.
    /// </summary>
    private static NumericStats CreateEmptyStats()
    {
        return new NumericStats
        {
            Count = 0,
            Mean = 0,
            Median = 0,
            StandardDeviation = 0,
            Variance = 0,
            Min = 0,
            Max = 0,
            Q1 = 0,
            Q3 = 0,
            Sum = 0
        };
    }
}
