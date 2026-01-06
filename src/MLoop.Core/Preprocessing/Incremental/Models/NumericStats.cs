namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Statistical metrics for numeric columns.
/// </summary>
public sealed class NumericStats
{
    /// <summary>
    /// Gets the count of non-null values.
    /// </summary>
    public required long Count { get; init; }

    /// <summary>
    /// Gets the arithmetic mean (average).
    /// </summary>
    public required double Mean { get; init; }

    /// <summary>
    /// Gets the median (50th percentile).
    /// </summary>
    public required double Median { get; init; }

    /// <summary>
    /// Gets the mode (most frequent value), if applicable.
    /// </summary>
    public double? Mode { get; init; }

    /// <summary>
    /// Gets the standard deviation.
    /// </summary>
    public required double StandardDeviation { get; init; }

    /// <summary>
    /// Gets the variance.
    /// </summary>
    public required double Variance { get; init; }

    /// <summary>
    /// Gets the minimum value.
    /// </summary>
    public required double Min { get; init; }

    /// <summary>
    /// Gets the maximum value.
    /// </summary>
    public required double Max { get; init; }

    /// <summary>
    /// Gets the range (max - min).
    /// </summary>
    public double Range => Max - Min;

    /// <summary>
    /// Gets the first quartile (25th percentile).
    /// </summary>
    public required double Q1 { get; init; }

    /// <summary>
    /// Gets the third quartile (75th percentile).
    /// </summary>
    public required double Q3 { get; init; }

    /// <summary>
    /// Gets the interquartile range (Q3 - Q1).
    /// </summary>
    public double IQR => Q3 - Q1;

    /// <summary>
    /// Gets the skewness (measure of distribution asymmetry).
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    /// <item><description>Skewness = 0: Symmetric distribution</description></item>
    /// <item><description>Skewness > 0: Right-skewed (tail on right)</description></item>
    /// <item><description>Skewness &lt; 0: Left-skewed (tail on left)</description></item>
    /// </list>
    /// </remarks>
    public double? Skewness { get; init; }

    /// <summary>
    /// Gets the kurtosis (measure of distribution "tailedness").
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    /// <item><description>Kurtosis = 3: Normal distribution (mesokurtic)</description></item>
    /// <item><description>Kurtosis > 3: Heavy tails (leptokurtic)</description></item>
    /// <item><description>Kurtosis &lt; 3: Light tails (platykurtic)</description></item>
    /// </list>
    /// </remarks>
    public double? Kurtosis { get; init; }

    /// <summary>
    /// Gets the count of detected outliers.
    /// </summary>
    public int OutlierCount { get; init; }

    /// <summary>
    /// Gets the outlier percentage.
    /// </summary>
    public double OutlierPercentage => Count > 0 ? (double)OutlierCount / Count * 100.0 : 0.0;

    /// <summary>
    /// Gets the sum of all values.
    /// </summary>
    public required double Sum { get; init; }

    /// <summary>
    /// Gets additional percentiles (if calculated).
    /// </summary>
    public Dictionary<double, double> Percentiles { get; init; } = new();
}
