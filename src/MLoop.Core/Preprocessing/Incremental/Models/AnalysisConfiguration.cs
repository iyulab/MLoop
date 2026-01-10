namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Configuration for sample analysis behavior.
/// </summary>
public sealed class AnalysisConfiguration
{
    /// <summary>
    /// Gets or sets the maximum number of unique values to track for categorical columns.
    /// </summary>
    /// <remarks>
    /// Columns with more unique values than this threshold will have truncated value lists.
    /// Default is 100.
    /// </remarks>
    public int MaxCategoricalValues { get; set; } = 100;

    /// <summary>
    /// Gets or sets the percentile values to calculate for numeric columns.
    /// </summary>
    /// <remarks>
    /// Default percentiles: [0.25, 0.5, 0.75] (Q1, Q2/Median, Q3).
    /// Can be customized to include more percentiles (e.g., [0.1, 0.25, 0.5, 0.75, 0.9]).
    /// </remarks>
    public double[] Percentiles { get; set; } = [0.25, 0.5, 0.75];

    /// <summary>
    /// Gets or sets the outlier detection method.
    /// </summary>
    /// <remarks>
    /// <para>Options:</para>
    /// <list type="bullet">
    /// <item><description><b>IQR</b>: Interquartile Range method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)</description></item>
    /// <item><description><b>ZScore</b>: Z-score method (|z| > 3.0)</description></item>
    /// <item><description><b>None</b>: Disable outlier detection</description></item>
    /// </list>
    /// </remarks>
    public OutlierDetectionMethod OutlierMethod { get; set; } = OutlierDetectionMethod.IQR;

    /// <summary>
    /// Gets or sets the IQR multiplier for outlier detection.
    /// </summary>
    /// <remarks>
    /// Standard value is 1.5. Higher values (e.g., 3.0) for more conservative detection.
    /// </remarks>
    public double IQRMultiplier { get; set; } = 1.5;

    /// <summary>
    /// Gets or sets the Z-score threshold for outlier detection.
    /// </summary>
    /// <remarks>
    /// Standard value is 3.0. Higher values (e.g., 4.0) for more conservative detection.
    /// </remarks>
    public double ZScoreThreshold { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets whether to calculate advanced statistics (skewness, kurtosis).
    /// </summary>
    /// <remarks>
    /// Advanced statistics provide distribution shape information but add computation cost.
    /// Default is true.
    /// </remarks>
    public bool CalculateAdvancedStats { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to detect missing value patterns.
    /// </summary>
    /// <remarks>
    /// Pattern detection identifies systematic missing data (e.g., missing at random, not random).
    /// Default is true.
    /// </remarks>
    public bool DetectMissingPatterns { get; set; } = true;

    /// <summary>
    /// Gets or sets the sample size for histogram calculation.
    /// </summary>
    /// <remarks>
    /// Histograms are sampled for performance. Default is 10,000 values.
    /// </remarks>
    public int HistogramSampleSize { get; set; } = 10_000;

    /// <summary>
    /// Gets or sets the number of histogram bins.
    /// </summary>
    /// <remarks>
    /// Default is 20 bins. More bins provide finer granularity but may be noisy.
    /// </remarks>
    public int HistogramBins { get; set; } = 20;
}

/// <summary>
/// Outlier detection methods.
/// </summary>
public enum OutlierDetectionMethod
{
    /// <summary>
    /// No outlier detection.
    /// </summary>
    None,

    /// <summary>
    /// Interquartile Range (IQR) method.
    /// </summary>
    /// <remarks>
    /// Outliers: values less than Q1 - 1.5*IQR or greater than Q3 + 1.5*IQR
    /// </remarks>
    IQR,

    /// <summary>
    /// Z-score method.
    /// </summary>
    /// <remarks>
    /// Outliers: |z-score| > threshold (default 3.0)
    /// </remarks>
    ZScore
}
