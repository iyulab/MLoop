using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.Statistics;

namespace MLoop.Core.Tests.Preprocessing.Incremental.Statistics;

public class NumericAnalyzerTests
{
    private static AnalysisConfiguration DefaultConfig => new();

    #region Basic Statistics Tests

    [Fact]
    public void Analyze_SimpleValues_CalculatesMean()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(3.0, stats.Mean);
    }

    [Fact]
    public void Analyze_SimpleValues_CalculatesMedian()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(3.0, stats.Median);
    }

    [Fact]
    public void Analyze_EvenCount_CalculatesMedianByInterpolation()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 1.0, 2.0, 3.0, 4.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(2.5, stats.Median);
    }

    [Fact]
    public void Analyze_SimpleValues_CalculatesMinMax()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 3.0, 1.0, 5.0, 2.0, 4.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(1.0, stats.Min);
        Assert.Equal(5.0, stats.Max);
        Assert.Equal(4.0, stats.Range);
    }

    [Fact]
    public void Analyze_SimpleValues_CalculatesSum()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 1.0, 2.0, 3.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(6.0, stats.Sum);
    }

    [Fact]
    public void Analyze_SimpleValues_CalculatesCount()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 1.0, 2.0, 3.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(3, stats.Count);
    }

    [Fact]
    public void Analyze_CalculatesVarianceAndStdDev()
    {
        // Values: 2, 4, 4, 4, 5, 5, 7, 9 → mean=5, sample variance=4.571
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(5.0, stats.Mean);
        Assert.Equal(stats.Variance, stats.StandardDeviation * stats.StandardDeviation, 10);
        Assert.True(stats.Variance > 0);
    }

    #endregion

    #region Quartile & Percentile Tests

    [Fact]
    public void Analyze_CalculatesQuartiles()
    {
        // 1,2,3,4,5,6,7,8,9,10 → Q1=3.25, Q3=7.75
        var values = Enumerable.Range(1, 10).Select(i => (double)i).ToArray();
        var col = new PrimitiveDataFrameColumn<double>("Num", values);

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.True(stats.Q1 > 0);
        Assert.True(stats.Q3 > stats.Q1);
        Assert.Equal(stats.Q3 - stats.Q1, stats.IQR);
    }

    [Fact]
    public void Analyze_CustomPercentiles_ReturnsRequestedPercentiles()
    {
        var values = Enumerable.Range(1, 100).Select(i => (double)i).ToArray();
        var col = new PrimitiveDataFrameColumn<double>("Num", values);
        var config = new AnalysisConfiguration
        {
            Percentiles = new[] { 0.1, 0.25, 0.5, 0.75, 0.9 }
        };

        var stats = NumericAnalyzer.Analyze(col, config);

        Assert.Equal(5, stats.Percentiles.Count);
        Assert.True(stats.Percentiles.ContainsKey(0.1));
        Assert.True(stats.Percentiles.ContainsKey(0.9));
        Assert.True(stats.Percentiles[0.1] < stats.Percentiles[0.9]);
    }

    #endregion

    #region Mode Tests

    [Fact]
    public void Analyze_RepeatedValues_FindsMode()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 1.0, 2.0, 2.0, 3.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(2.0, stats.Mode);
    }

    [Fact]
    public void Analyze_AllUniqueValues_ModeIsNull()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 1.0, 2.0, 3.0, 4.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Null(stats.Mode);
    }

    #endregion

    #region Outlier Detection Tests

    [Fact]
    public void Analyze_IQRMethod_DetectsOutliers()
    {
        // Normal range: 1-10, outlier: 100
        var values = new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0 };
        var col = new PrimitiveDataFrameColumn<double>("Num", values);
        var config = new AnalysisConfiguration { OutlierMethod = OutlierDetectionMethod.IQR };

        var stats = NumericAnalyzer.Analyze(col, config);

        Assert.True(stats.OutlierCount > 0);
        Assert.True(stats.OutlierPercentage > 0);
    }

    [Fact]
    public void Analyze_ZScoreMethod_DetectsOutliers()
    {
        // Normal range: ~0, outlier: 100
        var values = Enumerable.Repeat(5.0, 20).Append(100.0).ToArray();
        var col = new PrimitiveDataFrameColumn<double>("Num", values);
        var config = new AnalysisConfiguration { OutlierMethod = OutlierDetectionMethod.ZScore };

        var stats = NumericAnalyzer.Analyze(col, config);

        Assert.True(stats.OutlierCount > 0);
    }

    [Fact]
    public void Analyze_NoOutlierMethod_ZeroOutliers()
    {
        var values = new[] { 1.0, 2.0, 3.0, 100.0 };
        var col = new PrimitiveDataFrameColumn<double>("Num", values);
        var config = new AnalysisConfiguration { OutlierMethod = OutlierDetectionMethod.None };

        var stats = NumericAnalyzer.Analyze(col, config);

        Assert.Equal(0, stats.OutlierCount);
    }

    #endregion

    #region Advanced Statistics Tests

    [Fact]
    public void Analyze_AdvancedStatsEnabled_CalculatesSkewnessAndKurtosis()
    {
        var values = Enumerable.Range(1, 100).Select(i => (double)i).ToArray();
        var col = new PrimitiveDataFrameColumn<double>("Num", values);
        var config = new AnalysisConfiguration { CalculateAdvancedStats = true };

        var stats = NumericAnalyzer.Analyze(col, config);

        Assert.NotNull(stats.Skewness);
        Assert.NotNull(stats.Kurtosis);
    }

    [Fact]
    public void Analyze_AdvancedStatsDisabled_SkewnessAndKurtosisNull()
    {
        var values = Enumerable.Range(1, 100).Select(i => (double)i).ToArray();
        var col = new PrimitiveDataFrameColumn<double>("Num", values);
        var config = new AnalysisConfiguration { CalculateAdvancedStats = false };

        var stats = NumericAnalyzer.Analyze(col, config);

        Assert.Null(stats.Skewness);
        Assert.Null(stats.Kurtosis);
    }

    [Fact]
    public void Analyze_LessThan3Values_NoAdvancedStats()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 1.0, 2.0 });
        var config = new AnalysisConfiguration { CalculateAdvancedStats = true };

        var stats = NumericAnalyzer.Analyze(col, config);

        Assert.Null(stats.Skewness);
        Assert.Null(stats.Kurtosis);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Analyze_EmptyColumn_ReturnsZeroStats()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", Array.Empty<double>());

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(0, stats.Count);
        Assert.Equal(0, stats.Mean);
        Assert.Equal(0, stats.Sum);
    }

    [Fact]
    public void Analyze_AllNulls_ReturnsZeroStats()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", 5);
        // PrimitiveDataFrameColumn<double>(name, length) initializes with nulls

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(0, stats.Count);
    }

    [Fact]
    public void Analyze_NullsSkipped_CorrectCount()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new double?[] { 1.0, null, 3.0, null, 5.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(3, stats.Count);
        Assert.Equal(3.0, stats.Mean);
    }

    [Fact]
    public void Analyze_SingleValue_CorrectStats()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 42.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(1, stats.Count);
        Assert.Equal(42.0, stats.Mean);
        Assert.Equal(42.0, stats.Median);
        Assert.Equal(42.0, stats.Min);
        Assert.Equal(42.0, stats.Max);
        Assert.Equal(0.0, stats.Range);
        Assert.Equal(0.0, stats.Variance);
    }

    [Fact]
    public void Analyze_ConstantValues_ZeroVariance()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 5.0, 5.0, 5.0, 5.0 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(0.0, stats.Variance);
        Assert.Equal(0.0, stats.StandardDeviation);
        Assert.Equal(0.0, stats.IQR);
    }

    [Fact]
    public void Analyze_IntegerColumn_Works()
    {
        var col = new PrimitiveDataFrameColumn<int>("Num", new[] { 1, 2, 3, 4, 5 });

        var stats = NumericAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(5, stats.Count);
        Assert.Equal(3.0, stats.Mean);
    }

    [Fact]
    public void Analyze_NullColumn_ThrowsArgumentNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            NumericAnalyzer.Analyze(null!, DefaultConfig));
    }

    [Fact]
    public void Analyze_NullConfig_ThrowsArgumentNull()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 1.0 });

        Assert.Throws<ArgumentNullException>(() =>
            NumericAnalyzer.Analyze(col, null!));
    }

    #endregion
}
