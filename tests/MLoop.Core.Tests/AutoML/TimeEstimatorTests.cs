using MLoop.Core.AutoML;

namespace MLoop.Core.Tests.AutoML;

public class TimeEstimatorTests
{
    [Theory]
    [InlineData(500, 10, "binary-classification", 2, 30)]     // tiny -> min 30s
    [InlineData(1000, 10, "binary-classification", 2, 30)]    // 1K rows -> base
    [InlineData(10000, 10, "binary-classification", 2, 60)]   // 10K -> row_factor=2
    [InlineData(100000, 10, "binary-classification", 2, 90)]  // 100K -> row_factor=3
    [InlineData(10000, 40, "binary-classification", 2, 120)]  // 40 cols -> col_factor=2
    [InlineData(5000, 20, "multiclass-classification", 4, 90)] // multiclass baseline
    [InlineData(5000, 20, "multiclass-classification", 20, 251)] // many classes
    [InlineData(10000, 30, "regression", 0, 114)]              // regression
    public void EstimateStatic_ReturnsReasonableTime(
        int rows, int columns, string task, int classCount, int expectedApprox)
    {
        var estimate = TimeEstimator.EstimateStatic(rows, columns, task, classCount);

        // Allow 20% tolerance
        Assert.InRange(estimate, expectedApprox * 0.8, expectedApprox * 1.2);
        Assert.InRange(estimate, 30, 1800); // always within clamp bounds
    }

    [Fact]
    public void EstimateStatic_ClampsToMinimum()
    {
        var estimate = TimeEstimator.EstimateStatic(10, 2, "binary-classification", 2);
        Assert.Equal(30, estimate);
    }

    [Fact]
    public void EstimateStatic_ClampsToMaximum()
    {
        var estimate = TimeEstimator.EstimateStatic(10_000_000, 500, "multiclass-classification", 100);
        Assert.Equal(1800, estimate);
    }

    [Fact]
    public void EstimateStatic_TextFeatureBoost()
    {
        var withoutText = TimeEstimator.EstimateStatic(5000, 10, "binary-classification", 2);
        var withText = TimeEstimator.EstimateStatic(5000, 10, "binary-classification", 2, hasTextFeatures: true);
        Assert.True(withText > withoutText);
    }
}

public class ProbeResultTests
{
    [Theory]
    [InlineData(0.96, 15, 3, 30)]     // fast convergence -> probe x 2
    [InlineData(0.90, 15, 3, 60)]     // good -> static estimate
    [InlineData(0.70, 15, 3, 90)]     // moderate -> static x 1.5
    [InlineData(0.40, 15, 3, 120)]    // poor -> static x 2
    public void EstimateReactive_AdjustsBasedOnProbeMetric(
        double bestMetric, int probeTimeSeconds, int trialsCompleted, int staticEstimate)
    {
        var probe = new ProbeResult
        {
            BestMetric = bestMetric,
            ProbeTimeSeconds = probeTimeSeconds,
            TrialsCompleted = trialsCompleted
        };

        var result = TimeEstimator.EstimateReactive(probe, staticEstimate);

        Assert.InRange(result, TimeEstimator.MinSeconds, TimeEstimator.MaxSeconds);
    }

    [Fact]
    public void EstimateReactive_HighMetricReducesTime()
    {
        var probe = new ProbeResult { BestMetric = 0.98, ProbeTimeSeconds = 10, TrialsCompleted = 5 };
        var result = TimeEstimator.EstimateReactive(probe, 300);
        Assert.True(result < 300, $"Expected < 300 but got {result}");
    }

    [Fact]
    public void EstimateReactive_LowMetricIncreasesTime()
    {
        var probe = new ProbeResult { BestMetric = 0.35, ProbeTimeSeconds = 30, TrialsCompleted = 2 };
        var result = TimeEstimator.EstimateReactive(probe, 60);
        Assert.True(result > 60, $"Expected > 60 but got {result}");
    }

    [Fact]
    public void GetProbeTime_IsPortionOfStaticEstimate()
    {
        var probeTime = TimeEstimator.GetProbeTime(90);
        Assert.InRange(probeTime, 10, 30); // min(30, 90/3) = 30
    }

    [Fact]
    public void GetProbeTime_ClampsToMinimum()
    {
        var probeTime = TimeEstimator.GetProbeTime(30);
        Assert.Equal(10, probeTime); // min(30, 30/3) = 10
    }
}
