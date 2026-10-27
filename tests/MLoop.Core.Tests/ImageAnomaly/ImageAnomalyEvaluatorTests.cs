using MLoop.Core.ImageAnomaly;

namespace MLoop.Core.Tests.ImageAnomaly;

public class ImageAnomalyEvaluatorTests
{
    private static readonly double[] TwentyNormals = [.. Enumerable.Range(1, 20).Select(i => i / 10.0)];   // 0.1 .. 2.0

    // Counted by hand: 1.95 beats the 19 normals up to 1.9; 0.5 beats 0.1 .. 0.4 and ties 0.5 for half → 23.5 of 40.
    [Fact]
    public void TheAuroc_CountsPairs_TiesHalf()
    {
        Assert.Equal(23.5 / 40, ImageAnomalyEvaluator.Auroc(TwentyNormals, [1.95, 0.5]), 9);
        Assert.Equal(1.0, ImageAnomalyEvaluator.Auroc([0.1, 0.2], [0.3, 0.4]), 9);
        Assert.Equal(0.0, ImageAnomalyEvaluator.Auroc([0.3, 0.4], [0.1, 0.2]), 9);
        Assert.Equal(0.5, ImageAnomalyEvaluator.Auroc([0.2, 0.2], [0.2]), 9);
    }

    // Of 20 normal images 5 % is one false alarm: the threshold is the second highest (1.9); 10 % is two: the third (1.8).
    [Fact]
    public void TheMissedAnomalies_AreThoseAtOrBelowTheThresholdAFalseAlarmRateAllows()
    {
        Assert.Equal(1, ImageAnomalyEvaluator.Missed(TwentyNormals, [1.95, 0.5], 0.05));
        Assert.Equal(2, ImageAnomalyEvaluator.Missed(TwentyNormals, [1.85, 0.5], 0.05));
        Assert.Equal(1, ImageAnomalyEvaluator.Missed(TwentyNormals, [1.85, 0.5], 0.10));
        // a rate below one image allows none: the threshold is the highest normal score
        Assert.Equal(1, ImageAnomalyEvaluator.Missed([0.1, 0.2, 0.3], [0.25, 0.9], 0.05));
    }

    [Fact]
    public void TheQuality_CarriesBothCountsAndAsksForBothKindsOfImage()
    {
        var quality = ImageAnomalyEvaluator.Evaluate(TwentyNormals, [1.95, 0.5]);

        Assert.Equal((20, 2, 1, 1), (quality.Normals, quality.Anomalies, quality.MissedAt5, quality.MissedAt10));
        Assert.Equal(5, quality.AsMetrics().Count);
        Assert.Throws<ArgumentException>(() => ImageAnomalyEvaluator.Evaluate([], [1.0]));
        Assert.Throws<ArgumentException>(() => ImageAnomalyEvaluator.Evaluate([1.0], []));
    }
}
