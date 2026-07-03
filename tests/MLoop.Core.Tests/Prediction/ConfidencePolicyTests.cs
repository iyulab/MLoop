using MLoop.Core.Prediction;

namespace MLoop.Core.Tests.Prediction;

public class ConfidencePolicyTests
{
    [Fact]
    public void Classification_MaxClassProbability()
    {
        var row = new PredictionRow
        {
            Probabilities = new() { ["A"] = 0.7, ["B"] = 0.2, ["C"] = 0.1 },
        };
        Assert.Equal(0.7, ConfidencePolicy.Compute(row, "multiclass-classification")!.Value, 6);
    }

    [Fact]
    public void Binary_ConfidentNegative_MaxOverBothClasses()
    {
        // A confident "False" (P(True)=0.02) must read as high-confidence via the max, not low.
        var row = new PredictionRow
        {
            Probabilities = new() { ["True"] = 0.02, ["False"] = 0.98 },
            Score = 0.02,
        };
        Assert.Equal(0.98, ConfidencePolicy.Compute(row, "binary-classification")!.Value, 6);
    }

    [Theory]
    // Anomaly confidence = distance from the 0.5 decision boundary: |score-0.5|*2.
    [InlineData(0.986, 0.972)]
    [InlineData(0.197, 0.606)]
    [InlineData(0.5, 0.0)]
    [InlineData(1.0, 1.0)]
    public void Anomaly_BoundaryDistance(double anomalyScore, double expected)
    {
        var row = new PredictionRow { IsAnomaly = true, AnomalyScore = anomalyScore };
        Assert.Equal(expected, ConfidencePolicy.Compute(row, "anomaly-detection")!.Value, 3);
    }

    [Theory]
    // Regression band normalized by residual_std (k=3): 1 - half/(3*std). std=2 -> denom=6.
    [InlineData(14.0, 16.0, 0.8333)]  // narrow (half=1) -> high
    [InlineData(12.0, 18.0, 0.5000)]  // moderate (half=3)
    [InlineData(8.0, 22.0, 0.0000)]   // wide (half=7) -> clamped to 0
    public void Regression_BandNormalizedByResidualStd(double lower, double upper, double expected)
    {
        var row = new PredictionRow { Score = 15.0, ScoreLowerBound = lower, ScoreUpperBound = upper };
        Assert.Equal(expected, ConfidencePolicy.Compute(row, "regression", residualStd: 2.0)!.Value, 3);
    }

    [Fact]
    public void Regression_NarrowerBand_IsMoreConfident()
    {
        var narrow = new PredictionRow { Score = 15.0, ScoreLowerBound = 14.0, ScoreUpperBound = 16.0 };
        var wide = new PredictionRow { Score = 15.0, ScoreLowerBound = 10.0, ScoreUpperBound = 20.0 };
        Assert.True(ConfidencePolicy.Compute(narrow, "regression", 3.0) >
                    ConfidencePolicy.Compute(wide, "regression", 3.0));
    }

    [Fact]
    public void Regression_BandWithoutResidualStd_FallsBackToScoreRelative()
    {
        // No residual scale (old model): 1 - min(half/|Score|, 1). half=3, |Score|=15 -> 0.8.
        var row = new PredictionRow { Score = 15.0, ScoreLowerBound = 12.0, ScoreUpperBound = 18.0 };
        Assert.Equal(0.8, ConfidencePolicy.Compute(row, "regression", residualStd: null)!.Value, 3);
    }

    [Fact]
    public void Regression_NoBand_ReturnsNull()
    {
        // A bare regression point estimate carries no confidence signal (Score is a target value).
        var row = new PredictionRow { Score = 12.5 };
        Assert.Null(ConfidencePolicy.Compute(row, "regression"));
    }

    [Fact]
    public void RankingScalarScore_ClampedScore()
    {
        var row = new PredictionRow { Score = 0.42 };
        Assert.Equal(0.42, ConfidencePolicy.Compute(row, "ranking")!.Value, 6);
    }

    [Fact]
    public void Probabilities_TakePrecedenceOverScore()
    {
        var row = new PredictionRow
        {
            Probabilities = new() { ["A"] = 0.9, ["B"] = 0.1 },
            Score = 0.3,
        };
        Assert.Equal(0.9, ConfidencePolicy.Compute(row, "binary-classification")!.Value, 6);
    }

    [Fact]
    public void NonFiniteOrOutOfRange_ClampedToUnitInterval()
    {
        var nan = new PredictionRow { Probabilities = new() { ["A"] = double.NaN } };
        Assert.Equal(0.0, ConfidencePolicy.Compute(nan, "binary-classification")!.Value, 6);

        var over = new PredictionRow { Score = 1.7 };
        Assert.Equal(1.0, ConfidencePolicy.Compute(over, "ranking")!.Value, 6);
    }

    [Fact]
    public void NoSignal_ReturnsNull()
    {
        Assert.Null(ConfidencePolicy.Compute(new PredictionRow { ClusterId = 3 }, "clustering"));
    }
}
