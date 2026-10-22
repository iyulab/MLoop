using MLoop.Search;

namespace MLoop.Search.Tests;

public class MetricPolicyTests
{
    private sealed record Policy(string Name, bool HigherIsBetter) : IMetricPolicy;

    private static TrialResult Trial(params (string Name, double Value)[] metrics) => new()
    {
        Trainer = "T",
        Metrics = metrics.ToDictionary(m => m.Name, m => m.Value)
    };

    [Fact]
    public void ATrialIsScoredByThePolicysMetric()
    {
        Assert.True(new Policy("auc", true).TryScore(Trial(("auc", 0.8), ("accuracy", 0.7)), out var score));
        Assert.Equal(0.8, score);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ANonFiniteValue_CannotBeRanked(double value)
    {
        Assert.False(new Policy("auc", true).TryScore(Trial(("auc", value)), out _));
    }

    [Fact]
    public void AMissingMetricOrAFailedTrial_CannotBeRanked()
    {
        var policy = new Policy("auc", true);

        Assert.False(policy.TryScore(Trial(("accuracy", 0.9)), out _));
        Assert.False(policy.TryScore(new TrialResult { Trainer = "T", Error = "threw" }, out _));
    }

    [Fact]
    public void BetterFollowsTheDirection_AndATieIsNotBetter()
    {
        var higher = new Policy("auc", HigherIsBetter: true);
        var lower = new Policy("log_loss", HigherIsBetter: false);

        Assert.True(higher.IsBetter(0.9, 0.8));
        Assert.False(higher.IsBetter(0.8, 0.8));
        Assert.True(lower.IsBetter(0.2, 0.3));
        Assert.False(lower.IsBetter(0.3, 0.2));
    }
}
