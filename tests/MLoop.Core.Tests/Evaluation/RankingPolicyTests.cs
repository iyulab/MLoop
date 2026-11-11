using MLoop.Core.Evaluation;
using MLoop.Search;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// The policy a search ranks by takes its direction from <see cref="MetricDirection"/> — the search
/// ranks the way every other MLoop ranking does, including the names that once ranked upside down.
/// </summary>
public class RankingPolicyTests
{
    [Theory]
    [InlineData("accuracy", true)]
    [InlineData("auc", true)]
    [InlineData("macro_accuracy", true)]
    [InlineData("r_squared", true)]
    [InlineData("log_loss_reduction", true)]
    [InlineData("log_loss", false)]
    [InlineData("rmse", false)]
    [InlineData("mae", false)]
    public void TheDirection_IsTheDirectionAuthoritys(string metric, bool higherIsBetter)
    {
        var policy = RankingPolicy.For(metric);

        Assert.Equal(metric, policy.Name);
        Assert.Equal(higherIsBetter, policy.HigherIsBetter);
        Assert.Equal(!MetricDirection.IsLowerBetter(metric), policy.HigherIsBetter);
    }

    // A loss a model drives toward zero (log loss, calibration error) has an absolute smallest difference worth reading; a metric on
    // another scale has none the policy can know.
    [Theory]
    [InlineData("log_loss", 0.05)]
    [InlineData("ece", 0.02)]
    [InlineData("macro_accuracy", 0.0)]
    [InlineData("rmse", 0.0)]
    public void ALossBoundedAtZeroHasAnAbsoluteSlack(string metric, double slack)
        => Assert.Equal(slack, RankingPolicy.For(metric).AbsoluteSlack);

    [Fact]
    public void ItRanksTrials()
    {
        IMetricPolicy policy = RankingPolicy.For("rmse");
        var trial = new TrialResult { Trainer = "T", Metrics = new Dictionary<string, double> { ["rmse"] = 1.5 } };

        Assert.True(policy.TryScore(trial, out var score));
        Assert.True(policy.IsBetter(1.0, score));
    }

    [Fact]
    public void ANameIsRequired()
    {
        Assert.Throws<ArgumentException>(() => RankingPolicy.For(""));
    }
}
