using Microsoft.ML;
using MLoop.Core.Training;
using MLoop.Search;
using MLoop.Search.Validation;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// <see cref="EnsembleBudget"/> — the members of a blend are fitted only when their estimated fit times, summed, fit in the time left.
/// </summary>
public sealed class EnsembleBudgetTests
{
    private sealed class Scored
    {
        public bool Y { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }

    private static RetainedTrial Member(double fitSeconds)
    {
        var ml = new MLContext(seed: 1);
        var view = ml.Data.LoadFromEnumerable(new[] { new Scored { Y = true, Score = 1, Probability = 0.9f }, new Scored { Y = false, Score = -1, Probability = 0.1f } });
        return new RetainedTrial(new TrialResult { Trainer = "T" }, OutOfFoldScores.Read(view, "Y", null)!, _ => throw new NotSupportedException(), fitSeconds);
    }

    [Fact]
    public void Members_FitWhenTheirSummedEstimateIsWithinWhatIsLeft()
    {
        Assert.Null(EnsembleBudget.Shortfall([Member(10), Member(20)], TimeSpan.FromSeconds(30)));
        Assert.Null(EnsembleBudget.Shortfall([Member(10), Member(20)], TimeSpan.FromSeconds(31)));
    }

    [Fact]
    public void Members_DoNotFitWhenTheSumIsMoreThanWhatIsLeft_AndTheReasonSaysBothNumbers()
    {
        var why = EnsembleBudget.Shortfall([Member(10), Member(20)], TimeSpan.FromSeconds(29));

        Assert.NotNull(why);
        Assert.Contains("2 members", why, StringComparison.Ordinal);
        Assert.Contains("30 s", why, StringComparison.Ordinal);
        Assert.Contains("29 s", why, StringComparison.Ordinal);
    }

    [Fact]
    public void NoTimeLeftAtAll_IsReportedAsZero_NotNegative()
    {
        var why = EnsembleBudget.Shortfall([Member(5)], TimeSpan.FromSeconds(-12));

        Assert.NotNull(why);
        Assert.Contains("0 s of the time budget", why, StringComparison.Ordinal);
    }

    [Fact]
    public void TheSearchIsGivenOnlyAShareOfTheBudget() => Assert.InRange(EnsembleBudget.SearchShare, 0.5, 0.9);
}
