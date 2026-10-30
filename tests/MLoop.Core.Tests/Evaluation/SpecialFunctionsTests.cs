using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

public class SpecialFunctionsTests
{
    [Fact]
    public void LogGamma_IsTheLogarithmOfTheFactorial()
    {
        Assert.Equal(Math.Log(24), SpecialFunctions.LogGamma(5), 12);
        Assert.Equal(Math.Log(Math.Sqrt(Math.PI)), SpecialFunctions.LogGamma(0.5), 12);
    }

    // Values of the χ² distribution's upper quantiles, from the standard tables.
    [Theory]
    [InlineData(1, 0.05, 3.841458820694124)]
    [InlineData(2, 0.05, 5.991464547107979)]
    [InlineData(5, 0.01, 15.08627246938899)]
    [InlineData(10, 0.05, 18.30703805327515)]
    [InlineData(10, 0.001, 29.58829844533946)]
    [InlineData(30, 0.05, 43.77297182574219)]
    public void ChiSquareUpperQuantile_MatchesTheTables(int degrees, double tail, double expected)
        => Assert.Equal(expected, SpecialFunctions.ChiSquareUpperQuantile(degrees, tail), 6);

    [Fact]
    public void UpperIncompleteGamma_IsTheSurvivalOfAChiSquare()
    {
        // The upper tail of χ²(2) at x is exp(−x ÷ 2), and Q(1, x ÷ 2) must say the same.
        foreach (var x in new[] { 0.5, 2.0, 7.0, 19.0 })
            Assert.Equal(Math.Exp(-x / 2), SpecialFunctions.UpperIncompleteGamma(1, x / 2), 12);
    }

    [Fact]
    public void ChiSquareUpperQuantile_ReachesTheFarTailsAWideTableAsksFor()
    {
        // Five in a hundred thousand, the Bonferroni share of one column of a thousand wide: a deep tail, and the quantile must be above the shallower ones.
        var deep = SpecialFunctions.ChiSquareUpperQuantile(10, 0.05 / 1000);
        Assert.True(deep > SpecialFunctions.ChiSquareUpperQuantile(10, 0.05 / 50));
        Assert.Equal(0.05 / 1000, SpecialFunctions.UpperIncompleteGamma(5, deep / 2), 12);
    }

    [Fact]
    public void ChiSquareUpperQuantile_RefusesNonsense()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SpecialFunctions.ChiSquareUpperQuantile(0, 0.05));
        Assert.Throws<ArgumentOutOfRangeException>(() => SpecialFunctions.ChiSquareUpperQuantile(3, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => SpecialFunctions.ChiSquareUpperQuantile(3, 1));
    }
}
