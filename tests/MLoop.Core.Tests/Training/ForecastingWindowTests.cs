using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// SSA's cost grows with the cube of its window and its forecast turns unstable when the window is longer than the
/// season needs, so a window that grows with the series makes a long series impossible to forecast. These pin the
/// default's shape: a quarter of the series, up to a ceiling.
/// </summary>
public class ForecastingWindowTests
{
    [Theory]
    [InlineData(240, 60)]      // the lifecycle walk's series: read as before
    [InlineData(1_000, 250)]
    [InlineData(1_024, 256)]   // the last length whose quarter is under the ceiling
    public void ASeriesWhoseQuarterIsUnderTheCeilingKeepsItsQuarter(int length, int expected)
        => Assert.Equal(expected, ForecastingWindow.Default(length));

    [Theory]
    [InlineData(1_025)]
    [InlineData(10_000)]
    [InlineData(100_000)]
    [InlineData(1_000_000)]
    [InlineData(int.MaxValue)]
    public void ALongSeriesIsGivenTheCeilingNotAQuarter(int length)
        => Assert.Equal(ForecastingWindow.MaximumDefault, ForecastingWindow.Default(length));

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    public void AShortSeriesGetsTheSmallestWindowSsaAccepts(int length)
        => Assert.Equal(ForecastingWindow.Minimum, ForecastingWindow.Default(length));

    [Fact]
    public void TheDefaultNeverShrinksAsTheSeriesGrows()
    {
        var previous = 0;
        for (var length = 0; length <= 5_000; length++)
        {
            var window = ForecastingWindow.Default(length);
            Assert.True(window >= previous, $"{length} points: window {window} < {previous}");
            Assert.InRange(window, ForecastingWindow.Minimum, ForecastingWindow.MaximumDefault);
            previous = window;
        }
    }

    [Theory]
    [InlineData(256, 10, 523)]
    [InlineData(2, 1, 6)]
    [InlineData(730, 10, 1471)]
    public void TheRowsAWindowNeedsAreTwiceTheWindowPlusTheHorizonPlusOne(int window, int horizon, int expected)
        => Assert.Equal(expected, ForecastingWindow.MinimumRows(window, horizon));

    [Theory]
    // a horizon longer than the series: no window can help, only a shorter horizon
    [InlineData(300, 75, 500, "a horizon of at most 149")]
    // too many rows for the horizon but a window that is too long
    [InlineData(40, 20, 10, "a window_size of at most 14")]
    // both could change
    [InlineData(100, 40, 30, "a horizon of at most 19 or a window_size of at most 34")]
    // eight points with the default horizon: the horizon is what to cut
    [InlineData(8, 2, 10, "a horizon of at most 3")]
    // nothing but both smaller helps
    [InlineData(20, 15, 18, "both smaller (even the smallest window, 2, needs a horizon of at most 15)")]
    // not even the smallest window and a horizon of one: only more points help
    [InlineData(4, 2, 10, null)]
    public void WhatToChangeWhenTheSeriesIsTooShortNamesOnlyWhatCanHelp(int rows, int window, int horizon, string? expected)
    {
        // "or a smaller window_size" was all the refusal said, to a series of 300 points with a horizon of 500.
        Assert.Equal(expected, ForecastingWindow.Remedy(rows, window, horizon));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(1_000)]
    [InlineData(10_000)]
    public void TheDefaultWindowAlwaysLeavesTheRowsItNeeds(int length)
    {
        // a quarter of the series (or the ceiling) is under half of what is left after a horizon of ten
        var window = ForecastingWindow.Default(length);
        if (length >= ForecastingWindow.MinimumRows(ForecastingWindow.Minimum, 10))
            Assert.True(length >= ForecastingWindow.MinimumRows(window, 10), $"a series of {length} rows with the default window {window}");
    }
}
