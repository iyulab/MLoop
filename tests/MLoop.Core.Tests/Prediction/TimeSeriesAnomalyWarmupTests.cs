using MLoop.Core.Prediction;

namespace MLoop.Core.Tests.Prediction;

/// <summary>The rows a series detector had no window of earlier points for say they were not judged, rather than "normal".</summary>
public sealed class TimeSeriesAnomalyWarmupTests
{
    private static PredictionRow Row(bool alert, double score) => new() { IsAnomaly = alert, AnomalyScore = score };

    [Fact]
    public void TheLeadingRowsWithAScoreOfZero_CarryNoVerdict_AndOneWarningSaysHowMany()
    {
        var (rows, warning) = TimeSeriesAnomalyWarmup.Mark([Row(false, 0), Row(false, 0), Row(false, 0.03), Row(true, 0.9), Row(false, 0.02)]);

        Assert.All(rows.Take(2), r => { Assert.Null(r.IsAnomaly); Assert.Null(r.AnomalyScore); });
        Assert.False(rows[2].IsAnomaly);
        Assert.True(rows[3].IsAnomaly);
        Assert.Contains("The first 2 of 5 rows were not judged", warning, StringComparison.Ordinal);
    }

    [Fact]
    public void AScoreOfZeroLaterInTheSeries_IsAJudgedRow()
    {
        var (rows, warning) = TimeSeriesAnomalyWarmup.Mark([Row(false, 0.04), Row(false, 0), Row(false, 0.02)]);

        Assert.Null(warning);
        Assert.False(rows[1].IsAnomaly);
    }

    [Fact]
    public void ARequestShorterThanTheWindow_IsNotJudgedAtAll_AndSaysSo()
    {
        var (rows, warning) = TimeSeriesAnomalyWarmup.Mark([Row(false, 0)]);

        Assert.Null(Assert.Single(rows).IsAnomaly);
        Assert.StartsWith("None of the 1 row(s) was judged", warning, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAlert_IsNeverTakenForWarmUp()
    {
        var (rows, warning) = TimeSeriesAnomalyWarmup.Mark([Row(true, 0), Row(false, 0.1)]);

        Assert.Null(warning);
        Assert.True(rows[0].IsAnomaly);
    }
}
