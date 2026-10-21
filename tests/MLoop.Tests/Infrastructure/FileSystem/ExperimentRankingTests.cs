using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Infrastructure.FileSystem;

/// <summary>
/// Guard: best-experiment selection must honor metric direction. Before the fix, Status/Compare/
/// API sorted BestMetric descending unconditionally, ranking the worst clustering/forecasting/
/// recommendation model (highest average_distance/mae/rmse) as "best".
/// </summary>
public class ExperimentRankingTests
{
    [Fact]
    public void SelectBest_LowerBetterMetric_PicksSmallest()
    {
        var exps = new[]
        {
            Summary("exp-1", "average_distance", 5.0),
            Summary("exp-2", "average_distance", 2.0), // best — lowest distance
            Summary("exp-3", "average_distance", 9.0),
        };

        Assert.Equal("exp-2", ExperimentRanking.SelectBest(exps)!.Best.ExperimentId);
    }

    [Fact]
    public void SelectBest_HigherBetterMetric_PicksLargest()
    {
        var exps = new[]
        {
            Summary("exp-1", "accuracy", 0.70),
            Summary("exp-2", "accuracy", 0.95), // best — highest accuracy
            Summary("exp-3", "accuracy", 0.80),
        };

        Assert.Equal("exp-2", ExperimentRanking.SelectBest(exps)!.Best.ExperimentId);
    }

    [Fact]
    public void OrderByQuality_LowerBetter_OrdersAscending()
    {
        var exps = new[]
        {
            Summary("exp-1", "rmse", 3.0),
            Summary("exp-2", "rmse", 1.0),
            Summary("exp-3", "rmse", 2.0),
        };

        var ordered = ExperimentRanking.OrderByQuality(exps).Select(e => e.ExperimentId).ToArray();

        Assert.Equal(new[] { "exp-2", "exp-3", "exp-1" }, ordered);
    }

    [Fact]
    public void SelectBest_NoMetrics_ReturnsNull()
        => Assert.Null(ExperimentRanking.SelectBest(new[] { Summary("exp-1", "accuracy", null) }));

    [Fact]
    public void SelectBest_DoesNotRankAcrossWeightings()
    {
        // The unweighted 0.99 is on another scale than the weighted runs; the newest is weighted.
        var exps = new[]
        {
            Summary("exp-1", "auc", 0.99, ageDays: 3),
            Summary("exp-2", "auc", 0.80, ageDays: 2, weight: "w"),
            Summary("exp-3", "auc", 0.85, ageDays: 1, weight: "W"),
        };

        var pick = ExperimentRanking.SelectBest(exps)!;

        Assert.Equal("exp-3", pick.Best.ExperimentId);
        Assert.Equal(new ExperimentRanking.Measurement("auc", Weighting.Of("w")), pick.Measurement);
        Assert.Equal(1, pick.NotCompared);
        Assert.Equal("auc, weighted by 'W'", pick.Measurement.ToString());   // as the picked experiment recorded it
    }

    [Fact]
    public void OrderByQuality_PutsExperimentsMeasuredLikeTheNewestFirst()
    {
        var exps = new[]
        {
            Summary("exp-1", "auc", 0.99, ageDays: 3),
            Summary("exp-2", "auc", 0.80, ageDays: 2, weight: "w"),
            Summary("exp-3", "auc", 0.85, ageDays: 1, weight: "w"),
            Summary("exp-4", "auc", null, ageDays: 0),
        };

        var ordered = ExperimentRanking.OrderByQuality(exps).Select(e => e.ExperimentId).ToArray();

        Assert.Equal(new[] { "exp-3", "exp-2", "exp-1", "exp-4" }, ordered);
    }

    [Fact]
    public void SelectBest_DoesNotRankAcrossMetrics()
    {
        var exps = new[]
        {
            Summary("exp-1", "accuracy", 0.99, ageDays: 2),
            Summary("exp-2", "f1_score", 0.40, ageDays: 1),
        };

        var pick = ExperimentRanking.SelectBest(exps)!;

        Assert.Equal("exp-2", pick.Best.ExperimentId);
        Assert.Equal(1, pick.NotCompared);
    }

    [Fact]
    public void SelectBest_AllMeasuredAlike_ComparesEveryOne()
    {
        var exps = new[]
        {
            Summary("exp-1", "auc", 0.95, ageDays: 2),
            Summary("exp-2", "auc", 0.90, ageDays: 1),
        };

        var pick = ExperimentRanking.SelectBest(exps)!;

        Assert.Equal("exp-1", pick.Best.ExperimentId);
        Assert.Equal(0, pick.NotCompared);
    }

    private static ExperimentSummary Summary(string id, string metric, double? value, int ageDays = 0, string? weight = null)
        => new()
        {
            ModelName = "m",
            ExperimentId = id,
            Timestamp = new DateTime(2026, 1, 10).AddDays(-ageDays),
            WeightColumn = weight,
            Status = "Completed",
            MetricName = metric,
            BestMetric = value,
        };

    [Theory]
    [InlineData(null, "", true)]
    [InlineData(" ", null, true)]
    [InlineData("W", "w", true)]        // columns are matched without case
    [InlineData("w", null, false)]
    [InlineData("w", "v", false)]
    public void Weighting_IsTheSameWhenTheRowsWereCountedTheSameWay(string? a, string? b, bool same)
    {
        Assert.Equal(same, Weighting.Of(a) == Weighting.Of(b));
        Assert.Equal(!same, Weighting.Differ([Weighting.Of(a), Weighting.Of(b)]));
    }

    [Fact]
    public void Weighting_NamesItself_KeepingTheColumnsCase()
    {
        Assert.Equal("unweighted", Weighting.Of(null).ToString());
        Assert.Equal("weighted by 'Weight'", Weighting.Of("Weight").ToString());
    }
}
