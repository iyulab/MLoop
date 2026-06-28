using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Infrastructure.FileSystem;

/// <summary>
/// F-28 guard: best-experiment selection must honor metric direction. Before the fix, Status/Compare/
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

        Assert.Equal("exp-2", ExperimentRanking.SelectBest(exps)!.ExperimentId);
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

        Assert.Equal("exp-2", ExperimentRanking.SelectBest(exps)!.ExperimentId);
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

    private static ExperimentSummary Summary(string id, string metric, double? value)
        => new()
        {
            ModelName = "m",
            ExperimentId = id,
            Timestamp = new DateTime(2026, 1, 1),
            Status = "Completed",
            MetricName = metric,
            BestMetric = value,
        };
}
