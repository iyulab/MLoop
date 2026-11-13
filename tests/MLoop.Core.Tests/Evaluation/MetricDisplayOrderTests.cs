using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// The order a table shows a run's metrics in: the metric the run was judged by first, never a metric ahead of another because its
/// number is larger.
/// </summary>
public class MetricDisplayOrderTests
{
    private static string[] Names(IEnumerable<KeyValuePair<string, double>> metrics, string? primary) =>
        [.. MetricDisplayOrder.Of(metrics, primary).Select(m => m.Key)];

    [Fact]
    public void ARegressionRun_ShowsRSquaredFirst_AndTheIntervalWidthsLast()
    {
        var metrics = new Dictionary<string, double>
        {
            ["mse"] = 2415483241.79,
            ["interval_half_width_95"] = 127133.5,
            ["rmse"] = 49147.57,
            ["residual_std"] = 49147.57,
            ["interval_half_width_80"] = 48777,
            ["mae"] = 33575.9,
            ["norm_interval_q_95"] = 3.31,
            ["interval_beta"] = 5280.06,
            ["r_squared"] = 0.3991,
        };

        Assert.Equal(
            ["r_squared", "mae", "residual_std", "rmse", "mse", "interval_beta", "interval_half_width_80", "interval_half_width_95", "norm_interval_q_95"],
            Names(metrics, "r_squared"));
    }

    [Fact]
    public void AClassifier_ShowsItsRankingMetricFirst_ThenScoresHighestFirst_ThenLossesLowestFirst()
    {
        var metrics = new Dictionary<string, double>
        {
            ["negative_recall"] = 0.9748,
            ["auc"] = 0.9625,
            ["accuracy"] = 0.9246,
            ["ece"] = 0.0185,
            ["log_loss"] = 0.31,
            ["f1_score"] = 0.7097,
        };

        Assert.Equal(["auc", "negative_recall", "accuracy", "f1_score", "ece", "log_loss"], Names(metrics, "auc"));
    }

    [Fact]
    public void WithoutAPrimaryMetric_OrEqualValues_TheOrderIsStillOneOrder()
    {
        var metrics = new Dictionary<string, double> { ["b"] = 0.5, ["a"] = 0.5, ["c"] = 0.9 };

        Assert.Equal(["c", "a", "b"], Names(metrics, null));
        Assert.Equal(["c", "a", "b"], Names(metrics.Reverse().ToDictionary(m => m.Key, m => m.Value), "not_there"));
    }

    [Fact]
    public void ThePrimaryMetric_IsMatchedWithoutRegardToCase()
    {
        Assert.Equal("AUC", Names(new Dictionary<string, double> { ["accuracy"] = 0.99, ["AUC"] = 0.5 }, "auc")[0]);
    }
}
