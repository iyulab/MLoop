using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// F-27 guard: metric optimization direction (lower-is-better vs higher-is-better) is a single source
/// of truth in <see cref="MetricDirection"/>. It previously drifted across four sites, with three of
/// them missing clustering's average_distance/davies_bouldin_index and mape — so compare/evaluate
/// ranked a worse clustering model as "best". These cases pin the converged behavior.
/// </summary>
public class MetricDirectionTests
{
    [Theory]
    // Lower-is-better: error/distance metrics + clustering separation metrics.
    [InlineData("rmse", true)]
    [InlineData("mae", true)]
    [InlineData("mse", true)]
    [InlineData("mape", true)]
    [InlineData("log_loss", true)]
    [InlineData("root_mean_squared_error", true)]
    [InlineData("average_distance", true)]       // clustering canonical metric
    [InlineData("davies_bouldin_index", true)]   // clustering
    // Higher-is-better.
    [InlineData("accuracy", false)]
    [InlineData("macro_accuracy", false)]
    [InlineData("r_squared", false)]
    [InlineData("auc", false)]
    [InlineData("ndcg", false)]
    [InlineData("f1_score", false)]
    public void IsLowerBetter_ClassifiesByDirection(string metric, bool expected)
        => Assert.Equal(expected, MetricDirection.IsLowerBetter(metric));
}
