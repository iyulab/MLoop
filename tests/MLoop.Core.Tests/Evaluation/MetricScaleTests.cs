using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

public class MetricScaleTests
{
    [Theory]
    [InlineData("accuracy")]
    [InlineData("macro_accuracy")]
    [InlineData("micro_accuracy")]
    [InlineData("auc")]
    [InlineData("auprc")]
    [InlineData("f1_score")]
    [InlineData("ndcg")]
    [InlineData("detection_rate")]
    [InlineData("r_squared")]
    [InlineData("ACCURACY")]
    [InlineData("  accuracy  ")]
    public void IsUnitScaled_TrueForProportionsAndRates(string metric)
        => Assert.True(MetricScale.IsUnitScaled(metric));

    [Theory]
    [InlineData("rmse")]
    [InlineData("mae")]
    [InlineData("mse")]
    [InlineData("log_loss")]
    [InlineData("average_distance")]
    [InlineData("davies_bouldin_index")]
    [InlineData("mape")]
    public void IsUnitScaled_FalseForMetricsCarryingTheLabelsUnits(string metric)
        => Assert.False(MetricScale.IsUnitScaled(metric));

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("business_score_v3")]
    public void IsUnitScaled_FalseWhenUnknownOrAbsent(string? metric)
        => Assert.False(MetricScale.IsUnitScaled(metric));

    /// <summary>
    /// Every task's canonical primary metric is one this authority has an opinion about — either
    /// unit-scaled or knowingly not. A primary metric missing from both lists would be judged as
    /// "carries units" by default, which is the safe answer but an unexamined one.
    /// </summary>
    [Fact]
    public void EveryTasksPrimaryMetric_IsClassifiedOrKnownToCarryUnits()
    {
        var unitCarrying = new[] { "rmse", "mae", "mse", "log_loss", "average_distance", "davies_bouldin_index", "mape" };

        foreach (var metric in TaskMetadata.AllPrimaryMetrics)
        {
            var classified = MetricScale.IsUnitScaled(metric) || unitCarrying.Contains(metric);
            Assert.True(classified, $"'{metric}' is a task's primary metric but MetricScale has no opinion about it.");
        }
    }
}
