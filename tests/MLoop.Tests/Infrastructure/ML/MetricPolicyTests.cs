using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// Unit tests for <see cref="MetricPolicy"/> — the CLI promotion-gate metric policy
/// (alias/canonical resolution, quality-gate thresholds, degenerate-model detection).
/// Extracted from ModelRegistryTests alongside the helpers themselves (SRP).
/// </summary>
public class MetricPolicyTests
{
    [Theory]
    [InlineData("f1", "f1_score")]            // alias → canonical
    [InlineData("F1", "f1_score")]            // case-insensitive
    [InlineData("f1_score", "f1_score")]      // exact match
    [InlineData("r2", "r_squared")]
    [InlineData("log-loss", "log_loss")]      // hyphen normalized
    [InlineData("accuracy", "accuracy")]      // binary keeps accuracy
    [InlineData("auc", "auc")]
    public void ResolveMetricKey_MapsAliasToStoredKey(string input, string expected)
    {
        var available = new[] { "accuracy", "auc", "f1_score", "precision", "recall", "r_squared", "log_loss" };
        Assert.Equal(expected, MetricPolicy.ResolveMetricKey(input, available));
    }

    [Fact]
    public void ResolveMetricKey_MulticlassAccuracy_FallsBackToMicroMacro()
    {
        // Multiclass stores macro/micro_accuracy, not plain "accuracy".
        var available = new[] { "macro_accuracy", "micro_accuracy", "log_loss" };
        Assert.Equal("micro_accuracy", MetricPolicy.ResolveMetricKey("accuracy", available));
    }

    [Fact]
    public void ResolveMetricKey_UnknownMetric_ReturnsNull()
    {
        var available = new[] { "accuracy", "f1_score" };
        Assert.Null(MetricPolicy.ResolveMetricKey("nonsense", available));
    }

    [Theory]
    [InlineData("r_squared", 0.0)]
    [InlineData("auc", 0.5)]
    [InlineData("accuracy", 0.0)]
    [InlineData("f1_score", 0.0)]
    public void GetMinimumMetricThreshold_ReturnsExpectedValues(string metric, double expected)
    {
        var threshold = MetricPolicy.GetMinimumMetricThreshold(metric);
        Assert.NotNull(threshold);
        Assert.Equal(expected, threshold.Value);
    }

    [Theory]
    [InlineData("mae")]
    [InlineData("rmse")]
    [InlineData("mse")]
    public void GetMinimumMetricThreshold_ErrorMetrics_ReturnsNull(string metric)
    {
        var threshold = MetricPolicy.GetMinimumMetricThreshold(metric);
        Assert.Null(threshold);
    }

    [Theory]
    [InlineData("accuracy", 2, 0.5)]     // Binary: 1/2
    [InlineData("accuracy", 3, 0.3333)]  // 3-class: 1/3
    [InlineData("accuracy", 10, 0.1)]    // 10-class: 1/10
    [InlineData("macro_accuracy", 5, 0.2)] // 5-class: 1/5
    public void GetMinimumMetricThreshold_WithClassCount_ReturnsDynamicThreshold(
        string metric, int classCount, double expected)
    {
        var threshold = MetricPolicy.GetMinimumMetricThreshold(metric, classCount);
        Assert.NotNull(threshold);
        Assert.Equal(expected, threshold.Value, 3); // precision to 3 decimal places
    }

    [Fact]
    public void GetMinimumMetricThreshold_AccuracyWithoutClassCount_ReturnsZero()
    {
        var threshold = MetricPolicy.GetMinimumMetricThreshold("accuracy");
        Assert.NotNull(threshold);
        Assert.Equal(0.0, threshold.Value);
    }

    [Fact]
    public void GetMinimumMetricThreshold_AccuracyWithClassCount_HigherThanWithout()
    {
        var withoutClass = MetricPolicy.GetMinimumMetricThreshold("accuracy");
        var withClass = MetricPolicy.GetMinimumMetricThreshold("accuracy", 3);
        Assert.True(withClass > withoutClass);
    }

    [Theory]
    [InlineData("auto", "image-classification", "micro_accuracy")]
    [InlineData("auto", "regression", "r_squared")]
    [InlineData("auto", "binary-classification", "accuracy")]
    [InlineData("auto", "multiclass-classification", "macro_accuracy")]
    [InlineData("auto", "anomaly-detection", "auc")]
    [InlineData("f1", "binary-classification", "f1_score")]   // explicit metric still wins
    [InlineData("auto", "object-detection", null)]            // no canonical floor → null
    [InlineData("auto", null, null)]
    public void ResolveCanonicalMetricKey_FallsBackToTaskDefault(string metric, string? task, string? expected)
    {
        var available = new[]
        {
            "accuracy", "micro_accuracy", "macro_accuracy", "log_loss",
            "auc", "f1_score", "r_squared", "map_50"
        };
        Assert.Equal(expected, MetricPolicy.ResolveCanonicalMetricKey(metric, task, available));
    }

    [Fact]
    public void ResolveCanonicalMetricKey_TaskDefaultAbsentFromMetrics_ReturnsNull()
    {
        // regression default r_squared, but the metrics dict lacks it → null (no false threshold).
        var available = new[] { "mae", "rmse" };
        Assert.Null(MetricPolicy.ResolveCanonicalMetricKey("auto", "regression", available));
    }

    [Theory]
    // TD-06: after converging on the shared TaskMetadata source of truth, the gate resolves a
    // canonical metric for tasks the old DefaultMetricForTask switch left as null (clustering,
    // forecasting, ranking, recommendation, time-series-anomaly). Blocking is unaffected (these
    // are error/threshold-less metrics) but production comparison now uses the right metric.
    [InlineData("forecasting", "mae", "mae")]
    [InlineData("clustering", "average_distance", "average_distance")]
    [InlineData("ranking", "ndcg", "ndcg")]
    [InlineData("recommendation", "rmse", "rmse")]
    [InlineData("time-series-anomaly", "detection_rate", "detection_rate")]
    [InlineData("text-classification", "micro_accuracy", "micro_accuracy")]
    public void ResolveCanonicalMetricKey_ResolvesConvergedTaskMetrics(string task, string availableMetric, string expected)
    {
        var available = new[] { availableMetric, "log_loss" };
        Assert.Equal(expected, MetricPolicy.ResolveCanonicalMetricKey("auto", task, available));
    }

    #region IsClassificationDegenerateModel

    [Fact]
    public void IsClassificationDegenerateModel_BinaryHighAccZeroF1_ReturnsTrue()
    {
        var metrics = new Dictionary<string, double>
        {
            ["accuracy"] = 0.95,
            ["f1_score"] = 0.0
        };

        Assert.True(MetricPolicy.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_MulticlassHighAccZeroF1_ReturnsTrue()
    {
        var metrics = new Dictionary<string, double>
        {
            ["macro_accuracy"] = 0.80,
            ["macro_f1"] = 0.0
        };

        Assert.True(MetricPolicy.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_NormalMetrics_ReturnsFalse()
    {
        var metrics = new Dictionary<string, double>
        {
            ["accuracy"] = 0.85,
            ["f1_score"] = 0.78
        };

        Assert.False(MetricPolicy.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_LowAccZeroF1_ReturnsFalse()
    {
        // Low accuracy + zero F1 = just a bad model, not degenerate
        var metrics = new Dictionary<string, double>
        {
            ["accuracy"] = 0.3,
            ["f1_score"] = 0.0
        };

        Assert.False(MetricPolicy.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_RegressionMetrics_ReturnsFalse()
    {
        var metrics = new Dictionary<string, double>
        {
            ["r_squared"] = 0.9,
            ["rmse"] = 0.5
        };

        Assert.False(MetricPolicy.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_MulticlassNearZeroF1_ReturnsTrue()
    {
        // macro_f1 = 0.0005 (below 0.001 threshold) with decent macro_accuracy
        var metrics = new Dictionary<string, double>
        {
            ["macro_accuracy"] = 0.85,
            ["macro_f1"] = 0.0005
        };

        Assert.True(MetricPolicy.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_MulticlassLowButNotZeroF1_ReturnsFalse()
    {
        // macro_f1 = 0.05 (above 0.001 threshold) — poor but not degenerate
        var metrics = new Dictionary<string, double>
        {
            ["macro_accuracy"] = 0.6,
            ["macro_f1"] = 0.05
        };

        Assert.False(MetricPolicy.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_BinaryAlwaysPositiveZeroNegativeRecall_ReturnsTrue()
    {
        // D16: model always predicts the (majority) positive class — recall=1, F1 stays
        // high (2*prevalence/(1+prevalence)) since F1 is computed on the positive class,
        // so accuracy/f1_score/auc alone look healthy. negative_recall=0 is the only
        // signal that the model never once predicted the negative class. Reproduces the
        // live KAMP SEQ006 case: accuracy=0.747, f1_score=0.855, negative_recall=0.
        var metrics = new Dictionary<string, double>
        {
            ["accuracy"] = 0.7468,
            ["f1_score"] = 0.8550,
            ["recall"] = 1.0,
            ["negative_recall"] = 0.0
        };

        Assert.True(MetricPolicy.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_BinaryHealthyNegativeRecall_ReturnsFalse()
    {
        var metrics = new Dictionary<string, double>
        {
            ["accuracy"] = 0.85,
            ["f1_score"] = 0.78,
            ["recall"] = 0.80,
            ["negative_recall"] = 0.88
        };

        Assert.False(MetricPolicy.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_LowAccZeroNegativeRecall_ReturnsFalse()
    {
        // Low accuracy + zero negative_recall = just a bad model, not the degenerate pattern
        var metrics = new Dictionary<string, double>
        {
            ["accuracy"] = 0.4,
            ["f1_score"] = 0.3,
            ["negative_recall"] = 0.0
        };

        Assert.False(MetricPolicy.IsClassificationDegenerateModel(metrics));
    }

    #endregion

    #region GetMinimumMetricThreshold

    [Theory]
    [InlineData("r_squared", null, 0.0)]
    [InlineData("auc", null, 0.5)]
    [InlineData("f1_score", null, 0.0)]
    public void GetMinimumMetricThreshold_KnownMetrics_ReturnsThreshold(string metric, int? classCount, double expected)
    {
        var result = MetricPolicy.GetMinimumMetricThreshold(metric, classCount);
        Assert.NotNull(result);
        Assert.Equal(expected, result!.Value, precision: 4);
    }

    [Fact]
    public void GetMinimumMetricThreshold_AccuracyWithClassCount_ReturnsDynamic()
    {
        // 5 classes → threshold = 1/5 = 0.2
        var result = MetricPolicy.GetMinimumMetricThreshold("accuracy", 5);
        Assert.NotNull(result);
        Assert.Equal(0.2, result!.Value, precision: 4);
    }

    [Fact]
    public void GetMinimumMetricThreshold_UnknownMetric_ReturnsNull()
    {
        var result = MetricPolicy.GetMinimumMetricThreshold("custom_metric");
        Assert.Null(result);
    }

    [Fact]
    public void GetMinimumMetricThreshold_ErrorMetric_ReturnsNull()
    {
        var result = MetricPolicy.GetMinimumMetricThreshold("rmse");
        Assert.Null(result);
    }

    [Theory]
    [InlineData("f1_score", 2, 0.5)]
    [InlineData("f1_score", 3, 0.3333)]
    [InlineData("f1_score", 5, 0.2)]
    [InlineData("f1", 10, 0.1)]
    [InlineData("macro_f1", 4, 0.25)]
    [InlineData("macro_f1", 2, 0.5)]
    public void GetMinimumMetricThreshold_F1WithClassCount_ReturnsDynamic(string metric, int classCount, double expected)
    {
        var result = MetricPolicy.GetMinimumMetricThreshold(metric, classCount);
        Assert.NotNull(result);
        Assert.Equal(expected, result!.Value, precision: 4);
    }

    [Theory]
    [InlineData("f1_score")]
    [InlineData("f1")]
    [InlineData("macro_f1")]
    public void GetMinimumMetricThreshold_F1WithoutClassCount_ReturnsZero(string metric)
    {
        var result = MetricPolicy.GetMinimumMetricThreshold(metric);
        Assert.NotNull(result);
        Assert.Equal(0.0, result!.Value);
    }

    #endregion

    // MetricSanitizer records an undefined (NaN/∞) metric as the direction-aware worst-case
    // sentinel; its contract says such a value must never pass the promotion gate. The floor
    // gate only covers higher-is-better metrics, so this predicate is what blocks an
    // all-NaN-scoring degenerate regression (rmse = MaxValue) from auto-promoting.
    #region IsUndefinedMetricSentinel

    [Theory]
    [InlineData("rmse", double.MaxValue, true)]     // lower-better sentinel
    [InlineData("mae", double.MaxValue, true)]
    [InlineData("log_loss", double.MaxValue, true)]
    [InlineData("r_squared", double.MinValue, true)] // higher-better sentinel
    [InlineData("accuracy", double.MinValue, true)]
    [InlineData("rmse", double.MinValue, false)]     // wrong-direction extreme is not the sentinel
    [InlineData("r_squared", double.MaxValue, false)]
    [InlineData("rmse", 123.4, false)]               // ordinary values pass
    [InlineData("r_squared", -0.5, false)]           // bad-but-real values are the floor gate's job
    [InlineData("accuracy", 0.0, false)]
    public void IsUndefinedMetricSentinel_DetectsDirectionAwareWorstCase(string metric, double value, bool expected)
    {
        Assert.Equal(expected, MetricPolicy.IsUndefinedMetricSentinel(metric, value));
    }

    #endregion
}
