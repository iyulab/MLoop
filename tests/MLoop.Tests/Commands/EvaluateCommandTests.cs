using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

public class EvaluateCommandTests
{
    #region IsLowerBetterMetric

    [Theory]
    [InlineData("rmse", true)]
    [InlineData("mae", true)]
    [InlineData("mse", true)]
    [InlineData("log-loss", true)]
    [InlineData("RMSE", true)]
    [InlineData("root_mean_squared_error", true)]  // contains "error" → lower-better (now matches ModelRegistry/promotion)
    [InlineData("accuracy", false)]
    [InlineData("auc", false)]
    [InlineData("r_squared", false)]
    [InlineData("f1", false)]
    public void IsLowerBetterMetric_ClassifiesCorrectly(string metric, bool expected)
    {
        Assert.Equal(expected, EvaluateCommand.IsLowerBetterMetric(metric));
    }

    #endregion

    #region FormatMetricDifference

    [Fact]
    public void FormatMetricDifference_HigherBetter_PositiveIsGreen()
    {
        var result = EvaluateCommand.FormatMetricDifference("accuracy", 0.05);
        Assert.Contains("[green]", result);
        Assert.Contains("+", result);
    }

    [Fact]
    public void FormatMetricDifference_HigherBetter_NegativeIsRed()
    {
        var result = EvaluateCommand.FormatMetricDifference("accuracy", -0.05);
        Assert.Contains("[red]", result);
    }

    [Fact]
    public void FormatMetricDifference_LowerBetter_NegativeIsGreen()
    {
        var result = EvaluateCommand.FormatMetricDifference("rmse", -0.05);
        Assert.Contains("[green]", result);
    }

    [Fact]
    public void FormatMetricDifference_LowerBetter_PositiveIsRed()
    {
        var result = EvaluateCommand.FormatMetricDifference("rmse", 0.05);
        Assert.Contains("[red]", result);
        Assert.Contains("+", result);
    }

    [Fact]
    public void FormatMetricDifference_ZeroDifference_HigherBetter()
    {
        var result = EvaluateCommand.FormatMetricDifference("accuracy", 0.0);
        Assert.Contains("[green]", result);
    }

    [Fact]
    public void FormatMetricDifference_ZeroDifference_LowerBetter()
    {
        var result = EvaluateCommand.FormatMetricDifference("rmse", 0.0);
        Assert.Contains("[green]", result);
    }

    #endregion

    #region IsLowerBetterMetric

    [Fact]
    public void IsLowerBetterMetric_ClusteringAndErrorMetrics_ReturnsTrue()
    {
        // average_distance and davies_bouldin_index are lower-is-better (clustering's canonical
        // metric is average_distance), and mape is an error metric — but the local check only knew
        // rmse/mae/mse/loss. So evaluate's diff coloring and compare's sort treated a worse clustering
        // model (higher average_distance) as "best". ModelRegistry already knew these; the direction
        // logic was duplicated and drifted across 4 sites, now converged on MetricDirection.
        Assert.True(EvaluateCommand.IsLowerBetterMetric("average_distance"));
        Assert.True(EvaluateCommand.IsLowerBetterMetric("davies_bouldin_index"));
        Assert.True(EvaluateCommand.IsLowerBetterMetric("mape"));
    }

    [Fact]
    public void IsLowerBetterMetric_HigherBetterMetrics_ReturnsFalse()
    {
        Assert.False(EvaluateCommand.IsLowerBetterMetric("accuracy"));
        Assert.False(EvaluateCommand.IsLowerBetterMetric("r_squared"));
        Assert.False(EvaluateCommand.IsLowerBetterMetric("ndcg"));
        Assert.False(EvaluateCommand.IsLowerBetterMetric("auc"));
    }

    #endregion

    #region DetectOverfitting

    [Fact]
    public void DetectOverfitting_Regression_LargeR2Diff_ReturnsTrue()
    {
        var train = new Dictionary<string, double> { { "r_squared", 0.95 } };
        var test = new Dictionary<string, double> { { "r_squared", 0.80 } };

        Assert.True(EvaluateCommand.DetectOverfitting("regression", train, test));
    }

    [Fact]
    public void DetectOverfitting_Regression_SmallR2Diff_ReturnsFalse()
    {
        var train = new Dictionary<string, double> { { "r_squared", 0.90 } };
        var test = new Dictionary<string, double> { { "r_squared", 0.88 } };

        Assert.False(EvaluateCommand.DetectOverfitting("regression", train, test));
    }

    [Fact]
    public void DetectOverfitting_Classification_LargeAccDiff_ReturnsTrue()
    {
        var train = new Dictionary<string, double> { { "accuracy", 0.98 } };
        var test = new Dictionary<string, double> { { "accuracy", 0.80 } };

        Assert.True(EvaluateCommand.DetectOverfitting("classification", train, test));
    }

    [Fact]
    public void DetectOverfitting_Classification_SmallAccDiff_ReturnsFalse()
    {
        var train = new Dictionary<string, double> { { "accuracy", 0.92 } };
        var test = new Dictionary<string, double> { { "accuracy", 0.90 } };

        Assert.False(EvaluateCommand.DetectOverfitting("classification", train, test));
    }

    [Fact]
    public void DetectOverfitting_BinaryClassification_LargeAccDiff_ReturnsTrue()
    {
        // the task is stored as "binary-classification" (CLI-canonical), not "classification";
        // the old code only matched "classification", so overfitting detection was dead for every
        // real binary model.
        var train = new Dictionary<string, double> { { "accuracy", 0.98 } };
        var test = new Dictionary<string, double> { { "accuracy", 0.80 } };

        Assert.True(EvaluateCommand.DetectOverfitting("binary-classification", train, test));
    }

    // --- the eight tasks the three-task switch never reached ---

    [Theory]
    [InlineData("image-classification", "micro_accuracy")]
    [InlineData("text-classification", "micro_accuracy")]
    [InlineData("ranking", "ndcg")]
    [InlineData("anomaly-detection", "auc")]
    [InlineData("time-series-anomaly", "detection_rate")]
    public void DetectOverfitting_UnitScaledTasks_AreChecked(string task, string metric)
    {
        var train = new Dictionary<string, double> { { metric, 0.97 } };
        var test = new Dictionary<string, double> { { metric, 0.78 } };

        Assert.True(EvaluateCommand.DetectOverfitting(task, train, test));
    }

    /// <summary>
    /// A metric in the label's own units cannot be judged by an absolute 0.1 — that number is a
    /// statement about the unit. An rmse of 4.0 against 4.4 is the same story as 0.004 against
    /// 0.0044, and both are the story this check exists to tell.
    /// </summary>
    [Theory]
    [InlineData("recommendation", "rmse", 4.0, 4.4)]
    [InlineData("recommendation", "rmse", 0.004, 0.0044)]
    [InlineData("forecasting", "mae", 1200.0, 1500.0)]
    [InlineData("clustering", "average_distance", 2.0, 2.5)]
    public void DetectOverfitting_UnitCarryingMetrics_UseARelativeGap(
        string task, string metric, double trainValue, double testValue)
    {
        var train = new Dictionary<string, double> { { metric, trainValue } };
        var test = new Dictionary<string, double> { { metric, testValue } };

        Assert.True(EvaluateCommand.DetectOverfitting(task, train, test));
    }

    [Theory]
    [InlineData("recommendation", "rmse", 4.0, 4.1)]      // 2.5% — noise
    [InlineData("forecasting", "mae", 1200.0, 1250.0)]    // 4%
    public void DetectOverfitting_UnitCarryingMetrics_IgnoreASmallRelativeGap(
        string task, string metric, double trainValue, double testValue)
    {
        var train = new Dictionary<string, double> { { metric, trainValue } };
        var test = new Dictionary<string, double> { { metric, testValue } };

        Assert.False(EvaluateCommand.DetectOverfitting(task, train, test));
    }

    /// <summary>
    /// Scoring better on test than on train is not overfitting — it is the one thing the warning
    /// cannot mean. The check used to compare with <c>Math.Abs</c> and report it anyway.
    /// </summary>
    [Theory]
    [InlineData("regression", "r_squared", 0.80, 0.95)]        // higher is better: test ahead
    [InlineData("recommendation", "rmse", 4.4, 2.0)]           // lower is better: test ahead
    public void DetectOverfitting_TestScoringBetter_IsNotOverfitting(
        string task, string metric, double trainValue, double testValue)
    {
        var train = new Dictionary<string, double> { { metric, trainValue } };
        var test = new Dictionary<string, double> { { metric, testValue } };

        Assert.False(EvaluateCommand.DetectOverfitting(task, train, test));
    }

    /// <summary>A task with no canonical primary metric has nothing to compare.</summary>
    [Fact]
    public void DetectOverfitting_TaskWithoutAPrimaryMetric_ReturnsFalse()
    {
        var train = new Dictionary<string, double> { { "map_50", 0.95 } };
        var test = new Dictionary<string, double> { { "map_50", 0.50 } };

        Assert.False(EvaluateCommand.DetectOverfitting("object-detection", train, test));
    }

    [Fact]
    public void DetectOverfitting_MulticlassClassification_LargeMacroAccDiff_ReturnsTrue()
    {
        // multiclass is stored as "multiclass-classification" and its primary metric key is
        // "macro_accuracy" (not "accuracy") — so the old "classification"+"accuracy" check was
        // doubly wrong and dead for every multiclass model.
        var train = new Dictionary<string, double> { { "macro_accuracy", 0.97 } };
        var test = new Dictionary<string, double> { { "macro_accuracy", 0.78 } };

        Assert.True(EvaluateCommand.DetectOverfitting("multiclass-classification", train, test));
    }

    [Fact]
    public void DetectOverfitting_MulticlassClassification_SmallDiff_ReturnsFalse()
    {
        var train = new Dictionary<string, double> { { "macro_accuracy", 0.90 } };
        var test = new Dictionary<string, double> { { "macro_accuracy", 0.88 } };

        Assert.False(EvaluateCommand.DetectOverfitting("multiclass-classification", train, test));
    }

    [Fact]
    public void DetectOverfitting_MissingMetrics_ReturnsFalse()
    {
        var train = new Dictionary<string, double>();
        var test = new Dictionary<string, double>();

        Assert.False(EvaluateCommand.DetectOverfitting("regression", train, test));
        Assert.False(EvaluateCommand.DetectOverfitting("classification", train, test));
    }

    [Fact]
    public void DetectOverfitting_UnknownTask_ReturnsFalse()
    {
        var train = new Dictionary<string, double> { { "r_squared", 0.95 } };
        var test = new Dictionary<string, double> { { "r_squared", 0.50 } };

        Assert.False(EvaluateCommand.DetectOverfitting("clustering", train, test));
    }

    [Fact]
    public void DetectOverfitting_ExactlyAtThreshold_ReturnsFalse()
    {
        var train = new Dictionary<string, double> { { "r_squared", 0.90 } };
        var test = new Dictionary<string, double> { { "r_squared", 0.80 } };

        // diff = 0.1, threshold = 0.1, so > 0.1 is false
        Assert.False(EvaluateCommand.DetectOverfitting("regression", train, test));
    }

    #endregion
}
