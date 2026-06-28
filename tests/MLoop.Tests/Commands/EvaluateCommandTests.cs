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
    [InlineData("root_mean_squared_error", true)]  // contains "error" → lower-better (F-27: now matches ModelRegistry/promotion)
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
        // F-27: average_distance and davies_bouldin_index are lower-is-better (clustering's canonical
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
        // F-25: the task is stored as "binary-classification" (CLI-canonical), not "classification";
        // the old code only matched "classification", so overfitting detection was dead for every
        // real binary model.
        var train = new Dictionary<string, double> { { "accuracy", 0.98 } };
        var test = new Dictionary<string, double> { { "accuracy", 0.80 } };

        Assert.True(EvaluateCommand.DetectOverfitting("binary-classification", train, test));
    }

    [Fact]
    public void DetectOverfitting_MulticlassClassification_LargeMacroAccDiff_ReturnsTrue()
    {
        // F-25: multiclass is stored as "multiclass-classification" and its primary metric key is
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
