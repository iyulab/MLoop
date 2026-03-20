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
    [InlineData("root_mean_squared_error", false)]  // doesn't contain "rmse"
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
