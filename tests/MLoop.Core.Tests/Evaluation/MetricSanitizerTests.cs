using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// Pins <see cref="MetricSanitizer"/>: non-finite metric values (NaN/±∞) must be replaced with the
/// direction-aware WORST representable value before persistence/selection — never 0. A NaN→0 guard
/// (which this authority replaced) is actively wrong for lower-is-better metrics, laundering a broken
/// model's error into a perfect-looking 0 that then wins selection and passes the promotion gate.
/// The motivating crash: System.Text.Json cannot serialize non-finite doubles, so a single stray
/// Infinity in the metrics dict aborted the whole training pipeline at metrics.json save time.
/// </summary>
public class MetricSanitizerTests
{
    [Theory]
    // Lower-is-better → worst is the LARGEST value.
    [InlineData("rmse", double.PositiveInfinity, double.MaxValue)]
    [InlineData("mae", double.NegativeInfinity, double.MaxValue)]
    [InlineData("mse", double.NaN, double.MaxValue)]
    [InlineData("log_loss", double.PositiveInfinity, double.MaxValue)]
    // Higher-is-better → worst is the SMALLEST value.
    [InlineData("r_squared", double.NegativeInfinity, double.MinValue)]
    [InlineData("accuracy", double.NaN, double.MinValue)]
    [InlineData("auc", double.PositiveInfinity, double.MinValue)]
    public void SanitizeInPlace_ReplacesNonFinite_WithDirectionAwareWorst(string key, double input, double expected)
    {
        var metrics = new Dictionary<string, double> { [key] = input };

        var replaced = MetricSanitizer.SanitizeInPlace(metrics);

        Assert.Equal(expected, metrics[key]);
        Assert.Equal(new[] { key }, replaced);
    }

    [Fact]
    public void SanitizeInPlace_LeavesFiniteValuesUntouched_AndReportsNothing()
    {
        var metrics = new Dictionary<string, double>
        {
            ["r_squared"] = 0.0,      // a genuine degenerate-but-finite R² must survive as-is
            ["rmse"] = 3.14,
            ["mae"] = 0.0,
        };

        var replaced = MetricSanitizer.SanitizeInPlace(metrics);

        Assert.Empty(replaced);
        Assert.Equal(0.0, metrics["r_squared"]);
        Assert.Equal(3.14, metrics["rmse"]);
    }

    [Fact]
    public void SanitizeInPlace_ReportsOnlyTheReplacedKeys_InAMixedDict()
    {
        var metrics = new Dictionary<string, double>
        {
            ["rmse"] = 2.0,                        // finite — kept
            ["r_squared"] = double.NegativeInfinity, // replaced
            ["log_loss"] = double.NaN,               // replaced
        };

        var replaced = MetricSanitizer.SanitizeInPlace(metrics);

        Assert.Equal(2.0, metrics["rmse"]);
        Assert.Equal(double.MinValue, metrics["r_squared"]);
        Assert.Equal(double.MaxValue, metrics["log_loss"]);
        Assert.Equal(2, replaced.Count);
        Assert.Contains("r_squared", replaced);
        Assert.Contains("log_loss", replaced);
    }

    [Theory]
    [InlineData(null)]
    public void SanitizeInPlace_IsNullSafe(Dictionary<string, double>? metrics)
        => Assert.Empty(MetricSanitizer.SanitizeInPlace(metrics));

    [Fact]
    public void SanitizeInPlace_EmptyDict_IsNoOp()
        => Assert.Empty(MetricSanitizer.SanitizeInPlace(new Dictionary<string, double>()));

    [Fact]
    public void SanitizeAndReturn_ReturnsTheSameMutatedInstance()
    {
        var metrics = new Dictionary<string, double> { ["rmse"] = double.PositiveInfinity };

        var result = MetricSanitizer.SanitizeAndReturn(metrics);

        Assert.Same(metrics, result);
        Assert.Equal(double.MaxValue, result["rmse"]);
    }
}
