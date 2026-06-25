using Microsoft.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Preprocessing;

namespace MLoop.Core.Tests.Preprocessing;

public class PrepRouterTests
{
    [Fact]
    public void Route_keeps_statistical_in_csv_with_warning_when_prefeaturizer_unsupported()
    {
        // clustering/anomaly Execute sites ignore config.PreFeaturizer; routing a normalize
        // there must NOT silently drop it — it stays a CSV step (applied) + leakage warning.
        Assert.False(AutoMLRunner.SupportsPreFeaturizer("clustering"));

        var ctx = new MLContext(seed: 42);
        var steps = new List<PrepStep>
        {
            new() { Type = "normalize", Method = "min-max", Columns = new() { "age" } },
        };

        var result = new PrepRouter().Route(ctx, steps, supportsPreFeaturizer: false);

        Assert.Null(result.PreFeaturizer);                                   // not routed to preFeaturizer
        Assert.Empty(result.PreFeaturizerColumns);
        Assert.Contains(result.CsvSteps, s => s.Type == "normalize");        // applied in CSV stage instead
        Assert.Single(result.Warnings);                                      // leakage warning emitted

        // Task-aware message: must NOT tell user to "replace normalize with normalize"
        var warning = result.Warnings[0];
        Assert.DoesNotContain("대체하세요", warning);                         // no self-contradictory advice
        Assert.Contains("preFeaturizer", warning);                           // explains WHY (task limitation)
        Assert.Contains("누수", warning);                                    // still warns about leakage
    }

    [Fact]
    public void Route_unsupported_task_median_step_still_uses_original_leakage_warning()
    {
        // median/rolling/resample steps on non-supporting tasks should still use
        // the original LeakageWarning message (advice to switch to normalize/fill-mean is still valid).
        var ctx = new MLContext(seed: 42);
        var steps = new List<PrepStep>
        {
            new() { Type = "fill-missing", Method = "median", Columns = new() { "age" } },
        };

        var result = new PrepRouter().Route(ctx, steps, supportsPreFeaturizer: false);

        Assert.Single(result.Warnings);
        var warning = result.Warnings[0];
        // Original LeakageWarning contains the "대체하세요" advice — still correct for median
        Assert.Contains("대체하세요", warning);
        Assert.Contains("누수", warning);
    }

    [Fact]
    public void Route_sends_statistical_to_prefeaturizer_when_supported()
    {
        Assert.True(AutoMLRunner.SupportsPreFeaturizer("binary-classification"));

        var ctx = new MLContext(seed: 42);
        var steps = new List<PrepStep>
        {
            new() { Type = "normalize", Method = "min-max", Columns = new() { "age" } },
        };

        var result = new PrepRouter().Route(ctx, steps, supportsPreFeaturizer: true);

        Assert.NotNull(result.PreFeaturizer);                                // unchanged behavior
        Assert.DoesNotContain(result.CsvSteps, s => s.Type == "normalize");
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void Route_splits_statistical_into_prefeaturizer_and_keeps_rest_in_csv()
    {
        var ctx = new MLContext(seed: 42);
        var steps = new List<PrepStep>
        {
            new() { Type = "normalize", Method = "min-max", Columns = new() { "age" } }, // preFeaturizer
            new() { Type = "remove-columns", Columns = new() { "id" } },                 // csv
            new() { Type = "fill-missing", Method = "median", Columns = new() { "x" } }   // csv + warn
        };

        var result = new PrepRouter().Route(ctx, steps);

        Assert.NotNull(result.PreFeaturizer);                       // normalize → preFeaturizer
        Assert.Equal(2, result.CsvSteps.Count);                    // remove-columns + median fill
        Assert.DoesNotContain(result.CsvSteps, s => s.Type == "normalize");
        Assert.Single(result.Warnings);                            // median fill 누수 경고
    }

    [Fact]
    public void Route_returns_null_prefeaturizer_when_no_statistical_steps()
    {
        var ctx = new MLContext(seed: 42);
        var steps = new List<PrepStep> { new() { Type = "remove-columns", Columns = new() { "id" } } };
        var result = new PrepRouter().Route(ctx, steps);
        Assert.Null(result.PreFeaturizer);
        Assert.Single(result.CsvSteps);
        Assert.Empty(result.Warnings);
    }
}
