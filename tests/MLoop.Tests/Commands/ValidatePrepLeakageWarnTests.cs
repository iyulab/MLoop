using MLoop.CLI.Commands;
using MLoop.Core.Preprocessing;

namespace MLoop.Tests.Commands;

public class ValidatePrepLeakageWarnTests
{
    [Fact]
    public void InspectPrepLeakage_PreFeaturizerTask_flags_median_and_timeseries_not_normalize()
    {
        var prep = new List<PrepStep>
        {
            new() { Type = "fill-missing", Method = "median", Columns = new() { "age" } }, // warn
            new() { Type = "rolling", Column = "v", WindowSize = 3 },                       // warn
            new() { Type = "normalize", Method = "min-max", Columns = new() { "age" } }     // safe (fold-internal)
        };

        // regression supports the preFeaturizer → normalize is routed fold-internally, no leak.
        var warnings = ValidateCommand.InspectPrepLeakage(prep, "regression");

        Assert.Equal(2, warnings.Count); // median fill + rolling
        Assert.All(warnings, w => Assert.Contains("누수", w));
    }

    [Fact]
    public void InspectPrepLeakage_NonPreFeaturizerTask_alsoFlagsNormalize()
    {
        var prep = new List<PrepStep>
        {
            new() { Type = "fill-missing", Method = "median", Columns = new() { "age" } }, // warn (always)
            new() { Type = "normalize", Method = "min-max", Columns = new() { "age" } }     // warn (task can't fold-fit)
        };

        // clustering ignores the preFeaturizer → normalize is CSV-baked globally → leaks.
        // train-time (PrepRouter) warns here; validate must match (correctness gap T-B).
        var warnings = ValidateCommand.InspectPrepLeakage(prep, "clustering");

        Assert.Equal(2, warnings.Count); // median fill + normalize
        Assert.All(warnings, w => Assert.Contains("누수", w));
        // the normalize warning is the task-aware variant, not the "replace with normalize" advice
        Assert.Contains(warnings, w => w.Contains("preFeaturizer(fold-내 fit)를 지원하지 않아"));
    }
}
