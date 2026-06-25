using MLoop.CLI.Commands;
using MLoop.Core.Preprocessing;

namespace MLoop.Tests.Commands;

public class ValidatePrepLeakageWarnTests
{
    [Fact]
    public void InspectPrepLeakage_flags_median_and_timeseries_not_normalize()
    {
        var prep = new List<PrepStep>
        {
            new() { Type = "fill-missing", Method = "median", Columns = new() { "age" } }, // warn
            new() { Type = "rolling", Column = "v", WindowSize = 3 },                       // warn
            new() { Type = "normalize", Method = "min-max", Columns = new() { "age" } }     // safe
        };

        var warnings = ValidateCommand.InspectPrepLeakage(prep);

        Assert.Equal(2, warnings.Count); // median fill + rolling
        Assert.All(warnings, w => Assert.Contains("누수", w));
    }
}
