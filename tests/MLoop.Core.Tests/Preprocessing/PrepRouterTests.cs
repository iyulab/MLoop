using Microsoft.ML;
using MLoop.Core.Preprocessing;

namespace MLoop.Core.Tests.Preprocessing;

public class PrepRouterTests
{
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
