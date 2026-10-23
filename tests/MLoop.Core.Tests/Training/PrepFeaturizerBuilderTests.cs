using Microsoft.ML;
using MLoop.Core.Preprocessing;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

public class PrepFeaturizerBuilderTests
{
    [Fact]
    public void Build_returns_null_when_no_statistical_steps()
    {
        var ctx = new MLContext(seed: 42);
        var steps = new List<PrepStep> { new() { Type = "remove-columns", Columns = new() { "id" } } };
        Assert.Null(new PrepFeaturizerBuilder().Build(ctx, steps));
    }

    [Fact]
    public void Build_creates_estimator_for_normalize_minmax()
    {
        var ctx = new MLContext(seed: 42);
        var steps = new List<PrepStep>
        {
            new() { Type = "normalize", Method = "min-max", Columns = new() { "age" } }
        };
        var est = new PrepFeaturizerBuilder().Build(ctx, steps);
        Assert.NotNull(est);
    }
}
