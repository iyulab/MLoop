using Microsoft.ML;
using MLoop.Core.Models;
using Xunit;

namespace MLoop.Core.Tests.AutoML;

[Collection("FileSystem")]
public class PreFeaturizerInjectionTests
{
    [Fact]
    public void TrainingConfig_exposes_prefeaturizer()
    {
        var ctx = new MLContext(seed: 42);
        var est = ctx.Transforms.NormalizeMinMax("age", "age");
        var cfg = new TrainingConfig
        {
            ModelName = "test",
            DataFile = "x.csv",
            LabelColumn = "y",
            Task = "binary-classification",
            PreFeaturizer = est
        };
        Assert.NotNull(cfg.PreFeaturizer);
    }
}
