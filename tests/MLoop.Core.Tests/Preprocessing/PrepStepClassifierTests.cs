using MLoop.Core.Preprocessing;
using Xunit;

namespace MLoop.Core.Tests.Preprocessing;

public class PrepStepClassifierTests
{
    [Theory]
    [InlineData("normalize", "min-max", PrepCategory.PreFeaturizer)]
    [InlineData("scale", "z-score", PrepCategory.PreFeaturizer)]
    [InlineData("fill-missing", "mean", PrepCategory.PreFeaturizer)]
    [InlineData("fill-missing", "median", PrepCategory.UnsupportedLeakageWarn)]
    [InlineData("fill-missing", "constant", PrepCategory.CsvStage)]
    [InlineData("remove-columns", null, PrepCategory.CsvStage)]
    [InlineData("extract-date", null, PrepCategory.CsvStage)]
    [InlineData("rolling", null, PrepCategory.UnsupportedLeakageWarn)]
    [InlineData("resample", null, PrepCategory.UnsupportedLeakageWarn)]
    public void Classify_categorizes_by_type_and_method(string type, string? method, PrepCategory expected)
    {
        var step = new PrepStep { Type = type, Method = method };
        Assert.Equal(expected, PrepStepClassifier.Classify(step));
    }
}
