using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Prediction;

namespace MLoop.Tests.ML;

public class AutoStrategyNoteTests
{
    private static CategoricalMapper.MappingResult Mapping(double unknownRatio) => new()
    {
        Success = true,
        AppliedStrategy = CategoricalMapper.UnknownValueStrategy.UseMostFrequent,
        UnknownValueRatio = unknownRatio,
        StrategyReason = $"ratio {unknownRatio:P2}"
    };

    [Fact]
    public void Nothing_is_announced_when_no_value_was_unseen()
    {
        // Measured: every image-classification predict printed
        // "[Auto] Low unknown value ratio (0.00%) - safely replacing with most frequent values".
        Assert.Null(PredictionEngine.AutoStrategyNote(Mapping(0)));
    }

    [Fact]
    public void The_strategy_is_announced_when_some_values_were_unseen()
    {
        Assert.StartsWith("[Auto] ", PredictionEngine.AutoStrategyNote(Mapping(0.02)));
    }
}
