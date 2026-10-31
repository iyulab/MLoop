using MLoop.Core.DeepLearning;
using MLoop.Core.Models;

namespace MLoop.Core.DeepLearning.Tests.Handlers;

/// <summary>
/// The epoch count a deep-learning fit runs and reports against is one number: <c>training.epochs</c> when the run
/// sets it, the module's default otherwise — the trainers and the progress bar read it from the same place.
/// </summary>
public class EpochCountTests
{
    private static TrainingConfig Config(int? epochs) => new()
    {
        ModelName = "default",
        DataFile = "d.csv",
        LabelColumn = "y",
        Task = "object-detection",
        Epochs = epochs
    };

    [Fact]
    public void AnUnsetCount_IsTheDefault() => Assert.Equal(EpochProgress.DefaultEpochs, EpochProgress.Of(Config(null)));

    [Fact]
    public void ASetCount_IsUsedAsGiven() => Assert.Equal(3, EpochProgress.Of(Config(3)));

    [Fact]
    public void TheDefault_IsTenEpochs() => Assert.Equal(10, EpochProgress.DefaultEpochs);
}
