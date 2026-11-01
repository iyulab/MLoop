using MLoop.Core.DeepLearning;

namespace MLoop.Core.DeepLearning.Tests.Handlers;

/// <summary>
/// The loss a TorchSharp trainer logs at each step, gathered into a mean per epoch for the training log. A fit that finds nothing
/// could not be told from one that finds boxes it is not confident about while the log said only how many epochs had finished.
/// </summary>
public class EpochLossTests
{
    [Theory]
    [InlineData("Row: 1, Loss: 98.70762213545463", 98.70762213545463)]
    [InlineData("[Source=ObjectDetectionTrainer; TrainModel, Kind=Info] Row: 4, Loss: 1.55760960925679", 1.55760960925679)]
    [InlineData("Row: 12, Loss: 3.2E-05", 3.2e-05)]
    [InlineData("Row: 7, Loss: -0.5", -0.5)]
    public void TryParseStep_ReadsTheLossOfAStepLine(string message, double expected)
    {
        Assert.True(EpochLoss.TryParseStep(message, out var loss));
        Assert.Equal(expected, loss);
    }

    [Theory]
    [InlineData("Finished epoch 0")]
    [InlineData("Starting epoch 3")]
    [InlineData("Channel started")]
    [InlineData("Row: 3, Loss: NaN")]
    [InlineData("Row: 3, Loss: Infinity")]
    [InlineData("Row: 3, Loss: abc")]
    [InlineData("")]
    [InlineData(null)]
    public void TryParseStep_RefusesEverythingElse(string? message)
    {
        Assert.False(EpochLoss.TryParseStep(message, out _));
    }

    [Fact]
    public void FinishEpoch_ReturnsTheMeanOfTheStepsSinceTheLastEpochAndStartsOver()
    {
        var loss = new EpochLoss();
        loss.AddStep(4);
        loss.AddStep(2);
        var first = loss.FinishEpoch();
        loss.AddStep(1);
        var second = loss.FinishEpoch();

        Assert.Equal((3.0, 2), first);
        Assert.Equal((1.0, 1), second);
        Assert.Equal([3.0, 1.0], loss.EpochMeans);
    }

    [Fact]
    public void FinishEpoch_WithNoStepsLogged_SaysSoInsteadOfInventingAMean()
    {
        var loss = new EpochLoss();

        Assert.Null(loss.FinishEpoch());
        Assert.Empty(loss.EpochMeans);
    }

    [Fact]
    public void Describe_NamesTheEpochOfTheCountAndTheStepsTheMeanIsOf()
    {
        Assert.Equal(
            "Object detection (AutoFormerV2) epoch 3/10: loss 1.234 (mean of 17 steps)",
            EpochLoss.Describe("Object detection (AutoFormerV2)", 3, 10, 1.2344, 17));
    }

    [Fact]
    public async Task AddStep_FromManyThreads_LosesNoStep()
    {
        var loss = new EpochLoss();

        await Task.WhenAll(Enumerable.Range(0, 8).Select(_ => Task.Run(() =>
        {
            for (var i = 0; i < 1000; i++) loss.AddStep(1);
        })));

        Assert.Equal((1.0, 8000), loss.FinishEpoch());
    }
}
