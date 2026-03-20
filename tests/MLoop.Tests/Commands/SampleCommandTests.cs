using MLoop.CLI.Commands;
using MLoop.DataStore.Interfaces;

namespace MLoop.Tests.Commands;

public class SampleCommandTests
{
    #region ParseStrategy

    [Theory]
    [InlineData("random", SamplingStrategy.Random)]
    [InlineData("Random", SamplingStrategy.Random)]
    [InlineData("RANDOM", SamplingStrategy.Random)]
    [InlineData("recent", SamplingStrategy.Recent)]
    [InlineData("feedback-priority", SamplingStrategy.FeedbackPriority)]
    [InlineData("feedbackpriority", SamplingStrategy.FeedbackPriority)]
    public void ParseStrategy_ValidStrategies_ReturnsParsed(string input, SamplingStrategy expected)
    {
        Assert.Equal(expected, SampleCommand.ParseStrategy(input));
    }

    [Theory]
    [InlineData("unknown")]
    [InlineData("")]
    [InlineData("stratified")]
    public void ParseStrategy_InvalidStrategies_ReturnsNull(string input)
    {
        Assert.Null(SampleCommand.ParseStrategy(input));
    }

    #endregion
}
