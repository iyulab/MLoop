using MLoop.CLI.Commands;
using MLoop.Core.Data;
using MLoop.DataStore.Interfaces;

namespace MLoop.Tests.Commands;

public class SampleCommandTests
{
    #region ParseStrategy (Prediction Log)

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
    [InlineData("head")]
    public void ParseStrategy_InvalidStrategies_ReturnsNull(string input)
    {
        Assert.Null(SampleCommand.ParseStrategy(input));
    }

    #endregion

    #region ParseCsvStrategy

    [Theory]
    [InlineData("random", CsvSamplingStrategy.Random)]
    [InlineData("Random", CsvSamplingStrategy.Random)]
    [InlineData("RANDOM", CsvSamplingStrategy.Random)]
    [InlineData("head", CsvSamplingStrategy.Head)]
    [InlineData("Head", CsvSamplingStrategy.Head)]
    [InlineData("stratified", CsvSamplingStrategy.Stratified)]
    [InlineData("Stratified", CsvSamplingStrategy.Stratified)]
    public void ParseCsvStrategy_ValidStrategies_ReturnsParsed(string input, CsvSamplingStrategy expected)
    {
        Assert.Equal(expected, SampleCommand.ParseCsvStrategy(input));
    }

    [Theory]
    [InlineData("unknown")]
    [InlineData("")]
    [InlineData("recent")]
    [InlineData("feedback-priority")]
    public void ParseCsvStrategy_InvalidStrategies_ReturnsNull(string input)
    {
        Assert.Null(SampleCommand.ParseCsvStrategy(input));
    }

    #endregion
}
