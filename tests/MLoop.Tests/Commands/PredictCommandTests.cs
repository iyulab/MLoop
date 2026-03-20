using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

public class PredictCommandTests
{
    #region QuoteCsvField

    [Fact]
    public void QuoteCsvField_PlainValue_ReturnsUnquoted()
    {
        Assert.Equal("hello", PredictCommand.QuoteCsvField("hello"));
    }

    [Fact]
    public void QuoteCsvField_WithComma_ReturnsQuoted()
    {
        Assert.Equal("\"hello,world\"", PredictCommand.QuoteCsvField("hello,world"));
    }

    [Fact]
    public void QuoteCsvField_WithQuote_EscapesAndQuotes()
    {
        Assert.Equal("\"say \"\"hi\"\"\"", PredictCommand.QuoteCsvField("say \"hi\""));
    }

    [Fact]
    public void QuoteCsvField_WithNewline_ReturnsQuoted()
    {
        Assert.Equal("\"line1\nline2\"", PredictCommand.QuoteCsvField("line1\nline2"));
    }

    [Fact]
    public void QuoteCsvField_EmptyString_ReturnsEmpty()
    {
        Assert.Equal("", PredictCommand.QuoteCsvField(""));
    }

    [Fact]
    public void QuoteCsvField_NumericValue_ReturnsUnquoted()
    {
        Assert.Equal("42.5", PredictCommand.QuoteCsvField("42.5"));
    }

    #endregion
}
