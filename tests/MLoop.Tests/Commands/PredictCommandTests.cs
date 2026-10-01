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

    #region BuildMissingDataFileMessage

    [Fact]
    public void BuildMissingDataFileMessage_TrulyMissing_SaysNotFound()
    {
        var missing = Path.Combine(Path.GetTempPath(), $"mloop-missing-{Guid.NewGuid()}.csv");
        var msg = PredictCommand.BuildMissingDataFileMessage(missing);

        Assert.Contains("not found", msg, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void BuildMissingDataFileMessage_DirectoryForTabular_SaysDirectoryNotFile()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"mloop-dir-{Guid.NewGuid()}");
        Directory.CreateDirectory(dir);
        try
        {
            var msg = PredictCommand.BuildMissingDataFileMessage(dir);

            // Honest diagnosis: the path exists but is a directory, not a missing file.
            Assert.Contains("directory", msg, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("not found", msg, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.Delete(dir);
        }
    }

    #endregion

    [Theory]
    [InlineData(new[] { "qid", "f1", "Score" }, -1)]
    [InlineData(new[] { "x", "Score" }, -1)]
    [InlineData(new[] { "x", "PredictedLabel", "Score.a" }, 1)]
    public void DistributionColumn_CountsOnlyAPredictedClass(string[] headers, int expected)
    {
        // A ranking prediction file's distribution counted its first column — the query ids.
        Assert.Equal(expected, PredictCommand.DistributionColumn(headers));
    }
}
