using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

public class PredictCommandTests
{
    #region QuoteCsvField

    [Fact]
    public void QuoteCsvField_PlainValue_ReturnsUnquoted()
    {
        Assert.Equal("hello", MLoop.Core.Prediction.CsvFieldParser.Quote("hello"));
    }

    [Fact]
    public void QuoteCsvField_WithComma_ReturnsQuoted()
    {
        Assert.Equal("\"hello,world\"", MLoop.Core.Prediction.CsvFieldParser.Quote("hello,world"));
    }

    [Fact]
    public void QuoteCsvField_WithQuote_EscapesAndQuotes()
    {
        Assert.Equal("\"say \"\"hi\"\"\"", MLoop.Core.Prediction.CsvFieldParser.Quote("say \"hi\""));
    }

    [Fact]
    public void QuoteCsvField_WithNewline_ReturnsQuoted()
    {
        Assert.Equal("\"line1\nline2\"", MLoop.Core.Prediction.CsvFieldParser.Quote("line1\nline2"));
    }

    [Fact]
    public void QuoteCsvField_EmptyString_ReturnsEmpty()
    {
        Assert.Equal("", MLoop.Core.Prediction.CsvFieldParser.Quote(""));
    }

    [Fact]
    public void QuoteCsvField_NumericValue_ReturnsUnquoted()
    {
        Assert.Equal("42.5", MLoop.Core.Prediction.CsvFieldParser.Quote("42.5"));
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

    [Fact]
    public void Rows_without_an_answer_are_not_a_class_of_the_distribution()
    {
        // A time-series detector cannot judge the first rows of a series, so their label is empty;
        // counted as a class they read as a third answer beside False and True.
        var lines = new[] { "Value,PredictedLabel,Score" }
            .Concat(Enumerable.Repeat("1,,", 3))
            .Concat(Enumerable.Repeat("1,False,0.1", 6))
            .Concat(Enumerable.Repeat("9,True,0.9", 2));

        var distribution = PredictCommand.ReadPredictionDistribution(lines);

        Assert.NotNull(distribution);
        Assert.Equal(3, distribution!.Unanswered);
        Assert.Equal(["False", "True"], distribution.Classes.Select(c => c.Key));
        Assert.Equal([6, 2], distribution.Classes.Select(c => c.Value));
    }

    [Fact]
    public void A_single_answered_class_has_no_distribution_even_with_unanswered_rows()
    {
        var lines = new[] { "PredictedLabel", "", "", "yes", "yes" };

        Assert.Null(PredictCommand.ReadPredictionDistribution(lines));
    }

    [Fact]
    public void A_score_file_has_no_distribution()
    {
        var lines = new[] { "x,Score", "1,0.5", "2,0.7" };

        Assert.Null(PredictCommand.ReadPredictionDistribution(lines));
    }
}
