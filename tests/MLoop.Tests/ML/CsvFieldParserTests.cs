using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.ML;

public class CsvFieldParserTests
{
    #region ParseFields - Basic

    [Fact]
    public void ParseFields_SimpleValues_ReturnsSplit()
    {
        var result = CsvFieldParser.ParseFields("a,b,c");

        Assert.Equal(3, result.Length);
        Assert.Equal("a", result[0]);
        Assert.Equal("b", result[1]);
        Assert.Equal("c", result[2]);
    }

    [Fact]
    public void ParseFields_SingleField_ReturnsSingleElement()
    {
        var result = CsvFieldParser.ParseFields("hello");

        Assert.Single(result);
        Assert.Equal("hello", result[0]);
    }

    [Fact]
    public void ParseFields_EmptyString_ReturnsSingleEmpty()
    {
        var result = CsvFieldParser.ParseFields("");

        Assert.Single(result);
        Assert.Equal("", result[0]);
    }

    [Fact]
    public void ParseFields_EmptyFields_PreservesEmpty()
    {
        var result = CsvFieldParser.ParseFields(",,,");

        Assert.Equal(4, result.Length);
        Assert.All(result, f => Assert.Equal("", f));
    }

    #endregion

    #region ParseFields - Quoted Fields

    [Fact]
    public void ParseFields_QuotedWithComma_PreservesComma()
    {
        var result = CsvFieldParser.ParseFields("\"a,b\",c,d");

        Assert.Equal(3, result.Length);
        Assert.Equal("a,b", result[0]);
        Assert.Equal("c", result[1]);
        Assert.Equal("d", result[2]);
    }

    [Fact]
    public void ParseFields_EscapedQuotes_UnescapesCorrectly()
    {
        var result = CsvFieldParser.ParseFields("\"he said \"\"hi\"\"\",done");

        Assert.Equal(2, result.Length);
        Assert.Equal("he said \"hi\"", result[0]);
        Assert.Equal("done", result[1]);
    }

    [Fact]
    public void ParseFields_QuotedField_TrimsQuotes()
    {
        var result = CsvFieldParser.ParseFields("\"hello\",world");

        Assert.Equal(2, result.Length);
        Assert.Equal("hello", result[0]);
        Assert.Equal("world", result[1]);
    }

    [Fact]
    public void ParseFields_MixedQuotedAndUnquoted_ParsesCorrectly()
    {
        var result = CsvFieldParser.ParseFields("name,\"Seoul, Korea\",42");

        Assert.Equal(3, result.Length);
        Assert.Equal("name", result[0]);
        Assert.Equal("Seoul, Korea", result[1]);
        Assert.Equal("42", result[2]);
    }

    #endregion

    #region FormatLine

    [Fact]
    public void FormatLine_SimpleValues_JoinsWithComma()
    {
        var result = CsvFieldParser.FormatLine(new[] { "a", "b", "c" });

        Assert.Equal("a,b,c", result);
    }

    [Fact]
    public void FormatLine_ValueWithComma_QuotesField()
    {
        var result = CsvFieldParser.FormatLine(new[] { "hello, world", "test" });

        Assert.Equal("\"hello, world\",test", result);
    }

    [Fact]
    public void FormatLine_ValueWithQuotes_EscapesQuotes()
    {
        var result = CsvFieldParser.FormatLine(new[] { "say \"hi\"", "ok" });

        Assert.Equal("\"say \"\"hi\"\"\",ok", result);
    }

    [Fact]
    public void FormatLine_ValueWithNewline_QuotesField()
    {
        var result = CsvFieldParser.FormatLine(new[] { "line1\nline2", "test" });

        Assert.Equal("\"line1\nline2\",test", result);
    }

    #endregion

    #region Roundtrip

    [Theory]
    [InlineData("simple,fields,here")]
    [InlineData("\"quoted,field\",normal")]
    [InlineData("\"has \"\"quotes\"\"\",ok")]
    public void ParseFields_FormatLine_Roundtrip(string line)
    {
        var parsed = CsvFieldParser.ParseFields(line);
        var formatted = CsvFieldParser.FormatLine(parsed);
        var reparsed = CsvFieldParser.ParseFields(formatted);

        Assert.Equal(parsed, reparsed);
    }

    [Fact]
    public void FormatLine_ParseFields_Roundtrip_WithSpecialChars()
    {
        var fields = new[] { "Seoul, Korea", "he said \"hello\"", "normal", "line1\nline2" };
        var formatted = CsvFieldParser.FormatLine(fields);
        var parsed = CsvFieldParser.ParseFields(formatted);

        Assert.Equal(fields, parsed);
    }

    #endregion

    #region Korean Text

    [Fact]
    public void ParseFields_KoreanText_ParsesCorrectly()
    {
        var result = CsvFieldParser.ParseFields("이름,\"서울, 대한민국\",나이");

        Assert.Equal(3, result.Length);
        Assert.Equal("이름", result[0]);
        Assert.Equal("서울, 대한민국", result[1]);
        Assert.Equal("나이", result[2]);
    }

    #endregion
}
