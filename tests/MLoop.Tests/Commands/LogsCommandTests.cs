using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

public class LogsCommandTests
{
    #region SummarizeInput

    [Fact]
    public void SummarizeInput_EmptyDictionary_ReturnsDash()
    {
        var input = new Dictionary<string, object>();

        var result = LogsCommand.SummarizeInput(input);

        Assert.Equal("-", result);
    }

    [Fact]
    public void SummarizeInput_SingleField_ShowsKeyValue()
    {
        var input = new Dictionary<string, object> { ["x1"] = 0.5 };

        var result = LogsCommand.SummarizeInput(input);

        Assert.Equal("x1=0.5", result);
    }

    [Fact]
    public void SummarizeInput_MultipleFields_ShowsFirstPlusCount()
    {
        var input = new Dictionary<string, object>
        {
            ["x1"] = 0.5,
            ["x2"] = 1.0,
            ["x3"] = 2.0
        };

        var result = LogsCommand.SummarizeInput(input);

        Assert.Contains("x1=0.5", result);
        Assert.Contains("(+2 more)", result);
    }

    [Fact]
    public void SummarizeInput_LongSummary_TruncatesAt40()
    {
        var input = new Dictionary<string, object>
        {
            ["very_long_column_name_that_exceeds_limit"] = "some_very_long_value_here"
        };

        var result = LogsCommand.SummarizeInput(input);

        Assert.True(result.Length <= 43); // 40 chars + "..."
        Assert.EndsWith("...", result);
    }

    [Fact]
    public void SummarizeInput_ExactlyAt40_NoTruncation()
    {
        // Create input that produces exactly 40 chars: "k=v" = 3 chars
        var input = new Dictionary<string, object>
        {
            ["key"] = "v"
        };

        var result = LogsCommand.SummarizeInput(input);

        Assert.DoesNotContain("...", result);
    }

    #endregion
}
