using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

public class FeedbackCommandTests
{
    #region ValuesMatch

    [Fact]
    public void ValuesMatch_BothNull_ReturnsTrue()
    {
        Assert.True(FeedbackCommand.ValuesMatch(null, null));
    }

    [Fact]
    public void ValuesMatch_OneNull_ReturnsFalse()
    {
        Assert.False(FeedbackCommand.ValuesMatch("a", null));
        Assert.False(FeedbackCommand.ValuesMatch(null, "a"));
    }

    [Fact]
    public void ValuesMatch_SameString_ReturnsTrue()
    {
        Assert.True(FeedbackCommand.ValuesMatch("OK", "OK"));
    }

    [Fact]
    public void ValuesMatch_CaseInsensitiveString_ReturnsTrue()
    {
        Assert.True(FeedbackCommand.ValuesMatch("ok", "OK"));
        Assert.True(FeedbackCommand.ValuesMatch("Good", "GOOD"));
    }

    [Fact]
    public void ValuesMatch_DifferentStrings_ReturnsFalse()
    {
        Assert.False(FeedbackCommand.ValuesMatch("OK", "NG"));
    }

    [Fact]
    public void ValuesMatch_EqualIntegers_ReturnsTrue()
    {
        Assert.True(FeedbackCommand.ValuesMatch(42, 42));
    }

    [Fact]
    public void ValuesMatch_DifferentIntegers_ReturnsFalse()
    {
        Assert.False(FeedbackCommand.ValuesMatch(42, 43));
    }

    [Fact]
    public void ValuesMatch_EqualDoubles_ReturnsTrue()
    {
        Assert.True(FeedbackCommand.ValuesMatch(3.14, 3.14));
    }

    #endregion
}
