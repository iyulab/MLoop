using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// A column of numbers written for people (<c>1,234.5</c>, <c>12%</c>, <c>$3</c>) is read as words by the loader, and a
/// model that needed its magnitude has none. <see cref="FormattedNumbers"/> is the one place that says which text
/// columns those are — and leaves alone the columns of plain numbers, of words, and of too few values to judge.
/// </summary>
public class FormattedNumbersTests
{
    [Theory]
    [InlineData("1,234.5")]
    [InlineData("-1,234,567")]
    [InlineData("12%")]
    [InlineData("12.5 %")]
    [InlineData("$3")]
    [InlineData("€ 1,000.00")]
    [InlineData("₩5,000")]
    [InlineData("5,000원")]
    public void AFormattedNumber_IsFormatted(string value) => Assert.True(FormattedNumbers.IsFormatted(value));

    [Theory]
    [InlineData("1234.5")]
    [InlineData("-7")]
    [InlineData("hello")]
    [InlineData("12abc")]
    [InlineData("1,23")]        // not a thousands grouping
    [InlineData("")]
    public void APlainNumberOrWords_IsNotFormatted(string value) => Assert.False(FormattedNumbers.IsFormatted(value));

    [Fact]
    public void AColumnOfFormattedNumbers_Looks_AndNamesAnExample()
    {
        var column = Enumerable.Range(1, 40).Select(i => $"{i * 1000:N0}");

        Assert.True(FormattedNumbers.Looks(column, out var example));
        Assert.Equal("1,000", example);
    }

    [Fact]
    public void AColumnOfPlainNumbers_IsNotAskedAbout()
        => Assert.False(FormattedNumbers.Looks(Enumerable.Range(1, 40).Select(i => (i * 1000).ToString()), out _));

    [Fact]
    public void AColumnOfWords_IsNotFormattedNumbers()
        => Assert.False(FormattedNumbers.Looks(Enumerable.Range(1, 40).Select(i => $"item-{i}"), out _));

    [Fact]
    public void AFewValues_AreNotJudged()
        => Assert.False(FormattedNumbers.Looks(["1,000", "2,000", "3,000", "", " "], out _));

    [Fact]
    public void NinetyPercentIsTheBar_AndEmptyValuesDoNotCount()
    {
        var nine = Enumerable.Range(1, 9).Select(i => $"{i},000").Append("n/a").Concat(["", ""]);
        var eight = Enumerable.Range(1, 8).Select(i => $"{i},000").Concat(["n/a", "none"]);

        Assert.True(FormattedNumbers.Looks(nine, out _));
        Assert.False(FormattedNumbers.Looks(eight, out _));
    }

    [Fact]
    public void ExampleIsNull_WhenTheColumnIsNotJudged()
    {
        FormattedNumbers.Looks(["a", "b", "c", "d", "e", "f"], out var example);

        Assert.Null(example);
    }
}
