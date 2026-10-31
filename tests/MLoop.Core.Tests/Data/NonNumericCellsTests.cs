using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// Which cells of a numeric column the loader cannot read as numbers — counted over the whole file, because the loader
/// types a column from its start and loads the rest as missing values without a word.
/// </summary>
public class NonNumericCellsTests
{
    [Theory]
    [InlineData("3.14", true)]
    [InlineData("-2", true)]
    [InlineData("1e-3", true)]
    [InlineData("  7 ", true)]
    [InlineData("", true)]
    [InlineData("NaN", true)]
    [InlineData("NA", true)]
    [InlineData("?", true)]
    [InlineData("N/A", false)]
    [InlineData("--", false)]
    [InlineData("unknown", false)]
    [InlineData("1,234.5", false)]
    [InlineData("12%", false)]
    public void ACellIsANumberWhenTheLoaderReadsItAsOneOrAsMissing(string cell, bool reads)
        => Assert.Equal(reads, NonNumericCells.ReadsAsNumber(cell));

    [Fact]
    public void ItCountsPerColumnAndRemembersTheFirstOne()
    {
        var census = new NonNumericCells([true, false, true]);

        census.Add(["1", "x", "5"], line: 2);
        census.Add(["N/A", "y", "6"], line: 3);
        census.Add(["--", "z", "?"], line: 4);

        Assert.Equal((2L, 3L, "N/A"), census.Of(0));
        Assert.Equal((0L, 0L, (string?)null), census.Of(1)); // not asked: the layout loads it as text
        Assert.Equal((0L, 0L, (string?)null), census.Of(2));
    }

    [Fact]
    public void ARowShorterThanTheHeaderIsCountedAsFarAsItGoes()
    {
        var census = new NonNumericCells([true, true]);

        census.Add(["N/A"], line: 2);

        Assert.Equal((1L, 2L, "N/A"), census.Of(0));
        Assert.Equal(0L, census.Of(1).Count);
    }

    private static NonNumericCells Counted(int rows, int stray)
    {
        var census = new NonNumericCells([true]);
        for (var i = 0; i < rows; i++)
            census.Add([i < stray ? "N/A" : "1.5"], line: i + 2);
        return census;
    }

    [Theory]
    [InlineData(1000, 1, false)]   // one in a thousand: fewer than five and under half a percent
    [InlineData(1000, 4, false)]   // 0.4 %: still under both
    [InlineData(1000, 5, true)]    // five cells are worth a word whatever the file's size
    [InlineData(100, 1, true)]     // one in a hundred is a share of a small file
    [InlineData(200, 1, true)]     // exactly half a percent
    [InlineData(100, 0, false)]
    public void ItSpeaksOnlyWhenTheCellsAreEnoughToMatter(int rows, int stray, bool speaks)
        => Assert.Equal(speaks, Counted(rows, stray).Describe(0, "reading") is not null);

    [Fact]
    public void TheSentenceNamesTheColumnTheCountTheShareAndWhereItFirstHappens()
    {
        var census = new NonNumericCells([true]);
        census.Add(["1"], line: 2);
        for (var line = 3; line <= 12; line++)
            census.Add([line % 2 == 0 ? "N/A" : "--"], line);

        var said = census.Describe(0, "reading");

        Assert.NotNull(said);
        Assert.Contains("'reading'", said, StringComparison.Ordinal);
        Assert.Contains("10 of 11", said, StringComparison.Ordinal);
        Assert.Contains("(first: '--' on line 3)", said, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnNobodyAskedAboutHasNothingToSay()
    {
        var census = new NonNumericCells([false]);
        census.Add(["N/A"], line: 2);

        Assert.Null(census.Describe(0, "name"));
    }

    [Theory]
    [InlineData(100, 1, true)]    // numbers with one cell that is not
    [InlineData(100, 10, true)]   // nine in ten are numbers
    [InlineData(100, 11, false)]  // fewer than nine in ten
    [InlineData(100, 0, false)]   // nothing that is not a number: a number column
    [InlineData(4, 1, false)]     // too few values to call it a habit
    public void AColumnIsMostlyNumbersWhenNineInTenNonEmptyCellsAre(int rows, int stray, bool mostly)
        => Assert.Equal(mostly, Counted(rows, stray).MostlyNumbers(0));

    [Fact]
    public void EmptyCellsAreNeitherNumbersNorTheirContrary()
    {
        var census = new NonNumericCells([true]);
        for (var i = 0; i < 50; i++)
            census.Add([""], line: i + 2);
        census.Add(["N/A"], line: 52);

        Assert.False(census.MostlyNumbers(0)); // one non-empty value
    }
}
