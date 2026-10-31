using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

public class TableRowsTests
{
    [Fact]
    public void ABlankLine_IsNoRowOfATable()
    {
        Assert.False(TableRows.IsRow("", 2));
        Assert.False(TableRows.IsRow("  ", 3));
        Assert.True(TableRows.IsRow("1,2", 2));
        Assert.True(TableRows.IsRow(",", 2));
    }

    [Fact]
    public void ABlankLine_IsARecordOfAFileOfOneColumn()
    {
        Assert.True(TableRows.IsRow("", 1));
        Assert.True(TableRows.IsRow("  ", 1));
        Assert.True(TableRows.IsRow("1", 1));
    }
}
