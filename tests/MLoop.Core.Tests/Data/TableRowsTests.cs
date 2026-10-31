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

    [Fact]
    public async Task ARecord_IsCountedOnce_WhateverLinesItsQuotedFieldsTake_AndABlankLineIsNone()
    {
        var path = Path.Combine(Path.GetTempPath(), "mloop-table-rows-" + Guid.NewGuid().ToString("N") + ".csv");
        try
        {
            await File.WriteAllTextAsync(path, "a,note\n1,\"first\nsecond\nthird\"\n\n2,plain\r\n3,\"x\r\ny\"\n\n");
            Assert.Equal(3, await TableRows.CountAsync(path));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
