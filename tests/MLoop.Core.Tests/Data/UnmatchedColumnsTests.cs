using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// A type override that names no column of the data does nothing and says nothing; the training run uses this to say it.
/// </summary>
public class UnmatchedColumnsTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("mloop-unmatched-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* temp */ }
    }

    private string Csv(string content)
    {
        var path = Path.Combine(_dir, "data.csv");
        File.WriteAllText(path, content);
        return path;
    }

    [Fact]
    public void ANameTheFileLacks_IsReported()
    {
        var path = Csv("id,x,notes\n1,2,a\n");

        Assert.Equal(["typo"], CsvDataLoader.UnmatchedColumns(path, ["x", "typo"]));
    }

    [Fact]
    public void NamesCompareAsTheColumnRolesCompareThem_TrimmedAndIgnoringCase()
    {
        var path = Csv("id,Customer ID,notes\n1,2,a\n");

        Assert.Empty(CsvDataLoader.UnmatchedColumns(path, ["ID", " customer id ", "NOTES"]));
    }

    [Fact]
    public void AQuotedHeaderWithAComma_IsOneColumn()
    {
        var path = Csv("\"Last, First\",age\nA,3\n");

        Assert.Empty(CsvDataLoader.UnmatchedColumns(path, ["Last, First"]));
        Assert.Equal(["Last"], CsvDataLoader.UnmatchedColumns(path, ["Last"]));
    }

    [Fact]
    public void AFileWithNoHeader_ReportsNothing_BecauseItCannotTell()
    {
        var path = Csv("");

        Assert.Empty(CsvDataLoader.UnmatchedColumns(path, ["x"]));
    }

    [Fact]
    public void ARepeatedName_IsReportedOnce()
    {
        var path = Csv("a,b\n1,2\n");

        Assert.Equal(["z"], CsvDataLoader.UnmatchedColumns(path, ["z", "Z", "a"]));
    }
}
