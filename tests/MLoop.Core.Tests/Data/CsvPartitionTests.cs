using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

public sealed class CsvPartitionTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop-partition-" + Guid.NewGuid().ToString("N"));

    public CsvPartitionTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private string Write(string content)
    {
        var path = Path.Combine(_dir, "data.csv");
        File.WriteAllText(path, content);
        return path;
    }

    [Fact]
    public async Task EachValueBecomesAFileWithTheHeader_LargestFirst()
    {
        var path = Write("a,site,text\n1,busan,\"x, y\"\n2,seoul,z\n3,busan,w\n4,,v\n");

        var parts = await CsvPartition.ByColumnAsync(path, "site", Path.Combine(_dir, "out"));

        Assert.Equal(["busan", "(empty)", "seoul"], parts.Select(p => p.Value).ToArray()); // ties in ordinal order
        Assert.Equal(2, parts[0].Rows);
        var busan = File.ReadAllLines(parts[0].Path);
        Assert.Equal("a,site,text", busan[0]);
        Assert.Equal("1,busan,\"x, y\"", busan[1]); // a quoted field stays one field
    }

    [Fact]
    public async Task AMissingColumnNamesTheColumnsThereAre()
    {
        var path = Write("a,b\n1,2\n");

        var ex = await Assert.ThrowsAsync<ArgumentException>(() => CsvPartition.ByColumnAsync(path, "site", _dir));

        Assert.Contains("a, b", ex.Message);
    }

    [Fact]
    public async Task TooManyValuesIsRefused()
    {
        var path = Write("id\n" + string.Join("\n", Enumerable.Range(0, 5)) + "\n");

        await Assert.ThrowsAsync<InvalidOperationException>(() => CsvPartition.ByColumnAsync(path, "id", _dir, maxParts: 4));
    }
}
