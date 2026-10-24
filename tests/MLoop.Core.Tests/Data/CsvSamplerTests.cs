using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// Tests for CsvSampler — general-purpose CSV file sampling.
/// </summary>
public class CsvSamplerTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly CsvSampler _sampler;

    public CsvSamplerTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _sampler = new CsvSampler();
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
            Directory.Delete(_tempDirectory, recursive: true);
    }

    private string CreateCsvFile(string name, string content)
    {
        var path = Path.Combine(_tempDirectory, name);
        File.WriteAllText(path, content);
        return path;
    }

    #region Head Strategy

    [Fact]
    public async Task SampleAsync_HeadStrategy_ReturnsFirstNRows()
    {
        var input = CreateCsvFile("input.csv", """
            Name,Age,City
            Alice,25,Seoul
            Bob,30,Busan
            Charlie,35,Incheon
            Dave,40,Daegu
            Eve,28,Gwangju
            """);
        var output = Path.Combine(_tempDirectory, "output.csv");

        var result = await _sampler.SampleAsync(input, output, 3, CsvSamplingStrategy.Head);

        Assert.Equal(3, result.SampledCount);
        Assert.Equal(5, result.TotalRows);
        Assert.Equal(CsvSamplingStrategy.Head, result.StrategyUsed);
        Assert.True(File.Exists(output));

        var csv = new CsvHelperImpl();
        var data = await csv.ReadAsync(output);
        Assert.Equal(3, data.Count);
        Assert.Equal("Alice", data[0]["Name"]);
        Assert.Equal("Bob", data[1]["Name"]);
        Assert.Equal("Charlie", data[2]["Name"]);
    }

    [Fact]
    public async Task SampleAsync_HeadStrategy_MoreRowsThanAvailable_ReturnsAll()
    {
        var input = CreateCsvFile("small.csv", """
            A,B
            1,x
            2,y
            """);
        var output = Path.Combine(_tempDirectory, "output.csv");

        var result = await _sampler.SampleAsync(input, output, 100, CsvSamplingStrategy.Head);

        Assert.Equal(2, result.SampledCount);
        Assert.Equal(2, result.TotalRows);
    }

    #endregion

    #region Random Strategy

    [Fact]
    public async Task SampleAsync_RandomStrategy_ReturnsCorrectCount()
    {
        var input = CreateCsvFile("input.csv", """
            Name,Age
            Alice,25
            Bob,30
            Charlie,35
            Dave,40
            Eve,28
            """);
        var output = Path.Combine(_tempDirectory, "output.csv");

        var result = await _sampler.SampleAsync(input, output, 3, CsvSamplingStrategy.Random);

        Assert.Equal(3, result.SampledCount);
        Assert.Equal(5, result.TotalRows);

        var csv = new CsvHelperImpl();
        var data = await csv.ReadAsync(output);
        Assert.Equal(3, data.Count);
        // All rows should have valid Name and Age columns
        Assert.All(data, row =>
        {
            Assert.True(row.ContainsKey("Name"));
            Assert.True(row.ContainsKey("Age"));
            Assert.False(string.IsNullOrEmpty(row["Name"]));
        });
    }

    [Fact]
    public async Task SampleAsync_RandomStrategy_NoDuplicates()
    {
        var input = CreateCsvFile("input.csv", """
            Id,Value
            1,a
            2,b
            3,c
            4,d
            5,e
            6,f
            7,g
            8,h
            9,i
            10,j
            """);
        var output = Path.Combine(_tempDirectory, "output.csv");

        var result = await _sampler.SampleAsync(input, output, 5, CsvSamplingStrategy.Random);

        var csv = new CsvHelperImpl();
        var data = await csv.ReadAsync(output);
        var ids = data.Select(r => r["Id"]).ToList();
        Assert.Equal(ids.Distinct().Count(), ids.Count);
    }

    #endregion

    #region Stratified Strategy

    [Fact]
    public async Task SampleAsync_StratifiedStrategy_PreservesDistribution()
    {
        // 60% class A, 40% class B
        var lines = new List<string> { "Feature,Label" };
        for (int i = 0; i < 60; i++) lines.Add($"{i},A");
        for (int i = 0; i < 40; i++) lines.Add($"{i + 60},B");
        var input = CreateCsvFile("input.csv", string.Join("\n", lines));
        var output = Path.Combine(_tempDirectory, "output.csv");

        var result = await _sampler.SampleAsync(input, output, 20,
            CsvSamplingStrategy.Stratified, labelColumn: "Label");

        Assert.Equal(20, result.SampledCount);

        var csv = new CsvHelperImpl();
        var data = await csv.ReadAsync(output);
        var classA = data.Count(r => r["Label"] == "A");
        var classB = data.Count(r => r["Label"] == "B");

        // Should be roughly 60/40 (12/8)
        Assert.InRange(classA, 10, 14);
        Assert.InRange(classB, 6, 10);
    }

    [Fact]
    public async Task SampleAsync_StratifiedStrategy_RequiresLabel()
    {
        var input = CreateCsvFile("input.csv", "A,B\n1,2\n");
        var output = Path.Combine(_tempDirectory, "output.csv");

        await Assert.ThrowsAsync<ArgumentException>(() =>
            _sampler.SampleAsync(input, output, 1, CsvSamplingStrategy.Stratified));
    }

    [Fact]
    public async Task SampleAsync_StratifiedStrategy_NonexistentLabelColumn_ThrowsWithMessage()
    {
        var lines = new List<string> { "Feature,Label" };
        for (int i = 0; i < 20; i++) lines.Add($"{i},A");
        var input = CreateCsvFile("bad_label.csv", string.Join("\n", lines));
        var output = Path.Combine(_tempDirectory, "output.csv");

        var ex = await Assert.ThrowsAsync<ArgumentException>(() =>
            _sampler.SampleAsync(input, output, 5, CsvSamplingStrategy.Stratified, labelColumn: "NonExistent"));

        Assert.Contains("NonExistent", ex.Message);
        Assert.Contains("not found", ex.Message);
    }

    [Fact]
    public async Task SampleAsync_StratifiedStrategy_SingleClass_ReturnsCorrectCount()
    {
        var lines = new List<string> { "Feature,Label" };
        for (int i = 0; i < 10; i++) lines.Add($"{i},OnlyClass");
        var input = CreateCsvFile("input.csv", string.Join("\n", lines));
        var output = Path.Combine(_tempDirectory, "output.csv");

        var result = await _sampler.SampleAsync(input, output, 5,
            CsvSamplingStrategy.Stratified, labelColumn: "Label");

        Assert.Equal(5, result.SampledCount);
    }

    #endregion

    #region Streaming draw

    private string NumberedCsv(int rows, Func<int, string> label)
    {
        var lines = new List<string> { "Id,Label" };
        for (var i = 0; i < rows; i++) lines.Add($"{i},{label(i)}");
        return CreateCsvFile($"numbered-{rows}.csv", string.Join("\n", lines));
    }

    private static async Task<List<int>> IdsAsync(string path) =>
        [.. (await new CsvHelperImpl().ReadAsync(path)).Select(r => int.Parse(r["Id"], System.Globalization.CultureInfo.InvariantCulture))];

    [Theory]
    [InlineData(CsvSamplingStrategy.Random)]
    [InlineData(CsvSamplingStrategy.Stratified)]
    public async Task SampleAsync_KeepsTheRowsInFileOrder(CsvSamplingStrategy strategy)
    {
        var input = NumberedCsv(500, i => i % 3 == 0 ? "a" : "b");
        var output = Path.Combine(_tempDirectory, "ordered.csv");

        await new CsvSampler(seed: 1).SampleAsync(input, output, 50, strategy, labelColumn: "Label");

        var ids = await IdsAsync(output);
        Assert.Equal(50, ids.Count);
        Assert.Equal(ids.Order().ToList(), ids);
    }

    [Fact]
    public async Task SampleAsync_SameSeed_SameSample_OtherSeed_Other()
    {
        var input = NumberedCsv(1000, _ => "x");
        async Task<List<int>> Draw(int seed, string name)
        {
            var output = Path.Combine(_tempDirectory, name);
            await new CsvSampler(seed).SampleAsync(input, output, 20);
            return await IdsAsync(output);
        }

        Assert.Equal(await Draw(7, "a.csv"), await Draw(7, "b.csv"));
        Assert.NotEqual(await Draw(7, "c.csv"), await Draw(8, "d.csv"));
    }

    [Fact]
    public async Task SampleAsync_Random_DrawsFromTheWholeFile()
    {
        // A reservoir that kept only early rows would leave the second half empty.
        var input = NumberedCsv(10_000, _ => "x");
        var output = Path.Combine(_tempDirectory, "spread.csv");

        await new CsvSampler(seed: 3).SampleAsync(input, output, 1000);

        var late = (await IdsAsync(output)).Count(id => id >= 5000);
        Assert.InRange(late, 400, 600);
    }

    [Fact]
    public async Task SampleAsync_Head_CountsTheWholeFile()
    {
        var input = NumberedCsv(300, _ => "x");
        var output = Path.Combine(_tempDirectory, "head.csv");

        var result = await _sampler.SampleAsync(input, output, 10, CsvSamplingStrategy.Head);

        Assert.Equal(300, result.TotalRows);
        Assert.Equal(Enumerable.Range(0, 10).ToList(), await IdsAsync(output));
    }

    [Fact]
    public void Quotas_AddUpToTheSample_AndGiveEveryClassARow()
    {
        var quotas = CsvSampler.Quotas(new Dictionary<string, int> { ["rare"] = 3, ["common"] = 997 }, 10);

        Assert.Equal(10, quotas.Values.Sum());
        Assert.Equal(1, quotas["rare"]);

        // Large files: shares are proportions of the file, not products that overflow (100,000 × 1,870,921 > int).
        var large = CsvSampler.Quotas(new Dictionary<string, int> { ["0"] = 1_870_921, ["1"] = 129_079 }, 100_000);
        Assert.Equal(100_000, large.Values.Sum());
        Assert.InRange(large["1"], 6_453, 6_455);

        // Rounding leaves rows over; they go to classes that still have rows to give.
        var tight = CsvSampler.Quotas(new Dictionary<string, int> { ["a"] = 2, ["b"] = 2, ["c"] = 2 }, 5);
        Assert.Equal(5, tight.Values.Sum());
        Assert.All(tight, q => Assert.InRange(q.Value, 1, 2));
    }

    #endregion

    #region Edge Cases

    [Fact]
    public async Task SampleAsync_FileNotFound_ThrowsFileNotFound()
    {
        var output = Path.Combine(_tempDirectory, "output.csv");
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _sampler.SampleAsync("nonexistent.csv", output, 10));
    }

    [Fact]
    public async Task SampleAsync_EmptyFile_ThrowsInvalidOperation()
    {
        var input = CreateCsvFile("empty.csv", "A,B\n");
        var output = Path.Combine(_tempDirectory, "output.csv");

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            _sampler.SampleAsync(input, output, 10));
    }

    [Fact]
    public async Task SampleAsync_ZeroRows_ThrowsArgumentOutOfRange()
    {
        var input = CreateCsvFile("input.csv", "A,B\n1,2\n");
        var output = Path.Combine(_tempDirectory, "output.csv");

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            _sampler.SampleAsync(input, output, 0));
    }

    [Fact]
    public async Task SampleAsync_PreservesColumnOrder()
    {
        var input = CreateCsvFile("input.csv", """
            Zebra,Apple,Mango
            1,2,3
            4,5,6
            """);
        var output = Path.Combine(_tempDirectory, "output.csv");

        await _sampler.SampleAsync(input, output, 2, CsvSamplingStrategy.Head);

        var csv = new CsvHelperImpl();
        var data = await csv.ReadAsync(output);
        var keys = data[0].Keys.ToList();
        Assert.Equal("Zebra", keys[0]);
        Assert.Equal("Apple", keys[1]);
        Assert.Equal("Mango", keys[2]);
    }

    [Fact]
    public async Task SampleAsync_CreatesOutputDirectory()
    {
        var input = CreateCsvFile("input.csv", "A,B\n1,2\n3,4\n");
        var output = Path.Combine(_tempDirectory, "sub", "dir", "output.csv");

        var result = await _sampler.SampleAsync(input, output, 2, CsvSamplingStrategy.Head);

        Assert.True(File.Exists(output));
        Assert.Equal(2, result.SampledCount);
    }

    [Fact]
    public async Task SampleAsync_SingleRow_ReturnsOne()
    {
        var input = CreateCsvFile("single.csv", "X,Y\n42,hello\n");
        var output = Path.Combine(_tempDirectory, "output.csv");

        var result = await _sampler.SampleAsync(input, output, 10, CsvSamplingStrategy.Random);

        Assert.Equal(1, result.SampledCount);
        Assert.Equal(1, result.TotalRows);
    }

    [Fact]
    public async Task SampleAsync_CsvWithSpecialChars_PreservesValues()
    {
        var input = CreateCsvFile("special.csv", "Name,Description\nAlice,\"Has a, comma\"\nBob,\"Has \"\"quotes\"\"\"\n");
        var output = Path.Combine(_tempDirectory, "output.csv");

        var result = await _sampler.SampleAsync(input, output, 2, CsvSamplingStrategy.Head);

        var csv = new CsvHelperImpl();
        var data = await csv.ReadAsync(output);
        Assert.Equal(2, data.Count);
        Assert.Equal("Has a, comma", data[0]["Description"]);
        Assert.Contains("quotes", data[1]["Description"]);
    }

    #endregion

    #region Default Output Path

    [Fact]
    public async Task SampleAsync_ReturnsFullOutputPath()
    {
        var input = CreateCsvFile("test.csv", "A\n1\n2\n");
        var output = Path.Combine(_tempDirectory, "result.csv");

        var result = await _sampler.SampleAsync(input, output, 1);

        Assert.Equal(Path.GetFullPath(output), result.OutputPath);
    }

    #endregion
}
