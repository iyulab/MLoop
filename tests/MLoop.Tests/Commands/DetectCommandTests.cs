using MLoop.CLI.Commands;
using Xunit;

namespace MLoop.Tests.Commands;

[Collection("FileSystem")]
public class DetectCommandTests : IDisposable
{
    private readonly List<string> _tempFiles = new();

    public void Dispose()
    {
        foreach (var f in _tempFiles)
        {
            try { if (File.Exists(f)) File.Delete(f); } catch { }
        }
    }

    private string WriteCsv(params string[] lines)
    {
        var path = Path.Combine(Path.GetTempPath(), $"mloop-detect-{Guid.NewGuid():N}.csv");
        File.WriteAllLines(path, lines);
        _tempFiles.Add(path);
        return path;
    }

    [Fact]
    public async Task LoadSeries_SingleColumn_AutoSelects()
    {
        var csv = WriteCsv("Value", "1.5", "2.5", "3.5");

        var result = await DetectCommand.LoadSeriesAsync(csv, column: null, jsonOutput: true);

        Assert.NotNull(result);
        Assert.Equal("Value", result.Value.Column);
        Assert.Equal(new[] { 1.5, 2.5, 3.5 }, result.Value.Values);
    }

    [Fact]
    public async Task LoadSeries_MultiColumn_WithoutColumn_Fails()
    {
        var csv = WriteCsv("A,B", "1,2", "3,4");

        var result = await DetectCommand.LoadSeriesAsync(csv, column: null, jsonOutput: true);

        Assert.Null(result);
    }

    [Fact]
    public async Task LoadSeries_ColumnName_IsCaseInsensitive()
    {
        var csv = WriteCsv("Temp,Pressure", "20.5,1.0", "21.0,1.1");

        var result = await DetectCommand.LoadSeriesAsync(csv, column: "temp", jsonOutput: true);

        Assert.NotNull(result);
        Assert.Equal("Temp", result.Value.Column);
        Assert.Equal(new[] { 20.5, 21.0 }, result.Value.Values);
    }

    [Fact]
    public async Task LoadSeries_MissingColumn_Fails()
    {
        var csv = WriteCsv("A,B", "1,2");

        var result = await DetectCommand.LoadSeriesAsync(csv, column: "C", jsonOutput: true);

        Assert.Null(result);
    }

    [Fact]
    public async Task LoadSeries_NonNumericValue_Fails()
    {
        // SR-CNN needs a contiguous numeric series — a gap must be an input error, not skipped.
        var csv = WriteCsv("Value", "1.0", "oops", "3.0");

        var result = await DetectCommand.LoadSeriesAsync(csv, column: null, jsonOutput: true);

        Assert.Null(result);
    }

    [Fact]
    public void Create_ParsesArgumentsAndOptions()
    {
        var command = DetectCommand.Create();

        var parse = command.Parse(new[]
        {
            "data.csv", "--column", "Temp", "--threshold", "0.5",
            "--sensitivity", "80", "--period", "12", "--output", "out.csv", "--json"
        });

        Assert.Empty(parse.Errors);
    }

    [Fact]
    public void Create_RequiresDataFileArgument()
    {
        var command = DetectCommand.Create();

        var parse = command.Parse(Array.Empty<string>());

        Assert.NotEmpty(parse.Errors);
    }
}
