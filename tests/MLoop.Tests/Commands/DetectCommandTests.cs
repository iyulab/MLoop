using MLoop.CLI.Commands;
using MLoop.Core.Detection;
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
    public async Task WriteCsv_EmitsControlAndMarginBandsSeparately()
    {
        // The two bands answer different questions (chartable limits vs the gate behind the verdict),
        // so neither may be dropped and neither may be labelled just "Lower/Upper".
        var path = Path.Combine(Path.GetTempPath(), $"mloop-detect-out-{Guid.NewGuid():N}.csv");
        _tempFiles.Add(path);

        var result = new OneShotAnomalyResult
        {
            Period = 0,
            ResidualSigma = 0.5,
            Points = new[]
            {
                new OneShotAnomalyPoint
                {
                    Index = 0, Value = 10.0, IsAnomaly = false, Score = 0.1, ExpectedValue = 9.0,
                    ControlLower = 7.5, ControlUpper = 10.5, MarginLower = 8.9, MarginUpper = 9.1,
                }
            }
        };

        await DetectCommand.WriteCsvAsync(path, result);

        var lines = await File.ReadAllLinesAsync(path);
        Assert.Equal(DetectCommand.CsvHeader, lines[0]);
        Assert.Contains("ControlLower,ControlUpper,MarginLower,MarginUpper", lines[0]);
        Assert.Equal(lines[0].Split(',').Length, lines[1].Split(',').Length);
        Assert.Contains("7.5,10.5,8.9,9.1", lines[1]);
    }

    [Fact]
    public void Create_RequiresDataFileArgument()
    {
        var command = DetectCommand.Create();

        var parse = command.Parse(Array.Empty<string>());

        Assert.NotEmpty(parse.Errors);
    }
}
