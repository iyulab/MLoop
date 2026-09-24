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
        // The transform needs a contiguous numeric series — a gap must be an input error, not skipped.
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
            "data.csv", "--column", "Temp", "--threshold", "4",
            "--sensitivity", "95", "--output", "out.csv", "--json"
        });

        Assert.Empty(parse.Errors);
    }

    [Fact]
    public async Task Period_IsRefusedWithTheReason()
    {
        // The score assumes no period, so a supplied one could not change the result; the option
        // used to feed a seasonal decomposition that collapsed detection when given the true period.
        // A script still passing it gets the reason, not a parser error about a stray "12".
        var csv = WriteCsv(["Value", .. Enumerable.Range(0, 24).Select(i => (i % 7).ToString())]);

        var (exitCode, stdout, stderr) = await CliRunner.RunAsync(["detect", csv, "--period", "12", "--json"]);

        Assert.Equal(1, exitCode);
        Assert.Contains("--period was removed", stdout);
        Assert.Contains("--period was removed", stderr);
    }

    [Theory]
    [InlineData("--threshold", "0")]
    [InlineData("--sensitivity", "100")]
    public async Task An_out_of_range_option_is_named_as_the_user_typed_it(string option, string value)
    {
        var csv = WriteCsv(["Value", .. Enumerable.Range(0, 24).Select(i => (i % 7).ToString())]);

        var (exitCode, _, stderr) = await CliRunner.RunAsync(["detect", csv, option, value]);

        Assert.Equal(1, exitCode);
        Assert.Contains(option + " must be", stderr);
    }

    [Fact]
    public async Task A_too_short_series_is_refused_without_a_library_parameter_name()
    {
        var csv = WriteCsv(["Value", "1", "2", "3"]);

        var (exitCode, _, stderr) = await CliRunner.RunAsync(["detect", csv]);

        Assert.Equal(1, exitCode);
        Assert.Contains("At least 12 observations", stderr);
        Assert.DoesNotContain("(Parameter", stderr);
    }

    [Fact]
    public async Task WriteCsv_EmitsOneRowPerPointWithTheControlBand()
    {
        var path = Path.Combine(Path.GetTempPath(), $"mloop-detect-out-{Guid.NewGuid():N}.csv");
        _tempFiles.Add(path);
        var series = SpikeSeries(length: 24, spikeAt: 12);

        await DetectCommand.WriteCsvAsync(path, Detect(series));

        var lines = await File.ReadAllLinesAsync(path);
        Assert.Equal(DetectCommand.CsvHeader, lines[0]);
        Assert.EndsWith("ExpectedValue,ControlLower,ControlUpper", lines[0]);
        Assert.Equal(series.Count + 1, lines.Length);
        Assert.All(lines.Skip(1), l => Assert.Equal(lines[0].Split(',').Length, l.Split(',').Length));
    }

    // --- detection behaviour (moved from the core detector's tests when detection moved here) ---

    private static DetectCommand.Detection Detect(IReadOnlyList<double> series) =>
        DetectCommand.Detect(series, DetectCommand.DefaultThreshold, DetectCommand.DefaultSensitivity);

    private static List<double> SpikeSeries(int length = 60, int spikeAt = 30, double spikeValue = 100.0)
    {
        var series = Enumerable.Range(0, length).Select(i => 10.0 + (i % 3) * 0.1).ToList();
        series[spikeAt] = spikeValue;
        return series;
    }

    /// <summary>Deterministic pseudo-noise in (-1, 1) — a seeded <c>Random</c> is not guaranteed
    /// stable across runtimes, and these tests assert on dispersion.</summary>
    private static double Jitter(int i) => Math.Sin(i * 12.9898) * 43758.5453 % 1.0;

    /// <summary>
    /// Sine + noise around <paramref name="level"/> with four injected spikes. <paramref name="scale"/>
    /// multiplies every deviation from the level, so <c>level</c> and <c>scale</c> move the series'
    /// magnitude and its dispersion independently — the separation the control-limit contract is about.
    /// </summary>
    private static List<double> SeasonalNoisySeries(double level = 25.0, double scale = 1.0, int length = 300)
    {
        var spikes = new HashSet<int> { 50, 120, 200, 250 };
        return Enumerable.Range(0, length)
            .Select(i =>
            {
                var deviation = 2.2 * Math.Sin(2 * Math.PI * i / 63) + 0.4 * Jitter(i) + (spikes.Contains(i) ? 12.0 : 0.0);
                return level + scale * deviation;
            })
            .ToList();
    }

    private static double MedianControlWidth(DetectCommand.Detection result)
    {
        var widths = result.Report.Points.Select(p => p.Upper - p.Lower).OrderBy(w => w).ToList();
        return widths[widths.Count / 2];
    }

    [Fact]
    public void Detect_ControlLimits_TrackDispersionNotMagnitude()
    {
        var baseWidth = MedianControlWidth(Detect(SeasonalNoisySeries()));
        Assert.True(baseWidth > 0, "baseline control band collapsed to zero width");

        var shiftedRatio = MedianControlWidth(Detect(SeasonalNoisySeries(level: 1025.0))) / baseWidth;
        Assert.InRange(shiftedRatio, 0.8, 1.25); // level +1000, same spread => same band

        var noisierRatio = MedianControlWidth(Detect(SeasonalNoisySeries(scale: 5.0))) / baseWidth;
        Assert.InRange(noisierRatio, 3.5, 6.5); // spread x5 => band ~5x
    }

    [Fact]
    public void Detect_FindsExactlyTheInjectedSpikesOnASeasonalSeries()
    {
        // With the period passed in, the former detector flagged 191 of these 300 points.
        var result = Detect(SeasonalNoisySeries());

        Assert.Equal([50, 120, 200, 250], result.Report.Anomalies);

        // 300 points hold 4.8 cycles of 63 — not a whole number, which once pulled the estimate
        // short (to 60); the former detector found no period at all in this series.
        Assert.Equal(63, result.Period.Period);
    }

    [Fact]
    public void Detect_NormalPoints_LieWithinControlLimits()
    {
        // The SPC semantic: a control chart is unusable if normal operation plots as violations.
        var normal = Detect(SeasonalNoisySeries()).Report.Points.Where(p => !p.IsAnomaly).ToList();
        var inside = normal.Count(p => p.Value >= p.Lower && p.Value <= p.Upper);

        Assert.True(inside >= normal.Count * 0.95, $"only {inside}/{normal.Count} normal points inside the control band");
    }

    [Fact]
    public void Detect_ConstantSeries_HasNoPeriodAndNoAnomalies()
    {
        var result = Detect(Enumerable.Repeat(7.0, 48).ToList());

        Assert.Null(result.Period.Period);
        Assert.Empty(result.Report.Anomalies);
        Assert.All(result.Report.Points, p => Assert.False(double.IsNaN(p.Lower) || double.IsNaN(p.Upper)));
    }

    [Fact]
    public void Detect_IndexAndValueRoundTrip()
    {
        // Output order must match input order — the Index/Value pair is the caller's join key.
        var series = SpikeSeries(length: 24, spikeAt: 12);
        var result = Detect(series);

        Assert.All(result.Report.Points, p => Assert.Equal(series[p.Index], p.Value));
        Assert.Equal(Enumerable.Range(0, series.Count), result.Report.Points.Select(p => p.Index));
        Assert.True(result.Report.Points[12].IsAnomaly);
    }

    [Theory]
    [InlineData(11, 3.0, 99.73, "12")]              // too short
    [InlineData(48, 0.0, 99.73, "threshold")]       // threshold must be > 0
    [InlineData(48, 3.0, 100.0, "sensitivity")]     // coverage must be < 100
    public void Detect_RefusesInputItCannotScore_AndSaysWhy(int length, double threshold, double sensitivity, string named)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            DetectCommand.Detect(Enumerable.Repeat(1.0, length).ToList(), threshold, sensitivity));

        Assert.Contains(named, ex.Message);
    }

    [Fact]
    public void Create_RequiresDataFileArgument()
    {
        var command = DetectCommand.Create();

        var parse = command.Parse(Array.Empty<string>());

        Assert.NotEmpty(parse.Errors);
    }
}
