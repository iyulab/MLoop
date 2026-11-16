using System.Globalization;
using System.Text.Json;
using MLoop.Core.Monitoring;

namespace MLoop.Core.Tests.Monitoring;

/// <summary>
/// A column that rises through the whole training file (a serial number, a counter, a time) is outside its training range in every later window,
/// so its PSI says time has passed, not that the world moved. It is measured and shown, but it is not what the level, the range signal
/// or the retraining trigger read.
/// </summary>
public class DriftTrendTests
{
    private static readonly (string, bool)[] Columns = [("seq", true), ("x", true)];

    private static async Task<CsvSample> SampleAsync(IEnumerable<string> lines)
    {
        var path = Path.Combine(Path.GetTempPath(), $"trend-{Guid.NewGuid():N}.csv");
        try
        {
            await File.WriteAllLinesAsync(path, lines);
            return await DriftProfile.SampleCsvAsync(path, sampleSize: 200);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string Num(double v) => v.ToString("R", CultureInfo.InvariantCulture);

    [Fact]
    public async Task AColumnThatRisesThroughTheWholeFile_IsATrend()
    {
        var rng = new Random(1);
        var sample = await SampleAsync(new[] { "seq,x,flat" }.Concat(
            Enumerable.Range(1, 600).Select(i => $"{i},{Num(rng.NextDouble())},7")));

        Assert.Contains("seq", sample.TrendColumns);
        Assert.DoesNotContain("x", sample.TrendColumns);        // noise
        Assert.DoesNotContain("flat", sample.TrendColumns);     // never steps
    }

    [Fact]
    public async Task ATimeThatRepeatsWithinADay_IsStillATrend()
    {
        // 48 half-hours a day: the value ties 47 times in 48, and never falls.
        var sample = await SampleAsync(new[] { "date,y" }.Concat(
            Enumerable.Range(0, 2400).Select(i => $"{Num(0.01 * (i / 48))},{i % 3}")));

        Assert.Contains("date", sample.TrendColumns);
    }

    [Fact]
    public async Task AColumnThatEverFalls_IsNotATrend()
    {
        var lines = new[] { "seq,y" }.Concat(Enumerable.Range(1, 500).Select(i => $"{(i == 250 ? 3 : i)},1"));

        Assert.DoesNotContain("seq", (await SampleAsync(lines)).TrendColumns);
    }

    [Fact]
    public async Task AFileSortedByAGroupWithAHandfulOfSteps_IsNotATrend()
    {
        // Rows sorted by a numeric site id with five sites: it never falls, but five values are not time passing.
        var lines = new[] { "site,y" }.Concat(Enumerable.Range(0, 500).Select(i => $"{i / 100},1"));

        Assert.DoesNotContain("site", (await SampleAsync(lines)).TrendColumns);
    }

    [Fact]
    public async Task AFileTooShortToTell_HasNoTrend()
    {
        var lines = new[] { "seq,y" }.Concat(Enumerable.Range(1, 60).Select(i => $"{i},1"));

        Assert.Empty((await SampleAsync(lines)).TrendColumns);
    }

    [Fact]
    public async Task ABlankCell_DoesNotBreakATrend_ButATextCellDoes()
    {
        var blank = new[] { "seq,y" }.Concat(Enumerable.Range(1, 400).Select(i => $"{(i % 50 == 0 ? "" : i.ToString(CultureInfo.InvariantCulture))},1"));
        var text = new[] { "seq,y" }.Concat(Enumerable.Range(1, 400).Select(i => $"{(i == 200 ? "n/a" : i.ToString(CultureInfo.InvariantCulture))},1"));

        Assert.Contains("seq", (await SampleAsync(blank)).TrendColumns);
        Assert.DoesNotContain("seq", (await SampleAsync(text)).TrendColumns);
    }

    [Fact]
    public async Task TheProfile_MarksOnlyTheNumericTrendColumn()
    {
        var rng = new Random(2);
        var sample = await SampleAsync(new[] { "seq,x" }.Concat(Enumerable.Range(1, 600).Select(i => $"{i},{Num(rng.NextDouble())}")));

        var profile = DriftProfile.Build(sample.Project(["seq", "x"]), Columns, sample.TrendColumns);

        Assert.True(profile.Features[0].Trend);
        Assert.False(profile.Features[1].Trend);
    }

    private static DriftProfile TrendProfile()
    {
        var rng = new Random(3);
        var rows = Enumerable.Range(1, 1000)
            .Select(i => new[] { i.ToString(CultureInfo.InvariantCulture), Num(rng.NextDouble()) }).ToList();
        return DriftProfile.Build(rows, Columns, new HashSet<string> { "seq" });
    }

    private static IReadOnlyDictionary<string, object?> Row(double seq, double x)
        => new Dictionary<string, object?> { ["seq"] = seq, ["x"] = x };

    [Fact]
    public void ANormalWindow_IsNotSignificantBecauseTheSerialNumberKeepsCounting()
    {
        var profile = TrendProfile();
        var rng = new Random(4);
        var window = Enumerable.Range(0, 400).Select(i => Row(1001 + i, rng.NextDouble())).ToList();

        var report = DriftMeasure.Compare(profile, window);

        var seq = Assert.Single(report.Features, f => f.Column == "seq");
        Assert.True(seq.Trend);
        Assert.True(seq.Psi > 1, "the column itself is still measured");
        Assert.Equal(DriftLevel.None, report.Level);
        Assert.True(report.MaxPsi < 0.25, $"max PSI {report.MaxPsi}");
        Assert.DoesNotContain(report.RangeNotable, f => f.Column == "seq");
    }

    [Fact]
    public void AWindowThatMovedInAnotherColumn_IsStillFlagged()
    {
        var profile = TrendProfile();
        var rng = new Random(5);
        var window = Enumerable.Range(0, 400).Select(i => Row(1001 + i, 0.8 + 0.2 * rng.NextDouble())).ToList();

        var report = DriftMeasure.Compare(profile, window);

        Assert.Equal(DriftLevel.Significant, report.Level);
        Assert.Equal("x", report.Judged.First().Column);
    }

    [Fact]
    public void OnlyATrendColumn_LeavesNothingToJudge()
    {
        var profile = DriftProfile.Build(
            Enumerable.Range(1, 300).Select(i => new[] { i.ToString(CultureInfo.InvariantCulture) }).ToList(),
            [("seq", true)], new HashSet<string> { "seq" });

        var report = DriftMeasure.Compare(profile, Enumerable.Range(0, 300).Select(i => (IReadOnlyDictionary<string, object?>)new Dictionary<string, object?> { ["seq"] = 400.0 + i }).ToList());

        Assert.Equal(DriftLevel.None, report.Level);
        Assert.Equal(0, report.MaxPsi);
    }

    [Fact]
    public void AProfileWrittenBeforeTrendsWereKept_ReadsWithNone_AndOneWithoutTrendsWritesNoField()
    {
        const string Old = """{"version":1,"rows":3,"features":[{"column":"seq","isNumeric":true,"shares":[0.5,0.5]}]}""";
        var options = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, PropertyNameCaseInsensitive = true };

        var old = JsonSerializer.Deserialize<DriftProfile>(Old, options)!;
        Assert.False(old.Features[0].Trend);

        var written = JsonSerializer.Serialize(DriftProfile.Build([["1", "a"], ["2", "b"]], [("n", true), ("c", false)]), options);
        Assert.DoesNotContain("trend", written, StringComparison.Ordinal);

        var withTrend = JsonSerializer.Serialize(TrendProfile(), options);
        Assert.Contains("\"trend\":true", withTrend, StringComparison.Ordinal);
        Assert.True(JsonSerializer.Deserialize<DriftProfile>(withTrend, options)!.Features[0].Trend);
    }
}
