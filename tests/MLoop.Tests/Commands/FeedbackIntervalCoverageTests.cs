using System.Globalization;
using System.Text;
using System.Text.Json;
using MLoop.Tests.Documentation;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop feedback metrics</c> reads a regression's stated intervals against what happened: the share of actual values inside
/// them next to the level they stated, and no value-match accuracy for an answer that is a number. The predictions are seeded into the
/// project's logs the way a served regression writes them (an <c>interval</c> on the line), so the whole path is the real one —
/// feedback from a file, the log join, the metrics — only the model that answered is not needed.
/// </summary>
[Collection("FileSystem")]
public sealed class FeedbackIntervalCoverageTests : IClassFixture<EveryCommandRunsContractTests.TrainedProject>
{
    private readonly EveryCommandRunsContractTests.TrainedProject _project;

    public FeedbackIntervalCoverageTests(EveryCommandRunsContractTests.TrainedProject project) => _project = project;

    private string SeedRegressionLog(string model, params double[] actuals)
    {
        var dir = Path.Combine(_project.Root, ".mloop", "logs", model);
        Directory.CreateDirectory(dir);
        var lines = new List<string>();
        var rows = new List<string>();
        for (var i = 0; i < actuals.Length; i++)
        {
            var id = $"ivl{i:D4}x{Guid.NewGuid():N}"[..16];
            lines.Add(JsonSerializer.Serialize(new
            {
                id,
                modelName = model,
                experimentId = "exp-001",
                input = new { x = i },
                output = 15,
                timestamp = "2026-11-01T00:00:00+00:00",
                interval = new { lower = 10, upper = 20, confidence = 0.9 },
            }));
            rows.Add($"{id},{actuals[i].ToString(CultureInfo.InvariantCulture)}");
        }
        File.WriteAllLines(Path.Combine(dir, "2026-11-01.jsonl"), lines);
        var file = $"ivl-{model}.csv";
        File.WriteAllText(Path.Combine(_project.Root, file), "prediction_id,actual\n" + string.Join('\n', rows) + "\n", new UTF8Encoding(false));
        return file;
    }

    [Fact]
    public async Task ReportsTheShareOfActualValuesInsideTheStatedIntervalAndNoAccuracy()
    {
        // 36 inside [10, 20], 14 outside: 72 % of a stated 90 % — far enough below it that the best case of the share (about 83 %) is too.
        var actuals = Enumerable.Range(0, 36).Select(_ => 15.0).Concat(Enumerable.Range(0, 14).Select(_ => 40.0)).ToArray();
        var file = SeedRegressionLog("ivlmodel", actuals);
        var added = await _project.RunAsync("feedback", "add", "--from-file", file, "--json");
        Assert.True(added.Exit == 0, added.Out + added.Err);

        var seen = await _project.RunAsync("feedback", "metrics", "--name", "ivlmodel", "--json");

        Assert.True(seen.Exit == 0, seen.Out + seen.Err);
        var json = seen.Json();
        Assert.Equal(50, json.GetProperty("totalFeedback").GetInt32());
        Assert.Equal(JsonValueKind.Null, json.GetProperty("accuracy").ValueKind);
        var coverage = json.GetProperty("intervalCoverage");
        Assert.Equal(0.72, coverage.GetProperty("observed").GetDouble(), 9);
        Assert.Equal(0.9, coverage.GetProperty("nominal").GetDouble(), 9);
        Assert.Equal(50, coverage.GetProperty("n").GetInt32());
        Assert.Equal(0, coverage.GetProperty("unjudged").GetInt32());

        // The table says it, and says it is below the level — the reason the line exists.
        var table = await _project.RunAsync("feedback", "metrics", "--name", "ivlmodel");
        Assert.Contains("Interval coverage", table.Out, StringComparison.Ordinal);
        Assert.Contains("72.0%", table.Out, StringComparison.Ordinal);
        Assert.Contains("Below the stated level", table.Out, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DoesNotSayBelowTheStatedLevelForAShareThatCouldBeTheLevel()
    {
        // 9 inside of 10 on a stated 90 %: exactly the level, and ten rows cannot say otherwise.
        var file = SeedRegressionLog("ivlokay", [.. Enumerable.Repeat(15.0, 9), 40.0]);
        Assert.Equal(0, (await _project.RunAsync("feedback", "add", "--from-file", file, "--json")).Exit);

        var table = await _project.RunAsync("feedback", "metrics", "--name", "ivlokay");

        Assert.Contains("Interval coverage", table.Out, StringComparison.Ordinal);
        Assert.DoesNotContain("Below the stated level", table.Out, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheListSaysInsideOrOutsideForAnAnswerThatStatedAnIntervalNotMatchOrNot()
    {
        var file = SeedRegressionLog("ivllist", 15.0, 40.0);
        Assert.Equal(0, (await _project.RunAsync("feedback", "add", "--from-file", file, "--json")).Exit);

        var table = await _project.RunAsync("feedback", "list", "--name", "ivllist", "--limit", "1000");

        Assert.Contains("inside", table.Out, StringComparison.Ordinal);
        Assert.Contains("outside", table.Out, StringComparison.Ordinal);
        var json = (await _project.RunAsync("feedback", "list", "--name", "ivllist", "--limit", "1000", "--json")).Json();
        Assert.Contains(json.EnumerateArray(), e => e.GetProperty("interval").ValueKind == JsonValueKind.Object
            && e.GetProperty("interval").GetProperty("confidence").GetDouble() == 0.9);
    }
}
