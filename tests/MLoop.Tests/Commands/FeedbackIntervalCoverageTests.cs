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

    /// <summary>The logs a similarity model writes: a number answer and no interval on the line; a promoted model of the given task.</summary>
    private string SeedValueLogWithoutInterval(string model, string? task, int count)
    {
        var dir = Path.Combine(_project.Root, ".mloop", "logs", model);
        Directory.CreateDirectory(dir);
        var lines = new List<string>();
        var rows = new List<string>();
        for (var i = 0; i < count; i++)
        {
            var id = $"val{i:D4}x{Guid.NewGuid():N}"[..16];
            lines.Add(JsonSerializer.Serialize(new { id, modelName = model, experimentId = "exp-001", input = new { x = i }, output = 4.5, timestamp = "2026-11-01T00:00:00+00:00" }));
            rows.Add($"{id},4.6");
        }
        File.WriteAllLines(Path.Combine(dir, "2026-11-01.jsonl"), lines);
        if (task is not null)
        {
            var production = Path.Combine(_project.Root, "models", model, "production");
            Directory.CreateDirectory(production);
            File.WriteAllText(Path.Combine(production, "metadata.json"),
                JsonSerializer.Serialize(new { modelName = model, experimentId = "exp-001", task }), new UTF8Encoding(false));
        }
        var file = $"val-{model}.csv";
        File.WriteAllText(Path.Combine(_project.Root, file), "prediction_id,actual\n" + string.Join('\n', rows) + "\n", new UTF8Encoding(false));
        return file;
    }

    [Fact]
    public async Task AValueModelThatStatedNoIntervalHasNoAccuracyAndTheTableAndTheListSayWhy()
    {
        var file = SeedValueLogWithoutInterval("valmodel", "sentence-similarity", 40);
        Assert.Equal(0, (await _project.RunAsync("feedback", "add", "--from-file", file, "--json")).Exit);

        var json = (await _project.RunAsync("feedback", "metrics", "--name", "valmodel", "--json")).Json();
        Assert.Equal(40, json.GetProperty("totalFeedback").GetInt32());
        Assert.Equal(JsonValueKind.Null, json.GetProperty("accuracy").ValueKind);

        var table = await _project.RunAsync("feedback", "metrics", "--name", "valmodel");
        Assert.Contains("answers a value", table.Out, StringComparison.Ordinal);
        Assert.Contains("mloop evaluate", table.Out, StringComparison.Ordinal);

        var list = await _project.RunAsync("feedback", "list", "--name", "valmodel", "--limit", "5");
        Assert.Contains("no interval", list.Out, StringComparison.Ordinal);
        Assert.DoesNotContain("✗", list.Out, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheSameFeedbackWithoutAValueTaskIsMatchedAsItAlwaysWas()
    {
        // No promoted model, so no task to read: a numeric class label is matched, and 4.5 against 4.6 is wrong.
        var file = SeedValueLogWithoutInterval("valnotask", null, 40);
        Assert.Equal(0, (await _project.RunAsync("feedback", "add", "--from-file", file, "--json")).Exit);

        var json = (await _project.RunAsync("feedback", "metrics", "--name", "valnotask", "--json")).Json();

        Assert.Equal(0.0, json.GetProperty("accuracy").GetDouble());
    }

    /// <summary>Predictions of <paramref name="experiment"/> (class answers, "yes") logged for <paramref name="model"/>; returns the CSV of feedback whose actual is "yes" for the first <paramref name="right"/> and "no" for the rest.</summary>
    private string SeedExperimentLog(string model, string experiment, int count, int right)
    {
        var dir = Path.Combine(_project.Root, ".mloop", "logs", model);
        Directory.CreateDirectory(dir);
        var lines = new List<string>();
        var rows = new List<string>();
        for (var i = 0; i < count; i++)
        {
            var id = $"{experiment[^3..]}{i:D3}x{Guid.NewGuid():N}"[..16];
            lines.Add(JsonSerializer.Serialize(new { id, modelName = model, experimentId = experiment, input = new { x = i }, output = "yes", timestamp = "2026-11-01T00:00:00+00:00" }));
            rows.Add($"{id},{(i < right ? "yes" : "no")}");
        }
        File.AppendAllLines(Path.Combine(dir, "2026-11-01.jsonl"), lines);
        var file = $"scope-{model}-{experiment}.csv";
        File.WriteAllText(Path.Combine(_project.Root, file), "prediction_id,actual\n" + string.Join('\n', rows) + "\n", new UTF8Encoding(false));
        return file;
    }

    private void Promote(string model, string experiment, string task)
    {
        var production = Path.Combine(_project.Root, "models", model, "production");
        Directory.CreateDirectory(production);
        File.WriteAllText(Path.Combine(production, "metadata.json"),
            JsonSerializer.Serialize(new { modelName = model, experimentId = experiment, task }), new UTF8Encoding(false));
    }

    [Fact]
    public async Task MetricsAreOfTheProductionExperimentsAnswersLikeTriggerCheckAndAllOrOneOtherOnRequest()
    {
        // exp-001 answered 40 times, 10 right; exp-002 replaced it and has answered 50 times, 45 right.
        foreach (var (experiment, count, right) in new[] { ("exp-001", 40, 10), ("exp-002", 50, 45) })
            Assert.Equal(0, (await _project.RunAsync("feedback", "add", "--from-file", SeedExperimentLog("scoped", experiment, count, right), "--json")).Exit);
        Promote("scoped", "exp-002", "binary-classification");

        var byDefault = (await _project.RunAsync("feedback", "metrics", "--name", "scoped", "--json")).Json();
        Assert.Equal(50, byDefault.GetProperty("totalFeedback").GetInt32());
        Assert.Equal("exp-002", byDefault.GetProperty("experimentId").GetString());
        Assert.Equal(0.9, byDefault.GetProperty("accuracy").GetDouble(), 9);

        var every = (await _project.RunAsync("feedback", "metrics", "--name", "scoped", "--experiment", "all", "--json")).Json();
        Assert.Equal(90, every.GetProperty("totalFeedback").GetInt32());
        Assert.Equal(JsonValueKind.Null, every.GetProperty("experimentId").ValueKind);
        Assert.Equal(55.0 / 90.0, every.GetProperty("accuracy").GetDouble(), 9);

        var other = (await _project.RunAsync("feedback", "metrics", "--name", "scoped", "--experiment", "exp-001", "--json")).Json();
        Assert.Equal(40, other.GetProperty("totalFeedback").GetInt32());
        Assert.Equal(0.25, other.GetProperty("accuracy").GetDouble(), 9);

        // The retraining condition reads the same answers: the two commands agree on the model in production.
        var trigger = (await _project.RunAsync("trigger", "check", "--name", "scoped", "--json")).Json();
        var accuracy = trigger.GetProperty("conditions").EnumerateArray().First(c => c.GetProperty("name").GetString() == "accuracy_threshold");
        Assert.Equal(0.9, accuracy.GetProperty("currentValue").GetDouble(), 9);
    }

    [Fact]
    public async Task TheTableSaysWhoseAnswersAndHowMuchFeedbackOfOtherExperimentsIsNotCounted()
    {
        foreach (var (experiment, count, right) in new[] { ("exp-001", 40, 10), ("exp-002", 50, 45) })
            Assert.Equal(0, (await _project.RunAsync("feedback", "add", "--from-file", SeedExperimentLog("scopedtable", experiment, count, right), "--json")).Exit);
        Promote("scopedtable", "exp-002", "binary-classification");

        var table = await _project.RunAsync("feedback", "metrics", "--name", "scopedtable");
        var words = System.Text.RegularExpressions.Regex.Replace(System.Text.RegularExpressions.Regex.Replace(table.Out, "\u001b\\[[0-9;]*m", ""), @"\s+", " ");

        Assert.Contains("exp-002's answers", words, StringComparison.Ordinal);
        Assert.Contains("40 feedback on other experiments not counted", words, StringComparison.Ordinal);
        Assert.Contains("--experiment all", words, StringComparison.Ordinal);
    }

    [Fact]
    public async Task WithNothingPromotedTheMetricsAreOfEveryAnswerAsTheyAlwaysWere()
    {
        foreach (var (experiment, count, right) in new[] { ("exp-001", 40, 10), ("exp-002", 50, 45) })
            Assert.Equal(0, (await _project.RunAsync("feedback", "add", "--from-file", SeedExperimentLog("unpromoted", experiment, count, right), "--json")).Exit);

        var json = (await _project.RunAsync("feedback", "metrics", "--name", "unpromoted", "--json")).Json();

        Assert.Equal(90, json.GetProperty("totalFeedback").GetInt32());
        Assert.Equal(JsonValueKind.Null, json.GetProperty("experimentId").ValueKind);
    }
}
