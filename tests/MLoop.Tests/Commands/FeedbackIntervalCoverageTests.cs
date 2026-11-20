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

    /// <summary>A promoted value model with the RMSE it recorded on its held-out rows (or none), answering 4.5 each time, and feedback of <paramref name="actual"/> for every answer.</summary>
    private string SeedValueModel(string model, string task, double? heldOutRmse, double actual, int count)
    {
        var file = SeedValueLogWithoutInterval(model, task, count);
        var production = Path.Combine(_project.Root, "models", model, "production");
        File.WriteAllText(Path.Combine(production, "metadata.json"),
            JsonSerializer.Serialize(new { modelName = model, experimentId = "exp-001", task, metrics = heldOutRmse is { } rmse ? new Dictionary<string, double> { ["rmse"] = rmse } : null }),
            new UTF8Encoding(false));
        var rows = File.ReadAllLines(Path.Combine(_project.Root, file)).Skip(1)
            .Select(l => l.Split(',')[0] + "," + actual.ToString(CultureInfo.InvariantCulture));
        File.WriteAllText(Path.Combine(_project.Root, file), "prediction_id,actual\n" + string.Join('\n', rows) + "\n", new UTF8Encoding(false));
        return file;
    }

    [Fact]
    public async Task AValueModelsFeedbackErrorIsShownNextToTheErrorItMadeOnItsHeldOutRows()
    {
        // Answers of 4.5 against actual 4.6: an error of 0.1 on every row, as large as the 0.1 recorded at training — not a rise.
        var file = SeedValueModel("valerr", "regression", heldOutRmse: 0.1, actual: 4.6, count: 40);
        Assert.Equal(0, (await _project.RunAsync("feedback", "add", "--from-file", file, "--json")).Exit);

        var json = (await _project.RunAsync("feedback", "metrics", "--name", "valerr", "--json")).Json();
        var error = json.GetProperty("valueError");
        Assert.Equal(0.1, error.GetProperty("rmse").GetDouble(), 9);
        Assert.Equal(0.1, error.GetProperty("heldOutRmse").GetDouble(), 9);
        Assert.Equal(1.0, error.GetProperty("ratio").GetDouble(), 9);
        Assert.Equal(40, error.GetProperty("n").GetInt32());
        Assert.Equal("exp-001", error.GetProperty("experimentId").GetString());

        var table = await _project.RunAsync("feedback", "metrics", "--name", "valerr");
        Assert.Contains("Error, RMSE", table.Out, StringComparison.Ordinal);
        Assert.Contains("1.00×", table.Out, StringComparison.Ordinal);
        Assert.DoesNotContain("Established above", table.Out, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AnErrorEstablishedAboveTheRetrainingRatioIsSaidSoAndNamesTheCondition()
    {
        // 4.5 against 6.5: an error of 2 on every row against a held-out 0.1 — twenty times, on forty rows.
        var file = SeedValueModel("valrise", "regression", heldOutRmse: 0.1, actual: 6.5, count: 40);
        Assert.Equal(0, (await _project.RunAsync("feedback", "add", "--from-file", file, "--json")).Exit);

        var table = await _project.RunAsync("feedback", "metrics", "--name", "valrise");

        Assert.Contains("20.00×", table.Out, StringComparison.Ordinal);
        Assert.Contains("Established above", table.Out, StringComparison.Ordinal);
        Assert.Contains("error_rise", table.Out, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AValueModelThatRecordedNoHeldOutRmseShowsItsErrorAndSaysThereIsNothingToReadItAgainst()
    {
        var file = SeedValueModel("valnobase", "ranking", heldOutRmse: null, actual: 6.5, count: 40);
        Assert.Equal(0, (await _project.RunAsync("feedback", "add", "--from-file", file, "--json")).Exit);

        var json = (await _project.RunAsync("feedback", "metrics", "--name", "valnobase", "--json")).Json();
        var error = json.GetProperty("valueError");
        Assert.Equal(2.0, error.GetProperty("rmse").GetDouble(), 9);
        Assert.Equal(JsonValueKind.Null, error.GetProperty("heldOutRmse").ValueKind);
        Assert.Equal(JsonValueKind.Null, error.GetProperty("ratio").ValueKind);

        var table = await _project.RunAsync("feedback", "metrics", "--name", "valnobase");
        // The grid wraps a long line and colours each piece: read the words, without the colour codes and where the lines broke.
        var words = System.Text.RegularExpressions.Regex.Replace(System.Text.RegularExpressions.Regex.Replace(table.Out, "\u001b\\[[0-9;]*m", ""), @"\s+", " ");
        Assert.True(words.Contains("recorded no RMSE", StringComparison.Ordinal), words);
    }

    [Fact]
    public async Task AModelThatAnswersAClassHasNoValueError()
    {
        var file = SeedValueLogWithoutInterval("valclass", "binary-classification", 40);
        Assert.Equal(0, (await _project.RunAsync("feedback", "add", "--from-file", file, "--json")).Exit);

        var json = (await _project.RunAsync("feedback", "metrics", "--name", "valclass", "--json")).Json();

        Assert.Equal(JsonValueKind.Null, json.GetProperty("valueError").ValueKind);
        Assert.DoesNotContain("Error, RMSE", (await _project.RunAsync("feedback", "metrics", "--name", "valclass")).Out, StringComparison.Ordinal);
    }
}
