using System.Text.Json;
using FluentAssertions;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;

namespace MLoop.DataStore.Tests;

/// <summary>
/// Feedback for many predictions in one call: <see cref="IFeedbackCollector.RecordFeedbackBatchAsync"/> reads the prediction logs once for
/// every id it is given, records the rows it can and reports the rest by row number.
/// </summary>
public class FeedbackBatchTests : IDisposable
{
    private readonly string _testDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");
    private readonly FileFeedbackCollector _collector;
    private readonly FilePredictionLogger _logger;

    public FeedbackBatchTests()
    {
        Directory.CreateDirectory(_testDir);
        _collector = new FileFeedbackCollector(_testDir);
        _logger = new FilePredictionLogger(_testDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_testDir))
                Directory.Delete(_testDir, recursive: true);
        }
        catch
        {
            // Ignore cleanup errors
        }
    }

    [Fact]
    public async Task RecordsTheRowsItCanAndNamesTheRowsItCannot()
    {
        var known = await Log("model", "yes");

        var result = await _collector.RecordFeedbackBatchAsync(
        [
            new FeedbackRow(1, known, "yes"),
            new FeedbackRow(2, "0123456789abcdef", "no"),
            new FeedbackRow(3, "", "no"),
            new FeedbackRow(4, known, "  "),
            new FeedbackRow(5, known, "no", Weight: -1),
        ]);

        result.Recorded.Should().Be(1);
        result.Rejected.Select(r => r.Row).Should().Equal(2, 3, 4, 5);
        result.Rejected[0].Reason.Should().Contain("no logged prediction has this id");
        result.Rejected[0].PredictionId.Should().Be("0123456789abcdef");
        result.Rejected[1].Reason.Should().Contain("id is empty");
        result.Rejected[2].Reason.Should().Contain("actual value is empty");
        result.Rejected[3].Reason.Should().Contain("weight").And.Contain("-1");
        (await _collector.GetFeedbackAsync("model")).Should().ContainSingle(f => f.PredictionId == known);
    }

    [Fact]
    public async Task AnEntryIsTheOneTheSingleRecordWouldHaveWritten()
    {
        // The batch is a faster way to the same entries: model, experiment and the answer it is judged against come from the log line.
        var viaBatch = await Log("model-a", "yes", experimentId: "exp-007");
        var viaSingle = await Log("model-a", "yes", experimentId: "exp-007");

        await _collector.RecordFeedbackBatchAsync([new FeedbackRow(1, viaBatch, "no", "reviewer", 2.5)]);
        await _collector.RecordFeedbackAsync(viaSingle, "no", "reviewer", 2.5);

        var entries = await _collector.GetFeedbackAsync("model-a");
        var batch = entries.Single(e => e.PredictionId == viaBatch);
        var single = entries.Single(e => e.PredictionId == viaSingle);
        batch.ModelName.Should().Be(single.ModelName);
        batch.ExperimentId.Should().Be("exp-007").And.Be(single.ExperimentId);
        batch.PredictedValue.ToString().Should().Be(single.PredictedValue.ToString());
        batch.ActualValue.ToString().Should().Be("no").And.Be(single.ActualValue.ToString());
        batch.Source.Should().Be("reviewer");
        batch.Weight.Should().Be(2.5);
    }

    [Fact]
    public async Task ALabelCarriesNoLineEnd()
    {
        var id = await Log("model", "yes");

        await _collector.RecordFeedbackBatchAsync([new FeedbackRow(1, "  " + id + "\r", "yes\r")]);

        var entry = (await _collector.GetFeedbackAsync("model")).Single();
        entry.PredictionId.Should().Be(id);
        entry.ActualValue.ToString().Should().Be("yes");
    }

    [Fact]
    public async Task AnIdThatSeveralRowsNameStandsAsItsLastRow()
    {
        // Feedback is append-only and the latest entry for a prediction stands. A batch stamps its rows at one instant, and the order
        // of entries with the same instant is the sort's to choose - it picked the first row. Fifty predictions, each named twice,
        // the wrong answer first: they are all right if the later row stands.
        var rows = new List<FeedbackRow>();
        var ids = new List<string>();
        for (var i = 0; i < 50; i++)
            ids.Add(await Log("model", "yes"));
        foreach (var id in ids)
            rows.Add(new FeedbackRow(rows.Count + 1, id, "no"));
        foreach (var id in ids)
            rows.Add(new FeedbackRow(rows.Count + 1, id, "yes"));

        var result = await _collector.RecordFeedbackBatchAsync(rows);

        result.Recorded.Should().Be(100);
        result.Repeats.Should().HaveCount(50);
        result.Repeats.Should().OnlyContain(r => r.Rows.Count == 2 && r.Rows[0] < r.Rows[1]);
        var metrics = await _collector.CalculateMetricsAsync("model");
        metrics.TotalFeedback.Should().Be(50);
        metrics.Accuracy.Should().Be(1.0);
    }

    [Fact]
    public async Task AndTheOtherWayRound()
    {
        var id = await Log("model", "yes");

        await _collector.RecordFeedbackBatchAsync([new FeedbackRow(1, id, "yes"), new FeedbackRow(2, id, "no")]);

        (await _collector.CalculateMetricsAsync("model")).Accuracy.Should().Be(0.0);
    }

    [Fact]
    public async Task FindsAPredictionInAnOlderDayAndInALineWrittenAnotherWay()
    {
        // A day file the store did not write itself (hand-edited, another tool) need not start with {"id":"...".
        var today = await Log("model", "yes");
        var logDir = Path.Combine(_testDir, ".mloop", "logs", "model");
        await File.WriteAllTextAsync(Path.Combine(logDir, "2026-01-01.jsonl"),
            "{ \"modelName\": \"model\", \"output\": \"maybe\", \"id\": \"abc123\" }\n"
            + "{\"id\":\"esc\\u0041ped\",\"output\":\"escaped\"}\n");

        var result = await _collector.RecordFeedbackBatchAsync(
        [
            new FeedbackRow(1, today, "yes"),
            new FeedbackRow(2, "abc123", "maybe"),
            new FeedbackRow(3, "escAped", "escaped"),
        ]);

        result.Rejected.Should().BeEmpty();
        result.Recorded.Should().Be(3);
        var byId = (await _collector.GetFeedbackAsync("model")).ToDictionary(f => f.PredictionId);
        byId["abc123"].PredictedValue.ToString().Should().Be("maybe");
        byId["escAped"].PredictedValue.ToString().Should().Be("escaped");
    }

    [Fact]
    public async Task EntriesLandUnderTheModelThatAnsweredEach()
    {
        var a = await Log("model-a", "yes");
        var b = await Log("model-b", "no");

        await _collector.RecordFeedbackBatchAsync([new FeedbackRow(1, a, "yes"), new FeedbackRow(2, b, "yes")]);

        (await _collector.GetFeedbackAsync("model-a")).Should().ContainSingle(f => f.PredictionId == a);
        (await _collector.GetFeedbackAsync("model-b")).Should().ContainSingle(f => f.PredictionId == b);
    }

    [Fact]
    public async Task AnEmptyBatchRecordsNothing()
    {
        var result = await _collector.RecordFeedbackBatchAsync([]);

        result.Recorded.Should().Be(0);
        result.Rejected.Should().BeEmpty();
        Directory.Exists(Path.Combine(_testDir, ".mloop", "feedback")).Should().BeFalse();
    }

    private async Task<string> Log(string modelName, object output, string experimentId = "exp-001")
    {
        await _logger.LogPredictionAsync(modelName, experimentId, new Dictionary<string, object> { ["feature"] = "value" }, output, 0.9);

        var logFiles = Directory.GetFiles(Path.Combine(_testDir, ".mloop", "logs", modelName), "*.jsonl");
        var lastLine = (await File.ReadAllLinesAsync(logFiles[0])).Last();
        using var doc = JsonDocument.Parse(lastLine);
        return doc.RootElement.GetProperty("id").GetString()!;
    }
}
