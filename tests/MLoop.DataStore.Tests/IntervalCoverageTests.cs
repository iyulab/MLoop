using FluentAssertions;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;

namespace MLoop.DataStore.Tests;

/// <summary>
/// The interval a regression answer states is kept with the prediction, travels with the feedback that names it, and
/// <see cref="FeedbackMetrics.IntervalCoverage"/> reads how often the actual value fell inside it — the check that a
/// stated "90 % interval" still holds in operation. A value-equality accuracy means nothing for such an answer.
/// </summary>
public class IntervalCoverageTests : IDisposable
{
    private readonly string _testDir = Path.Combine(Path.GetTempPath(), $"mloop-interval-{Guid.NewGuid():N}");
    private readonly FileFeedbackCollector _collector;
    private readonly FilePredictionLogger _logger;

    public IntervalCoverageTests()
    {
        Directory.CreateDirectory(_testDir);
        _collector = new FileFeedbackCollector(_testDir);
        _logger = new FilePredictionLogger(_testDir);
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(_testDir, recursive: true);
        }
        catch (IOException)
        {
            // A temp directory a test leaves behind is not the test's result.
        }
    }

    private async Task<string> Log(object output, PredictionInterval? interval, string model = "m", string experiment = "exp-001")
    {
        var id = PredictionIds.New();
        await _logger.LogBatchAsync(model, experiment,
        [
            new PredictionLogEntry(model, experiment, new Dictionary<string, object> { ["x"] = 1 }, output, null,
                DateTimeOffset.UtcNow, id, Interval: interval),
        ]);
        return id;
    }

    private static readonly PredictionInterval Ten20 = new(10, 20, 0.9);

    [Fact]
    public async Task The_interval_of_a_logged_prediction_reads_back_and_is_written_only_when_there_is_one()
    {
        var withInterval = await Log(15.0, Ten20);
        var without = await Log("yes", null);

        var logs = await _logger.GetLogsAsync("m");

        logs.Single(l => l.Id == withInterval).Interval.Should().Be(Ten20);
        logs.Single(l => l.Id == without).Interval.Should().BeNull();
        var raw = File.ReadAllLines(Directory.GetFiles(Path.Combine(_testDir, ".mloop", "logs", "m"), "*.jsonl").Single());
        raw.Single(l => l.Contains(withInterval, StringComparison.Ordinal)).Should().Contain("\"interval\":{\"lower\":10,\"upper\":20,\"confidence\":0.9}");
        raw.Single(l => l.Contains(without, StringComparison.Ordinal)).Should().NotContain("interval");
    }

    [Fact]
    public async Task A_log_line_written_before_intervals_were_kept_reads_without_one()
    {
        var path = Path.Combine(_testDir, ".mloop", "logs", "m", "2026-01-02.jsonl");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllTextAsync(path,
            """{"id":"abc","modelName":"m","experimentId":"exp-001","input":{"x":1},"output":15.5,"timestamp":"2026-01-02T00:00:00+00:00"}""" + "\n");

        var logs = await _logger.GetLogsAsync("m");

        logs.Should().ContainSingle().Which.Interval.Should().BeNull();
    }

    [Fact]
    public async Task Feedback_carries_the_interval_of_the_prediction_it_names_one_at_a_time_and_in_a_batch()
    {
        var one = await Log(15.0, Ten20);
        var two = await Log(16.0, new PredictionInterval(12, 19, 0.8));
        var none = await Log(17.0, null);

        await _collector.RecordFeedbackAsync(one, 14.0);
        var batch = await _collector.RecordFeedbackBatchAsync(
        [
            new FeedbackRow(1, two, 18.0),
            new FeedbackRow(2, none, 17.5),
        ]);

        batch.Recorded.Should().Be(2);
        var feedback = (await _collector.GetFeedbackAsync("m")).ToDictionary(f => f.PredictionId);
        feedback[one].Interval.Should().Be(Ten20);
        feedback[two].Interval.Should().Be(new PredictionInterval(12, 19, 0.8));
        feedback[none].Interval.Should().BeNull();
    }

    [Fact]
    public async Task Coverage_is_the_share_of_actual_values_inside_the_stated_interval_and_accuracy_is_left_out()
    {
        // Eight of ten fall inside [10, 20] (the bounds belong to it); two miss.
        double[] actuals = [10, 12, 15, 15, 18, 20, 11, 19, 9.99, 25];
        foreach (var actual in actuals)
            await _collector.RecordFeedbackAsync(await Log(15.0, Ten20), actual);

        var metrics = await _collector.CalculateMetricsAsync("m");

        metrics.TotalFeedback.Should().Be(10);
        metrics.IntervalCoverage.Should().NotBeNull();
        metrics.IntervalCoverage!.Nominal.Should().BeApproximately(0.9, 1e-12);
        metrics.IntervalCoverage.Observed.Should().BeApproximately(0.8, 1e-12);
        metrics.IntervalCoverage.N.Should().Be(10);
        metrics.IntervalCoverage.Unjudged.Should().Be(0);
        // The value-equality accuracy of a regression answer is not a measure: it is not reported.
        metrics.Accuracy.Should().BeNull();
        metrics.EffectiveSampleSize.Should().BeNull();
    }

    [Fact]
    public async Task Coverage_counts_each_prediction_by_its_weight()
    {
        // The row inside carries weight 1, the row outside weight 3 — the share a sample drawn with unequal
        // probability stands for is 1/4, not 1/2.
        await _collector.RecordFeedbackAsync(await Log(15.0, Ten20), 15.0, weight: 1);
        await _collector.RecordFeedbackAsync(await Log(15.0, Ten20), 30.0, weight: 3);

        var metrics = await _collector.CalculateMetricsAsync("m");

        metrics.Weighted.Should().BeTrue();
        metrics.IntervalCoverage!.Observed.Should().BeApproximately(0.25, 1e-12);
        metrics.IntervalCoverage.N.Should().Be(2);
        // Kish: (1 + 3)² / (1² + 3²) = 1.6 rows' worth of evidence.
        metrics.IntervalCoverage.EffectiveN.Should().BeApproximately(1.6, 1e-12);
    }

    [Fact]
    public async Task The_nominal_level_is_the_weighted_mean_when_the_stated_levels_differ()
    {
        // A retrained model can state another level; the stated level the rows stand for is their mean.
        await _collector.RecordFeedbackAsync(await Log(15.0, new PredictionInterval(10, 20, 0.9)), 15.0);
        await _collector.RecordFeedbackAsync(await Log(15.0, new PredictionInterval(10, 20, 0.8)), 15.0);

        var metrics = await _collector.CalculateMetricsAsync("m");

        metrics.IntervalCoverage!.Nominal.Should().BeApproximately(0.85, 1e-12);
    }

    [Fact]
    public async Task A_prediction_without_an_interval_is_counted_apart_and_keeps_its_own_accuracy()
    {
        // A project that predicted a class before it predicted a value, or a regression logged before intervals
        // were kept: the rows that state no interval are not judged by one, and are counted so the share is read
        // against how much of the feedback it covers.
        await _collector.RecordFeedbackAsync(await Log(15.0, Ten20), 15.0);
        await _collector.RecordFeedbackAsync(await Log("yes", null), "yes");
        await _collector.RecordFeedbackAsync(await Log("yes", null), "no");

        var metrics = await _collector.CalculateMetricsAsync("m");

        metrics.TotalFeedback.Should().Be(3);
        metrics.IntervalCoverage!.N.Should().Be(1);
        metrics.IntervalCoverage.Unjudged.Should().Be(2);
        metrics.Accuracy.Should().BeApproximately(0.5, 1e-12);
        metrics.EffectiveSampleSize.Should().BeApproximately(2, 1e-12);
    }

    [Fact]
    public async Task Without_an_interval_in_the_feedback_there_is_no_coverage_and_accuracy_is_as_before()
    {
        await _collector.RecordFeedbackAsync(await Log("yes", null), "yes");
        await _collector.RecordFeedbackAsync(await Log("yes", null), "no");

        var metrics = await _collector.CalculateMetricsAsync("m");

        metrics.IntervalCoverage.Should().BeNull();
        metrics.Accuracy.Should().BeApproximately(0.5, 1e-12);
    }

    [Fact]
    public async Task An_actual_value_that_is_not_a_number_cannot_be_inside_an_interval()
    {
        await _collector.RecordFeedbackAsync(await Log(15.0, Ten20), 15.0);
        await _collector.RecordFeedbackAsync(await Log(15.0, Ten20), "not a number");

        var metrics = await _collector.CalculateMetricsAsync("m");

        metrics.IntervalCoverage!.N.Should().Be(1);
        metrics.IntervalCoverage.Unjudged.Should().Be(1);
    }

    [Fact]
    public async Task An_interval_that_cannot_hold_a_value_judges_nothing()
    {
        // Bounds the wrong way round: written by hand or by an older producer, kept as stated, never judged.
        var id = await Log(15.0, new PredictionInterval(20, 10, 0.9));
        await _collector.RecordFeedbackAsync(id, 15.0);

        var metrics = await _collector.CalculateMetricsAsync("m");

        metrics.IntervalCoverage.Should().BeNull();
    }

    [Fact]
    public async Task Recording_feedback_again_corrects_the_prediction_and_it_is_still_counted_once()
    {
        var id = await Log(15.0, Ten20);
        await _collector.RecordFeedbackAsync(id, 30.0);
        await Task.Delay(5);
        await _collector.RecordFeedbackAsync(id, 15.0);

        var metrics = await _collector.CalculateMetricsAsync("m");

        metrics.IntervalCoverage!.N.Should().Be(1);
        metrics.IntervalCoverage.Observed.Should().Be(1.0);
    }

    [Fact]
    public async Task Coverage_of_one_experiment_reads_only_its_answers()
    {
        await _collector.RecordFeedbackAsync(await Log(15.0, Ten20, experiment: "exp-001"), 15.0);
        await _collector.RecordFeedbackAsync(await Log(15.0, Ten20, experiment: "exp-002"), 99.0);

        var first = await _collector.CalculateMetricsAsync("m", experimentId: "exp-001");
        var second = await _collector.CalculateMetricsAsync("m", experimentId: "exp-002");

        first.IntervalCoverage!.Observed.Should().Be(1.0);
        second.IntervalCoverage!.Observed.Should().Be(0.0);
    }

    [Fact]
    public async Task A_feedback_line_written_before_intervals_were_kept_reads_without_one()
    {
        var path = Path.Combine(_testDir, ".mloop", "feedback", "m", "2026-01-02.jsonl");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllTextAsync(path,
            """{"predictionId":"p1","modelName":"m","predictedValue":15.5,"actualValue":15.5,"timestamp":"2026-01-02T00:00:00+00:00"}""" + "\n");

        var metrics = await _collector.CalculateMetricsAsync("m");

        metrics.IntervalCoverage.Should().BeNull();
        metrics.Accuracy.Should().Be(1.0);
    }

    [Fact]
    public async Task AValueAnswerThatStatedNoIntervalHasNoAccuracyWhenTheModelAnswersValues()
    {
        // 40 answers each 0.1 from the truth — a good model; "the same value" reads every one wrong.
        for (var i = 0; i < 40; i++)
            await _collector.RecordFeedbackAsync(await Log(4.5, interval: null), 4.6);

        var asValues = await _collector.CalculateMetricsAsync("m", valueAnswers: true);
        var asClasses = await _collector.CalculateMetricsAsync("m");

        Assert.Equal(40, asValues.TotalFeedback);
        Assert.Null(asValues.Accuracy);
        Assert.Null(asValues.EffectiveSampleSize);
        Assert.Null(asValues.IntervalCoverage);
        // The reading every other model keeps: a numeric class label is matched, not read near.
        Assert.Equal(0.0, asClasses.Accuracy);
    }

    [Fact]
    public async Task ValueAnswersThatStatedAnIntervalStillReadAsCoverageAndTheOnesThatDidNotAreCountedAsUnjudged()
    {
        for (var i = 0; i < 10; i++)
            await _collector.RecordFeedbackAsync(await Log(15.0, Ten20), 15.0);   // inside [10, 20]
        for (var i = 0; i < 5; i++)
            await _collector.RecordFeedbackAsync(await Log(15.0, interval: null), 15.2);

        var metrics = await _collector.CalculateMetricsAsync("m", valueAnswers: true);

        Assert.Null(metrics.Accuracy);
        Assert.NotNull(metrics.IntervalCoverage);
        Assert.Equal(10, metrics.IntervalCoverage!.N);
        Assert.Equal(5, metrics.IntervalCoverage.Unjudged);
        Assert.Equal(15, metrics.TotalFeedback);
    }
}
