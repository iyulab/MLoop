using FluentAssertions;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;

namespace MLoop.DataStore.Tests;

/// <summary>
/// <see cref="FeedbackMetrics.ValueError"/>: how far a value answer was from the actual value its feedback names — the mean of the squared
/// differences by weight, with the spread of the squares and the rows' worth as evidence. Asked for by the caller that knows the answer is a
/// value (a class has no distance); read whether or not the answer stated an interval.
/// </summary>
public class ValueErrorTests : IDisposable
{
    private readonly string _testDir = Path.Combine(Path.GetTempPath(), $"mloop-value-error-{Guid.NewGuid():N}");
    private readonly FileFeedbackCollector _collector;
    private readonly FilePredictionLogger _logger;

    public ValueErrorTests()
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

    private async Task<string> Answer(object output, PredictionInterval? interval = null)
    {
        var id = PredictionIds.New();
        await _logger.LogBatchAsync("m", "exp-001",
        [
            new PredictionLogEntry("m", "exp-001", new Dictionary<string, object> { ["x"] = 1 }, output, null, DateTimeOffset.UtcNow, id, interval),
        ]);
        return id;
    }

    [Fact]
    public async Task The_error_is_the_weighted_mean_of_the_squared_differences_and_only_when_a_value_answer_is_asked_for()
    {
        await _collector.RecordFeedbackAsync(await Answer(10.0), 12.0);   // 2² = 4
        await _collector.RecordFeedbackAsync(await Answer(20.0), 17.0);   // 3² = 9

        var asked = await _collector.CalculateMetricsAsync("m", valueAnswers: true);
        var notAsked = await _collector.CalculateMetricsAsync("m");

        asked.ValueError.Should().NotBeNull();
        asked.ValueError!.MeanSquaredError.Should().BeApproximately(6.5, 1e-9);
        asked.ValueError.Rmse.Should().BeApproximately(Math.Sqrt(6.5), 1e-9);
        asked.ValueError.SquaredErrorSd.Should().BeApproximately(2.5, 1e-9);   // squares 4 and 9: mean 6.5, each 2.5 away
        asked.ValueError.N.Should().Be(2);
        asked.ValueError.EffectiveN.Should().BeApproximately(2.0, 1e-9);
        notAsked.ValueError.Should().BeNull();
    }

    [Fact]
    public async Task A_row_counts_by_its_weight_and_the_rows_worth_shrinks_with_unequal_weights()
    {
        await _collector.RecordFeedbackAsync(await Answer(10.0), 12.0, weight: 3.0);   // 4, weight 3
        await _collector.RecordFeedbackAsync(await Answer(20.0), 17.0, weight: 1.0);   // 9, weight 1

        var error = (await _collector.CalculateMetricsAsync("m", valueAnswers: true)).ValueError!;

        error.MeanSquaredError.Should().BeApproximately((3 * 4 + 1 * 9) / 4.0, 1e-9);
        error.EffectiveN.Should().BeApproximately(16.0 / 10.0, 1e-9);   // Kish: (Σw)² ÷ Σw²
        error.N.Should().Be(2);
    }

    [Fact]
    public async Task An_answer_that_stated_an_interval_is_read_as_well_as_one_that_did_not()
    {
        await _collector.RecordFeedbackAsync(await Answer(10.0, new PredictionInterval(9, 11, 0.9)), 14.0);   // 16, outside its interval
        await _collector.RecordFeedbackAsync(await Answer(20.0), 22.0);                                       // 4

        var metrics = await _collector.CalculateMetricsAsync("m", valueAnswers: true);

        metrics.ValueError!.N.Should().Be(2);
        metrics.ValueError.MeanSquaredError.Should().BeApproximately(10.0, 1e-9);
        metrics.IntervalCoverage.Should().NotBeNull();   // the coverage reading is unchanged
    }

    [Fact]
    public async Task Feedback_that_is_not_a_number_on_both_sides_is_not_read_and_a_row_of_weight_zero_counts_for_nothing()
    {
        await _collector.RecordFeedbackAsync(await Answer("yes"), "no");
        await _collector.RecordFeedbackAsync(await Answer(10.0), "unknown");
        await _collector.RecordFeedbackAsync(await Answer(10.0), 13.0, weight: 0.0);
        await _collector.RecordFeedbackAsync(await Answer(10.0), "13");   // a number written as text is a number: 3² = 9

        var metrics = await _collector.CalculateMetricsAsync("m", valueAnswers: true);

        metrics.ValueError!.N.Should().Be(1);
        metrics.ValueError.MeanSquaredError.Should().BeApproximately(9.0, 1e-9);
    }

    [Fact]
    public async Task Nothing_to_read_is_null_not_zero()
    {
        await _collector.RecordFeedbackAsync(await Answer("yes"), "no");

        (await _collector.CalculateMetricsAsync("m", valueAnswers: true)).ValueError.Should().BeNull();
        (await _collector.CalculateMetricsAsync("nothing-here", valueAnswers: true)).ValueError.Should().BeNull();
    }

    [Fact]
    public async Task The_latest_feedback_for_a_prediction_stands()
    {
        var id = await Answer(10.0);
        await _collector.RecordFeedbackAsync(id, 30.0);
        await Task.Delay(5);
        await _collector.RecordFeedbackAsync(id, 11.0);   // corrected: 1² = 1

        var error = (await _collector.CalculateMetricsAsync("m", valueAnswers: true)).ValueError!;

        error.N.Should().Be(1);
        error.MeanSquaredError.Should().BeApproximately(1.0, 1e-9);
    }
}
