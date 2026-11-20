using FluentAssertions;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;

namespace MLoop.Ops.Tests;

/// <summary>
/// <c>error_rise</c>: a model that answers values is judged by how far its answers were from the actual values its feedback names, against
/// the RMSE it made on the rows it held out at training — on the evidence (the 95 % lower end of the feedback RMSE), not on the point estimate,
/// and only with enough rows. A class answer has no distance, and a model that recorded no held-out RMSE has nothing to compare with: each says so.
/// </summary>
public class ErrorRiseTests : IDisposable
{
    private readonly string _testDir = Path.Combine(Path.GetTempPath(), $"mloop-error-rise-{Guid.NewGuid():N}");
    private readonly FileFeedbackCollector _collector;
    private readonly FilePredictionLogger _logger;
    // An explicit ratio: these tests state what the rule does at a ratio, and the default one is a measured number that may move.
    private static readonly RetrainingCondition Rise = RetrainingDefaults.ErrorRise(1.5);

    public ErrorRiseTests()
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

    private FeedbackBasedTrigger TriggerFor(string task, double? heldOutRmse) => new(
        _collector,
        (_, _) => Task.FromResult<string?>("exp-001"),
        (_, _) => Task.FromResult<string?>(task),
        (_, _) => Task.FromResult(heldOutRmse));

    /// <summary>One answer per error, each answer 100 + its index, each actual value the answer plus the error (so the feedback RMSE is known).</summary>
    private async Task Feedback(string model, IReadOnlyList<double> errors, Func<int, double?>? weight = null, PredictionInterval? interval = null, Func<double, object>? actual = null)
    {
        for (var i = 0; i < errors.Count; i++)
        {
            var id = PredictionIds.New();
            var answer = 100.0 + i;
            await _logger.LogBatchAsync(model, "exp-001",
            [
                new PredictionLogEntry(model, "exp-001", new Dictionary<string, object> { ["x"] = i }, answer, null, DateTimeOffset.UtcNow, id, interval),
            ]);
            var value = answer + errors[i];
            await _collector.RecordFeedbackAsync(id, actual is null ? value : actual(value), weight: weight?.Invoke(i));
        }
    }

    private static IReadOnlyList<double> Alternating(int count, double size) => [.. Enumerable.Range(0, count).Select(i => i % 2 == 0 ? size : -size)];

    [Fact]
    public async Task An_error_as_large_as_the_held_out_error_is_not_a_rise()
    {
        await Feedback("reg", Alternating(60, 1.0));

        var result = await TriggerFor("regression", 1.0).EvaluateAsync("reg", [Rise]);

        var r = result.ConditionResults.Single();
        r.IsMet.Should().BeFalse();
        r.CurrentValue.Should().BeApproximately(1.0, 1e-9);
        r.Details.Should().Contain("1.00×").And.Contain("is under 1.5×");
        result.ShouldRetrain.Should().BeFalse();
    }

    [Fact]
    public async Task An_error_established_over_the_ratio_asks_for_retraining()
    {
        await Feedback("reg", Alternating(60, 3.0));

        var result = await TriggerFor("regression", 1.0).EvaluateAsync("reg", [Rise]);

        var r = result.ConditionResults.Single();
        r.IsMet.Should().BeTrue();
        r.CurrentValue.Should().BeApproximately(3.0, 1e-9);
        r.Details.Should().Contain("3.00×").And.Contain("95% interval from");
        result.ShouldRetrain.Should().BeTrue();
        result.RecommendedAction.Should().Contain(RetrainingDefaults.ErrorRiseName);
    }

    [Fact]
    public async Task One_large_error_among_thirty_is_a_point_estimate_not_an_established_rise()
    {
        // 29 errors of 0.1 and one of 10: the feedback RMSE is about 1.8 times the held-out error on its face, but a single row carries it —
        // the 95 % lower end of the RMSE is far below the ratio, so it is not established.
        IReadOnlyList<double> errors = [.. Enumerable.Repeat(0.1, 29), 10.0];
        await Feedback("reg", errors);

        var result = await TriggerFor("regression", 1.0).EvaluateAsync("reg", [Rise]);

        var r = result.ConditionResults.Single();
        r.CurrentValue.Should().BeGreaterThan(1.5);
        r.IsMet.Should().BeFalse();
        r.Details.Should().Contain("not established");
    }

    [Fact]
    public async Task Too_little_feedback_is_not_judged()
    {
        await Feedback("reg", Alternating(10, 5.0));

        var result = await TriggerFor("regression", 1.0).EvaluateAsync("reg", [Rise]);

        var r = result.ConditionResults.Single();
        r.IsMet.Should().BeFalse();
        r.Details.Should().Contain("too few to judge").And.Contain($"needs {RetrainingDefaults.MinimumAccuracyFeedback}");
    }

    [Fact]
    public async Task A_weighted_sample_counts_by_its_effective_rows()
    {
        // 40 rows, one of them standing for a thousand: the evidence is a handful of rows, whatever the count says.
        await Feedback("reg", Alternating(40, 5.0), weight: i => i == 0 ? 1000.0 : 1.0);

        var result = await TriggerFor("regression", 1.0).EvaluateAsync("reg", [Rise]);

        var r = result.ConditionResults.Single();
        r.IsMet.Should().BeFalse();
        r.Details.Should().Contain("effective rows of feedback").And.Contain("too few to judge");
    }

    [Fact]
    public async Task An_answer_that_stated_an_interval_has_a_point_error_too()
    {
        // The interval [90, 110] around each answer holds none of these actual values and coverage reads them; the answer is off by 20 as well.
        await Feedback("reg", Alternating(60, 20.0), interval: new PredictionInterval(90, 110, 0.9));

        var result = await TriggerFor("regression", 1.0).EvaluateAsync("reg", [Rise]);

        result.ConditionResults.Single().IsMet.Should().BeTrue();
    }

    [Theory]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    [InlineData("ner")]
    public async Task A_class_or_text_answer_has_no_distance_and_says_so(string task)
    {
        await Feedback("clf", Alternating(60, 3.0));

        var result = await TriggerFor(task, 1.0).EvaluateAsync("clf", [Rise]);

        var r = result.ConditionResults.Single();
        r.IsMet.Should().BeFalse();
        r.CurrentValue.Should().BeNull();
        r.Details.Should().Contain("does not apply").And.Contain("a class or a text");
    }

    [Fact]
    public async Task A_ranking_score_has_no_error_to_read_and_says_why()
    {
        await Feedback("rank", Alternating(60, 3.0));

        var result = await TriggerFor("ranking", null).EvaluateAsync("rank", [Rise]);

        var r = result.ConditionResults.Single();
        r.IsMet.Should().BeFalse();
        r.CurrentValue.Should().BeNull();
        r.Details.Should().Contain("does not apply").And.Contain("not on the scale of its labels");
    }

    [Fact]
    public async Task A_value_model_that_recorded_no_held_out_error_has_nothing_to_compare_with()
    {
        await Feedback("reg", Alternating(60, 3.0));

        var result = await TriggerFor("regression", null).EvaluateAsync("reg", [Rise]);

        var r = result.ConditionResults.Single();
        r.IsMet.Should().BeFalse();
        r.CurrentValue.Should().BeNull();
        r.Details.Should().Contain("recorded no RMSE");
    }

    [Fact]
    public async Task Feedback_whose_actual_value_is_not_a_number_is_not_read()
    {
        await Feedback("reg", Alternating(60, 3.0), actual: _ => "unknown");

        var result = await TriggerFor("regression", 1.0).EvaluateAsync("reg", [Rise]);

        var r = result.ConditionResults.Single();
        r.IsMet.Should().BeFalse();
        r.CurrentValue.Should().BeNull();
        r.Details.Should().Contain("No error data available");
    }

    [Fact]
    public void The_default_condition_is_in_the_list_every_surface_reads()
    {
        var standard = RetrainingDefaults.ErrorRise();
        standard.Type.Should().Be(ConditionType.ErrorRise);
        standard.Name.Should().Be(RetrainingDefaults.ErrorRiseName).And.Be("error_rise");
        standard.Threshold.Should().Be(RetrainingDefaults.ErrorRiseRatio);
        RetrainingDefaults.All.Should().Contain(standard);
    }

    [Fact]
    public void The_lower_bound_of_the_feedback_error_narrows_as_the_evidence_grows_and_never_goes_below_zero()
    {
        // Squares of mean 4 and spread 2: at 30 effective rows the lower end sits below the mean, at 3,000 it is nearly the mean, with few rows it floors at 0.
        var thirty = RetrainingDefaults.ErrorLowerBound(new ValueError(4.0, 2.0, 30, 30));
        var thousands = RetrainingDefaults.ErrorLowerBound(new ValueError(4.0, 2.0, 3000, 3000));
        var floor = RetrainingDefaults.ErrorLowerBound(new ValueError(0.5, 20.0, 4, 4));

        thirty.Should().BeLessThan(2.0).And.BeGreaterThan(1.5);
        thousands.Should().BeGreaterThan(thirty).And.BeLessThan(2.0);
        floor.Should().Be(0.0);
    }
}
