using FluentAssertions;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;

namespace MLoop.Ops.Tests;

public class CompositeRetrainingTriggerTests : IDisposable
{
    private readonly string _testDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");

    public CompositeRetrainingTriggerTests() => Directory.CreateDirectory(_testDir);

    public void Dispose()
    {
        try { Directory.Delete(_testDir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public async Task Defaults_are_every_condition_the_project_measures()
    {
        var trigger = new CompositeRetrainingTrigger(_testDir, new FileFeedbackCollector(_testDir), new FilePredictionLogger(_testDir));

        var defaults = await trigger.GetDefaultConditionsAsync("m");

        defaults.Select(c => c.Type).Should().Equal(
            ConditionType.AccuracyDrop, ConditionType.IntervalCoverage, ConditionType.ErrorRise, ConditionType.FeedbackVolume, ConditionType.TimeBased, ConditionType.DataDrift);
    }

    [Fact]
    public async Task Each_condition_is_measured_by_the_trigger_that_can_measure_it()
    {
        var trigger = new CompositeRetrainingTrigger(_testDir, new FileFeedbackCollector(_testDir), new FilePredictionLogger(_testDir));

        var evaluation = await trigger.EvaluateAsync("m", RetrainingDefaults.All);

        // No training history: the interval condition is met by the time trigger. No feedback: the
        // feedback conditions are measured (not "unsupported") and are not met. No production model:
        // drift is not met and says why.
        evaluation.ConditionResults.Select(r => r.Condition).Should().Equal(RetrainingDefaults.All);
        evaluation.ConditionResults.Single(r => r.Condition.Type == ConditionType.TimeBased).IsMet.Should().BeTrue();
        evaluation.ConditionResults.Where(r => r.Condition.Type != ConditionType.TimeBased)
            .Should().OnlyContain(r => !r.IsMet && (r.Details == null || !r.Details.Contains("not supported")));
        evaluation.ShouldRetrain.Should().BeTrue();
        evaluation.RecommendedAction.Should().Contain(RetrainingDefaults.IntervalName);
    }

    /// <summary>A production slot the way promotion leaves it, as far as the trigger reads it: the experiment, its task and the metrics it recorded on its held-out rows.</summary>
    private void Promote(string model, string task, string metricsJson)
    {
        var production = Path.Combine(_testDir, "models", model, "production");
        Directory.CreateDirectory(production);
        File.WriteAllText(Path.Combine(production, "metadata.json"),
            $$"""{"modelName":"{{model}}","experimentId":"exp-001","task":"{{task}}","metrics":{{metricsJson}}}""");
    }

    private async Task GiveFeedback(string model, int rows, double error)
    {
        var logger = new FilePredictionLogger(_testDir);
        var collector = new FileFeedbackCollector(_testDir);
        for (var i = 0; i < rows; i++)
        {
            var id = PredictionIds.New();
            await logger.LogBatchAsync(model, "exp-001",
            [
                new PredictionLogEntry(model, "exp-001", new Dictionary<string, object> { ["x"] = i }, 50.0 + i, null, DateTimeOffset.UtcNow, id),
            ]);
            await collector.RecordFeedbackAsync(id, 50.0 + i + (i % 2 == 0 ? error : -error));
        }
    }

    [Fact]
    public async Task Error_rise_compares_the_feedback_with_the_error_production_recorded_on_its_held_out_rows()
    {
        Promote("price", "regression", """{"r_squared":0.9,"rmse":2.0}""");
        await GiveFeedback("price", rows: 60, error: 6.0);   // three times the held-out error
        var trigger = new CompositeRetrainingTrigger(_testDir, new FileFeedbackCollector(_testDir), new FilePredictionLogger(_testDir));

        var evaluation = await trigger.EvaluateAsync("price", [RetrainingDefaults.ErrorRise()]);

        var result = evaluation.ConditionResults.Single();
        result.IsMet.Should().BeTrue();
        result.CurrentValue.Should().BeApproximately(3.0, 1e-9);
        evaluation.RecommendedAction.Should().Contain("error_rise");
    }

    [Fact]
    public async Task Error_rise_stays_quiet_when_the_error_is_what_it_was_at_training()
    {
        Promote("price", "regression", """{"rmse":2.0}""");
        await GiveFeedback("price", rows: 60, error: 2.0);
        var trigger = new CompositeRetrainingTrigger(_testDir, new FileFeedbackCollector(_testDir), new FilePredictionLogger(_testDir));

        var evaluation = await trigger.EvaluateAsync("price", [RetrainingDefaults.ErrorRise()]);

        evaluation.ConditionResults.Single().IsMet.Should().BeFalse();
        evaluation.ShouldRetrain.Should().BeFalse();
    }

    [Fact]
    public async Task Error_rise_does_not_apply_to_a_ranking_model_that_recorded_no_rmse()
    {
        Promote("rank", "ranking", """{"ndcg":0.8}""");
        await GiveFeedback("rank", rows: 60, error: 6.0);
        var trigger = new CompositeRetrainingTrigger(_testDir, new FileFeedbackCollector(_testDir), new FilePredictionLogger(_testDir));

        var evaluation = await trigger.EvaluateAsync("rank", [RetrainingDefaults.ErrorRise()]);

        var result = evaluation.ConditionResults.Single();
        result.IsMet.Should().BeFalse();
        result.Details.Should().Contain("does not apply");
    }
}
