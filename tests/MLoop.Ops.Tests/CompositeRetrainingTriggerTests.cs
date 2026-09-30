using FluentAssertions;
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
        var trigger = new CompositeRetrainingTrigger(_testDir, new FileFeedbackCollector(_testDir));

        var defaults = await trigger.GetDefaultConditionsAsync("m");

        defaults.Select(c => c.Type).Should().Equal(
            ConditionType.AccuracyDrop, ConditionType.FeedbackVolume, ConditionType.TimeBased);
    }

    [Fact]
    public async Task Each_condition_is_measured_by_the_trigger_that_can_measure_it()
    {
        var trigger = new CompositeRetrainingTrigger(_testDir, new FileFeedbackCollector(_testDir));

        var evaluation = await trigger.EvaluateAsync("m", RetrainingDefaults.All);

        // No training history: the interval condition is met by the time trigger. No feedback: the
        // feedback conditions are measured (not "unsupported") and are not met.
        evaluation.ConditionResults.Select(r => r.Condition).Should().Equal(RetrainingDefaults.All);
        evaluation.ConditionResults.Single(r => r.Condition.Type == ConditionType.TimeBased).IsMet.Should().BeTrue();
        evaluation.ConditionResults.Where(r => r.Condition.Type != ConditionType.TimeBased)
            .Should().OnlyContain(r => !r.IsMet && (r.Details == null || !r.Details.Contains("not supported")));
        evaluation.ShouldRetrain.Should().BeTrue();
        evaluation.RecommendedAction.Should().Contain(RetrainingDefaults.IntervalName);
    }
}
