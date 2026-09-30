using MLoop.CLI.Commands;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;

namespace MLoop.Tests.Commands;

public class TriggerCommandTests
{
    #region BuildConditions

    [Fact]
    public void BuildConditions_NoThresholds_ReturnsTheSharedDefaults()
    {
        var result = TriggerCommand.BuildConditions(null, null);

        Assert.Equal(RetrainingDefaults.All, result);
    }

    [Fact]
    public void BuildConditions_Accuracy_ReplacesOnlyTheAccuracyDefault()
    {
        var result = TriggerCommand.BuildConditions(0.85, null);

        Assert.Equal(RetrainingDefaults.All.Count, result.Count);
        var accuracy = Assert.Single(result, c => c.Type == ConditionType.AccuracyDrop);
        Assert.Equal(0.85, accuracy.Threshold);
        Assert.Equal(RetrainingDefaults.AccuracyName, accuracy.Name);
        Assert.Contains("85", accuracy.Description!);
        Assert.Contains(result, c => c.Type == ConditionType.TimeBased);
        Assert.Contains(RetrainingDefaults.Feedback(), result);
    }

    [Fact]
    public void BuildConditions_Feedback_ReplacesOnlyTheFeedbackDefault()
    {
        var result = TriggerCommand.BuildConditions(null, 50);

        var feedback = Assert.Single(result, c => c.Type == ConditionType.FeedbackVolume);
        Assert.Equal(50, feedback.Threshold);
        // One name for one condition, whichever surface built it.
        Assert.Equal(RetrainingDefaults.FeedbackVolumeName, feedback.Name);
        Assert.Contains(RetrainingDefaults.Accuracy(), result);
    }

    [Fact]
    public void BuildConditions_BothThresholds_KeepTheTimeCondition()
    {
        var result = TriggerCommand.BuildConditions(0.9, 50);

        Assert.Equal(
            [ConditionType.AccuracyDrop, ConditionType.FeedbackVolume, ConditionType.TimeBased],
            result.Select(c => c.Type));
    }

    #endregion

    #region FormatThreshold

    [Fact]
    public void FormatThreshold_AccuracyDrop_ShowsPercentage()
    {
        var condition = new RetrainingCondition(ConditionType.AccuracyDrop, "test", 0.85);

        var result = TriggerCommand.FormatThreshold(condition);

        Assert.Contains("< ", result);
        Assert.Contains("85", result);
        Assert.Contains("[yellow]", result);
    }

    [Fact]
    public void FormatThreshold_FeedbackVolume_ShowsCount()
    {
        var condition = new RetrainingCondition(ConditionType.FeedbackVolume, "test", 100);

        var result = TriggerCommand.FormatThreshold(condition);

        Assert.Contains(">=", result);
        Assert.Contains("100", result);
    }

    [Fact]
    public void FormatThreshold_TimeBased_ShowsDays()
    {
        var condition = new RetrainingCondition(ConditionType.TimeBased, "test", 30);

        var result = TriggerCommand.FormatThreshold(condition);

        Assert.Contains("30", result);
        Assert.Contains("days", result);
    }

    [Fact]
    public void FormatThreshold_UnknownType_ShowsRawValue()
    {
        var condition = new RetrainingCondition(ConditionType.DataDrift, "test", 0.5);

        var result = TriggerCommand.FormatThreshold(condition);

        Assert.Contains("0.5", result);
    }

    #endregion

    #region FormatCurrentValue

    [Fact]
    public void FormatCurrentValue_AccuracyDrop_Met_ShowsGreen()
    {
        var condition = new RetrainingCondition(ConditionType.AccuracyDrop, "test", 0.85);
        var cr = new ConditionResult(condition, IsMet: true, CurrentValue: 0.7, Details: null);

        var result = TriggerCommand.FormatCurrentValue(cr);

        Assert.Contains("[green]", result);
        Assert.Contains("70", result);
    }

    [Fact]
    public void FormatCurrentValue_AccuracyDrop_NotMet_ShowsWhite()
    {
        var condition = new RetrainingCondition(ConditionType.AccuracyDrop, "test", 0.85);
        var cr = new ConditionResult(condition, IsMet: false, CurrentValue: 0.9, Details: null);

        var result = TriggerCommand.FormatCurrentValue(cr);

        Assert.Contains("[white]", result);
    }

    [Fact]
    public void FormatCurrentValue_FeedbackVolume_ShowsCount()
    {
        var condition = new RetrainingCondition(ConditionType.FeedbackVolume, "test", 100);
        var cr = new ConditionResult(condition, IsMet: true, CurrentValue: 150, Details: null);

        var result = TriggerCommand.FormatCurrentValue(cr);

        Assert.Contains("150", result);
    }

    [Fact]
    public void FormatCurrentValue_TimeBased_ShowsDays()
    {
        var condition = new RetrainingCondition(ConditionType.TimeBased, "test", 30);
        var cr = new ConditionResult(condition, IsMet: false, CurrentValue: 15, Details: null);

        var result = TriggerCommand.FormatCurrentValue(cr);

        Assert.Contains("15", result);
        Assert.Contains("days", result);
    }

    #endregion
}
