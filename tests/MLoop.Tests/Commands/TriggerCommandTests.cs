using MLoop.CLI.Commands;
using MLoop.Ops.Interfaces;

namespace MLoop.Tests.Commands;

public class TriggerCommandTests
{
    #region BuildConditions

    [Fact]
    public void BuildConditions_NoThresholds_ReturnsEmpty()
    {
        var result = TriggerCommand.BuildConditions(null, null);

        Assert.Empty(result);
    }

    [Fact]
    public void BuildConditions_AccuracyOnly_ReturnsSingleCondition()
    {
        var result = TriggerCommand.BuildConditions(0.85, null);

        Assert.Single(result);
        Assert.Equal(ConditionType.AccuracyDrop, result[0].Type);
        Assert.Equal(0.85, result[0].Threshold);
        Assert.Equal("accuracy_threshold", result[0].Name);
        Assert.Contains("85", result[0].Description!);
    }

    [Fact]
    public void BuildConditions_FeedbackOnly_ReturnsSingleCondition()
    {
        var result = TriggerCommand.BuildConditions(null, 100);

        Assert.Single(result);
        Assert.Equal(ConditionType.FeedbackVolume, result[0].Type);
        Assert.Equal(100, result[0].Threshold);
        Assert.Equal("feedback_threshold", result[0].Name);
        Assert.Contains("100", result[0].Description!);
    }

    [Fact]
    public void BuildConditions_BothThresholds_ReturnsTwoConditions()
    {
        var result = TriggerCommand.BuildConditions(0.9, 50);

        Assert.Equal(2, result.Count);
        Assert.Equal(ConditionType.AccuracyDrop, result[0].Type);
        Assert.Equal(ConditionType.FeedbackVolume, result[1].Type);
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
