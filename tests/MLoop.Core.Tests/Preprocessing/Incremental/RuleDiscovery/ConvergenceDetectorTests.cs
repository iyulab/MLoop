using MLoop.Core.Preprocessing.Incremental.RuleDiscovery;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.RuleDiscovery;

public sealed class ConvergenceDetectorTests
{
    private readonly ConvergenceDetector _detector = new();

    [Fact]
    public void HasConverged_NoRules_ReturnsFalse()
    {
        var previous = Array.Empty<PreprocessingRule>();
        var current = Array.Empty<PreprocessingRule>();

        var result = _detector.HasConverged(previous, current);

        Assert.False(result);
    }

    [Fact]
    public void HasConverged_IdenticalRules_ReturnsTrue()
    {
        var rule1 = CreateTestRule("rule1", "Col1", PatternType.MissingValue);
        var rule2 = CreateTestRule("rule2", "Col2", PatternType.WhitespaceIssue);

        var previous = new[] { rule1, rule2 };
        var current = new[] { rule1, rule2 };

        var result = _detector.HasConverged(previous, current, threshold: 0.02);

        Assert.True(result);
    }

    [Fact]
    public void HasConverged_OneNewRule_SmallChange_ReturnsTrue()
    {
        var rule1 = CreateTestRule("rule1", "Col1", PatternType.MissingValue);
        var rule2 = CreateTestRule("rule2", "Col2", PatternType.WhitespaceIssue);
        var rule3 = CreateTestRule("rule3", "Col3", PatternType.OutlierAnomaly);

        var previous = new[] { rule1, rule2 };
        var current = new[] { rule1, rule2, rule3 }; // 1 new rule = 50% change

        var result = _detector.HasConverged(previous, current, threshold: 0.6); // 60% threshold

        Assert.True(result);
    }

    [Fact]
    public void HasConverged_ManyNewRules_ReturnsFalse()
    {
        var rule1 = CreateTestRule("rule1", "Col1", PatternType.MissingValue);
        var rule2 = CreateTestRule("rule2", "Col2", PatternType.WhitespaceIssue);
        var rule3 = CreateTestRule("rule3", "Col3", PatternType.OutlierAnomaly);

        var previous = new[] { rule1 };
        var current = new[] { rule1, rule2, rule3 }; // 2 new rules = 200% change

        var result = _detector.HasConverged(previous, current, threshold: 0.02);

        Assert.False(result);
    }

    [Fact]
    public void HasConverged_RulesRemoved_CountsAsChange()
    {
        var rule1 = CreateTestRule("rule1", "Col1", PatternType.MissingValue);
        var rule2 = CreateTestRule("rule2", "Col2", PatternType.WhitespaceIssue);
        var rule3 = CreateTestRule("rule3", "Col3", PatternType.OutlierAnomaly);

        var previous = new[] { rule1, rule2, rule3 };
        var current = new[] { rule1 }; // 2 removed = 66% change rate

        var result = _detector.HasConverged(previous, current, threshold: 0.50);

        Assert.False(result); // Should not converge with 66% change and 50% threshold
    }

    [Fact]
    public void GetConvergenceInfo_CalculatesMetricsCorrectly()
    {
        var rule1 = CreateTestRule("rule1", "Col1", PatternType.MissingValue);
        var rule2 = CreateTestRule("rule2", "Col2", PatternType.WhitespaceIssue);
        var rule3 = CreateTestRule("rule3", "Col3", PatternType.OutlierAnomaly);
        var rule4 = CreateTestRule("rule4", "Col4", PatternType.EncodingIssue);

        var previous = new[] { rule1, rule2 };
        var current = new[] { rule1, rule2, rule3, rule4 }; // 2 new rules

        var info = _detector.GetConvergenceInfo(previous, current, threshold: 0.02);

        Assert.Equal(2, info.NewRules);
        Assert.Equal(0, info.ModifiedRules);
        Assert.Equal(0, info.RemovedRules);
        Assert.Equal(2, info.StableRules);
        Assert.Equal(4, info.TotalRules);
        Assert.Equal(2, info.PreviousRules);
        Assert.Equal(1.0, info.ChangeRate); // 2 new / 2 previous = 100%
        Assert.False(info.HasConverged);
    }

    [Fact]
    public void GetConvergenceInfo_ModifiedRules_CountsAsChange()
    {
        var rule1 = CreateTestRule("rule1", "Col1", PatternType.MissingValue, confidence: 0.5);
        var rule1Modified = CreateTestRule("rule1", "Col1", PatternType.MissingValue, confidence: 0.9);
        var rule2 = CreateTestRule("rule2", "Col2", PatternType.WhitespaceIssue);

        var previous = new[] { rule1, rule2 };
        var current = new[] { rule1Modified, rule2 };

        var info = _detector.GetConvergenceInfo(previous, current, threshold: 0.02);

        Assert.Equal(1, info.ModifiedRules);
        Assert.Equal(0, info.NewRules);
        Assert.Equal(1, info.StableRules);
    }

    [Fact]
    public void GetConvergenceInfo_Status_ReflectsConvergence()
    {
        var rule1 = CreateTestRule("rule1", "Col1", PatternType.MissingValue);

        var previous = new[] { rule1 };
        var current = new[] { rule1 };

        var info = _detector.GetConvergenceInfo(previous, current, threshold: 0.02);

        Assert.Contains("Converged", info.Status);
    }

    private static PreprocessingRule CreateTestRule(
        string id,
        string columnName,
        PatternType patternType,
        double confidence = 0.8)
    {
        return new PreprocessingRule
        {
            Id = id,
            Type = PreprocessingRuleType.MissingValueStrategy,
            ColumnNames = new[] { columnName },
            Description = $"Test rule for {columnName}",
            PatternType = patternType,
            Confidence = confidence,
            RequiresHITL = false,
            Priority = 5,
            AffectedRows = 100,
            DiscoveredInStage = 1
        };
    }
}
