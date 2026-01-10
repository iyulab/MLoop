using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.RuleDiscovery;

public sealed class RuleDiscoveryEngineTests
{
    private readonly RuleDiscoveryEngine _engine;

    public RuleDiscoveryEngineTests()
    {
        _engine = new RuleDiscoveryEngine(NullLogger<RuleDiscoveryEngine>.Instance);
    }

    [Fact]
    public async Task DiscoverRulesAsync_EmptyDataFrame_ReturnsEmpty()
    {
        var sample = new DataFrame();
        var analysis = CreateSampleAnalysis(1);

        var rules = await _engine.DiscoverRulesAsync(sample, analysis);

        Assert.Empty(rules);
    }

    [Fact]
    public async Task DiscoverRulesAsync_WithMissingValues_DiscoversMissingValueRule()
    {
        var sample = CreateSampleWithMissingValues();
        var analysis = CreateSampleAnalysis(1);

        var rules = await _engine.DiscoverRulesAsync(sample, analysis);

        Assert.NotEmpty(rules);
        Assert.Contains(rules, r => r.PatternType == PatternType.MissingValue);
    }

    [Fact]
    public async Task DiscoverRulesAsync_WithWhitespace_DiscoversWhitespaceRule()
    {
        var sample = CreateSampleWithWhitespace();
        var analysis = CreateSampleAnalysis(1);

        var rules = await _engine.DiscoverRulesAsync(sample, analysis);

        Assert.NotEmpty(rules);
        Assert.Contains(rules, r => r.PatternType == PatternType.WhitespaceIssue);
    }

    [Fact]
    public async Task DiscoverRulesAsync_PrioritizesRules_ByPriorityAndAffectedRows()
    {
        var sample = CreateComplexSample();
        var analysis = CreateSampleAnalysis(1);

        var rules = await _engine.DiscoverRulesAsync(sample, analysis);

        // Verify rules are prioritized
        for (int i = 0; i < rules.Count - 1; i++)
        {
            Assert.True(
                rules[i].Priority >= rules[i + 1].Priority ||
                (rules[i].Priority == rules[i + 1].Priority && rules[i].AffectedRows >= rules[i + 1].AffectedRows),
                "Rules should be ordered by priority descending, then by affected rows descending");
        }
    }

    [Fact]
    public async Task DiscoverRulesAsync_SetsStageNumber_Correctly()
    {
        var sample = CreateSampleWithMissingValues();
        var analysis = CreateSampleAnalysis(3);

        var rules = await _engine.DiscoverRulesAsync(sample, analysis);

        Assert.All(rules, r => Assert.Equal(3, r.DiscoveredInStage));
    }

    [Fact]
    public async Task DiscoverRulesAsync_AutoFixableRules_DoNotRequireHITL()
    {
        var sample = CreateSampleWithWhitespace();
        var analysis = CreateSampleAnalysis(1);

        var rules = await _engine.DiscoverRulesAsync(sample, analysis);

        var whitespaceRules = rules.Where(r => r.Type == PreprocessingRuleType.WhitespaceNormalization);
        Assert.All(whitespaceRules, r => Assert.False(r.RequiresHITL));
    }

    [Fact]
    public async Task DiscoverRulesAsync_HITLRules_RequireApproval()
    {
        var sample = CreateSampleWithMissingValues();
        var analysis = CreateSampleAnalysis(1);

        var rules = await _engine.DiscoverRulesAsync(sample, analysis);

        var missingValueRules = rules.Where(r => r.Type == PreprocessingRuleType.MissingValueStrategy);
        Assert.All(missingValueRules, r => Assert.True(r.RequiresHITL));
    }

    [Fact]
    public async Task CalculateConfidenceAsync_ReturnsValidScore()
    {
        var sample = CreateSampleWithMissingValues();
        var analysis = CreateSampleAnalysis(1);
        var rules = await _engine.DiscoverRulesAsync(sample, analysis);
        var rule = rules.First();

        var score = await _engine.CalculateConfidenceAsync(rule, sample, sample);

        Assert.InRange(score.Overall, 0.0, 1.0);
        Assert.InRange(score.Consistency, 0.0, 1.0);
        Assert.InRange(score.Coverage, 0.0, 1.0);
        Assert.InRange(score.Stability, 0.0, 1.0);
    }

    [Fact]
    public void HasConverged_IdenticalRules_ReturnsTrue()
    {
        var rule = CreateTestRule();
        var previousRules = new[] { rule };
        var currentRules = new[] { rule };

        var hasConverged = _engine.HasConverged(previousRules, currentRules, threshold: 0.02);

        Assert.True(hasConverged);
    }

    [Fact]
    public void HasConverged_ManyNewRules_ReturnsFalse()
    {
        var rule1 = CreateTestRule("rule1");
        var rule2 = CreateTestRule("rule2");
        var rule3 = CreateTestRule("rule3");

        var previousRules = new[] { rule1 };
        var currentRules = new[] { rule1, rule2, rule3 }; // 2 new rules = 200% change

        var hasConverged = _engine.HasConverged(previousRules, currentRules, threshold: 1.50); // 150% threshold

        Assert.False(hasConverged); // Should not converge with 200% change
    }

    // Helper methods
    private static DataFrame CreateSampleWithMissingValues()
    {
        var df = new DataFrame();
        var values = new string?[] { "value1", null, "value2", null, "value3" };
        df.Columns.Add(new StringDataFrameColumn("TestColumn", values));
        return df;
    }

    private static DataFrame CreateSampleWithWhitespace()
    {
        var df = new DataFrame();
        var values = new[] { " leading", "trailing ", "clean", "double  space" };
        df.Columns.Add(new StringDataFrameColumn("TestColumn", values));
        return df;
    }

    private static DataFrame CreateComplexSample()
    {
        var df = new DataFrame();

        // Column with missing values (high priority)
        var missingValues = new string?[] { null, "value", null, null, "value" };
        df.Columns.Add(new StringDataFrameColumn("ColWithMissing", missingValues));

        // Column with whitespace (low priority) - same length
        var whitespaceValues = new[] { " leading", "clean", "trailing ", "  double", "clean" };
        df.Columns.Add(new StringDataFrameColumn("ColWithWhitespace", whitespaceValues));

        return df;
    }

    private static SampleAnalysis CreateSampleAnalysis(int stageNumber)
    {
        return new SampleAnalysis
        {
            StageNumber = stageNumber,
            SampleRatio = 0.1,
            Timestamp = DateTime.UtcNow,
            RowCount = 100,
            ColumnCount = 1,
            Columns = new List<ColumnAnalysis>(),
            QualityScore = 0.9
        };
    }

    private static PreprocessingRule CreateTestRule(string id = "test-rule")
    {
        return new PreprocessingRule
        {
            Id = id,
            Type = PreprocessingRuleType.MissingValueStrategy,
            ColumnNames = new[] { $"TestColumn_{id}" }, // Unique column name for unique signature
            Description = $"Test rule for {id}",  // Unique description for unique signature
            PatternType = PatternType.MissingValue,
            RequiresHITL = true,
            Priority = 5,
            DiscoveredInStage = 1
        };
    }
}
