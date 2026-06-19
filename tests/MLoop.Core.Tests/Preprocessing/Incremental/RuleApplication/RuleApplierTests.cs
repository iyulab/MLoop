using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.Core.Preprocessing.Incremental.RuleApplication;
using MLoop.Core.Preprocessing.Incremental.RuleApplication.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.RuleApplication;

public class RuleApplierTests
{
    private readonly RuleApplier _applier;

    public RuleApplierTests()
    {
        _applier = new RuleApplier(NullLogger<RuleApplier>.Instance);
    }

    private static DataFrame CreateTestDataFrame()
    {
        var df = new DataFrame();
        df.Columns.Add(new PrimitiveDataFrameColumn<double>("Feature1", new double[] { 1.0, 2.0, 3.0 }));
        df.Columns.Add(new StringDataFrameColumn("Category", new[] { "A", "B", "C" }));
        return df;
    }

    private static PreprocessingRule CreateRule(
        string id = "rule-1",
        PreprocessingRuleType type = PreprocessingRuleType.MissingValueStrategy,
        string[] columnNames = null!)
    {
        return new PreprocessingRule
        {
            Id = id,
            Type = type,
            ColumnNames = columnNames ?? new[] { "Feature1" },
            Description = $"Test rule {id}",
            PatternType = PatternType.MissingValue,
            RequiresHITL = false,
            Priority = 5,
            DiscoveredInStage = 1
        };
    }

    #region ValidateRule Tests

    [Fact]
    public void ValidateRule_ColumnsExist_ReturnsTrue()
    {
        var df = CreateTestDataFrame();
        var rule = CreateRule(columnNames: new[] { "Feature1" });

        Assert.True(_applier.ValidateRule(df, rule));
    }

    [Fact]
    public void ValidateRule_ColumnMissing_ReturnsFalse()
    {
        var df = CreateTestDataFrame();
        var rule = CreateRule(columnNames: new[] { "NonExistent" });

        Assert.False(_applier.ValidateRule(df, rule));
    }

    [Fact]
    public void ValidateRule_MultipleColumnsAllExist_ReturnsTrue()
    {
        var df = CreateTestDataFrame();
        var rule = CreateRule(columnNames: new[] { "Feature1", "Category" });

        Assert.True(_applier.ValidateRule(df, rule));
    }

    [Fact]
    public void ValidateRule_MultipleColumnsOneMissing_ReturnsFalse()
    {
        var df = CreateTestDataFrame();
        var rule = CreateRule(columnNames: new[] { "Feature1", "Missing" });

        Assert.False(_applier.ValidateRule(df, rule));
    }

    #endregion

    #region ApplyRuleAsync Tests

    [Fact]
    public async Task ApplyRuleAsync_UnimplementedStrategy_ReportsNotImplemented_NotSilentSuccess()
    {
        // Regression guard: placeholder strategies must NOT report Success=true with 0 rows
        // affected — that produced cleaned data identical to input while claiming success.
        var df = CreateTestDataFrame();
        var rule = CreateRule(); // MissingValueStrategy (not yet implemented)

        var result = await _applier.ApplyRuleAsync(df, rule);

        Assert.False(result.Success);
        Assert.Equal(RuleApplicationStatus.NotImplemented, result.Status);
        Assert.NotNull(result.ErrorMessage);
        Assert.Equal(0, result.RowsAffected);
    }

    [Fact]
    public async Task ApplyRuleAsync_InvalidColumn_ReturnsFailure()
    {
        var df = CreateTestDataFrame();
        var rule = CreateRule(columnNames: new[] { "NonExistent" });

        var result = await _applier.ApplyRuleAsync(df, rule);

        Assert.False(result.Success);
        Assert.Equal(RuleApplicationStatus.Failed, result.Status);
        Assert.NotNull(result.ErrorMessage);
        Assert.Equal(3, result.RowsSkipped);
        Assert.Equal(0, result.RowsAffected);
    }

    [Theory]
    [InlineData(PreprocessingRuleType.MissingValueStrategy)]
    [InlineData(PreprocessingRuleType.OutlierHandling)]
    [InlineData(PreprocessingRuleType.WhitespaceNormalization)]
    [InlineData(PreprocessingRuleType.DateFormatStandardization)]
    [InlineData(PreprocessingRuleType.CategoryMapping)]
    [InlineData(PreprocessingRuleType.TypeConversion)]
    [InlineData(PreprocessingRuleType.EncodingNormalization)]
    [InlineData(PreprocessingRuleType.NumericFormatStandardization)]
    [InlineData(PreprocessingRuleType.BusinessLogicDecision)]
    public async Task ApplyRuleAsync_AllRuleTypes_NotImplemented_DoNotReportSilentSuccess(PreprocessingRuleType ruleType)
    {
        // No application strategy is implemented yet; every type must surface NotImplemented
        // rather than silently succeeding. See ISSUE-mloop-20260619-ruleapplier-noop-apply.
        var df = CreateTestDataFrame();
        var rule = CreateRule(type: ruleType);

        var result = await _applier.ApplyRuleAsync(df, rule);

        Assert.False(result.Success);
        Assert.Equal(RuleApplicationStatus.NotImplemented, result.Status);
    }

    [Fact]
    public async Task ApplyRuleAsync_ReturnsCorrectRule()
    {
        var df = CreateTestDataFrame();
        var rule = CreateRule(id: "my-rule");

        var result = await _applier.ApplyRuleAsync(df, rule);

        Assert.Equal("my-rule", result.Rule.Id);
    }

    #endregion

    #region ApplyRulesAsync Tests

    [Fact]
    public async Task ApplyRulesAsync_MultipleRules_ReturnsAllResults()
    {
        var df = CreateTestDataFrame();
        var rules = new[]
        {
            CreateRule("r1"),
            CreateRule("r2"),
            CreateRule("r3")
        };

        var result = await _applier.ApplyRulesAsync(df, rules);

        Assert.Equal(3, result.TotalRules);
        // No strategy is implemented yet: all surface as NotImplemented (not silent success).
        Assert.Equal(0, result.SuccessfulRules);
        Assert.Equal(3, result.FailedRules);
        Assert.Equal(3, result.Results.Count);
        Assert.All(result.Results, r => Assert.Equal(RuleApplicationStatus.NotImplemented, r.Status));
    }

    [Fact]
    public async Task ApplyRulesAsync_ValidationFailureVsNotImplemented_DistinguishedByStatus()
    {
        var df = CreateTestDataFrame();
        var rules = new[]
        {
            CreateRule("valid-cols", columnNames: new[] { "Feature1" }),   // validates, but not implemented
            CreateRule("bad-col", columnNames: new[] { "Missing" }),        // validation failure
            CreateRule("valid-cols2", columnNames: new[] { "Category" })    // validates, but not implemented
        };

        var result = await _applier.ApplyRulesAsync(df, rules);

        Assert.Equal(3, result.TotalRules);
        Assert.Equal(0, result.SuccessfulRules);
        Assert.Equal(3, result.FailedRules);
        Assert.Equal(2, result.Results.Count(r => r.Status == RuleApplicationStatus.NotImplemented));
        Assert.Equal(1, result.Results.Count(r => r.Status == RuleApplicationStatus.Failed));
    }

    [Fact]
    public async Task ApplyRulesAsync_EmptyRules_ReturnsEmpty()
    {
        var df = CreateTestDataFrame();
        var rules = Array.Empty<PreprocessingRule>();

        var result = await _applier.ApplyRulesAsync(df, rules);

        Assert.Equal(0, result.TotalRules);
        Assert.Empty(result.Results);
    }

    [Fact]
    public async Task ApplyRulesAsync_ReportsProgress()
    {
        var df = CreateTestDataFrame();
        var rules = new[] { CreateRule("r1"), CreateRule("r2") };
        var progressReports = new List<RuleApplicationProgress>();
        var progress = new Progress<RuleApplicationProgress>(p => progressReports.Add(p));

        await _applier.ApplyRulesAsync(df, rules, progress);

        await Task.Delay(100); // Allow async progress callbacks
        Assert.NotEmpty(progressReports);
    }

    [Fact]
    public async Task ApplyRulesAsync_CancellationRequested_ThrowsOperationCanceled()
    {
        var df = CreateTestDataFrame();
        var rules = new[] { CreateRule("r1"), CreateRule("r2") };
        var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            _applier.ApplyRulesAsync(df, rules, cancellationToken: cts.Token));
    }

    [Fact]
    public async Task ApplyRulesAsync_ContinueOnFailureFalse_StopsOnFirstFailure()
    {
        var strictApplier = new RuleApplier(
            NullLogger<RuleApplier>.Instance,
            continueOnFailure: false);

        var df = CreateTestDataFrame();
        var rules = new[]
        {
            CreateRule("bad1", columnNames: new[] { "Missing1" }),
            CreateRule("good", columnNames: new[] { "Feature1" }),
        };

        var result = await strictApplier.ApplyRulesAsync(df, rules);

        // Should stop after first failure
        Assert.Single(result.Results);
        Assert.False(result.Results[0].Success);
    }

    [Fact]
    public async Task ApplyRulesAsync_RecordsTotalDuration()
    {
        var df = CreateTestDataFrame();
        var rules = new[] { CreateRule("r1") };

        var result = await _applier.ApplyRulesAsync(df, rules);

        Assert.True(result.TotalDuration >= TimeSpan.Zero);
    }

    #endregion
}
