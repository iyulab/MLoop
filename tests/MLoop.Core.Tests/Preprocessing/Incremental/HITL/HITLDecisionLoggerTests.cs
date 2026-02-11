using Microsoft.Extensions.Logging.Abstractions;
using MLoop.Core.Preprocessing.Incremental.HITL;
using MLoop.Core.Preprocessing.Incremental.HITL.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.HITL;

public class HITLDecisionLoggerTests : IDisposable
{
    private readonly string _tempDir;
    private readonly HITLDecisionLogger _logger;

    public HITLDecisionLoggerTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"mloop-hitl-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
        _logger = new HITLDecisionLogger(_tempDir, NullLogger<HITLDecisionLogger>.Instance);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private static HITLDecisionLog CreateTestLog(
        string ruleId = "rule-1",
        string questionId = "q-1",
        string selectedOption = "A",
        string? recommendedOption = "A")
    {
        var rule = new PreprocessingRule
        {
            Id = ruleId,
            Type = PreprocessingRuleType.MissingValueStrategy,
            ColumnNames = new[] { "Col1" },
            Description = "Test rule",
            PatternType = PatternType.MissingValue,
            RequiresHITL = true,
            Priority = 5,
            DiscoveredInStage = 1
        };

        var question = new HITLQuestion
        {
            Id = questionId,
            Type = HITLQuestionType.MultipleChoice,
            Context = "Test context",
            Question = "What to do?",
            Options = new[]
            {
                new HITLOption { Key = "A", Label = "Keep", Description = "Keep as-is", Action = ActionType.KeepAsIs },
                new HITLOption { Key = "B", Label = "Delete", Description = "Delete rows", Action = ActionType.Delete }
            },
            RecommendedOption = recommendedOption,
            RelatedRule = rule
        };

        return new HITLDecisionLog
        {
            Id = Guid.NewGuid().ToString(),
            SessionId = "test-session",
            Question = question,
            Answer = new HITLAnswer
            {
                QuestionId = questionId,
                SelectedOption = selectedOption,
                TimeToDecide = TimeSpan.FromSeconds(5)
            },
            ApprovedRule = rule,
            UserId = "test-user"
        };
    }

    #region LogDecisionAsync Tests

    [Fact]
    public async Task LogDecisionAsync_CreatesJsonFile()
    {
        var log = CreateTestLog();

        await _logger.LogDecisionAsync(log);

        var logDir = Path.Combine(_tempDir, "hitl-decisions");
        var files = Directory.GetFiles(logDir, "*.json");
        Assert.Single(files);
    }

    [Fact]
    public async Task LogDecisionAsync_MultipleDecisions_CreatesMultipleFiles()
    {
        await _logger.LogDecisionAsync(CreateTestLog(questionId: "q-1"));
        await Task.Delay(1100); // Ensure different timestamp
        await _logger.LogDecisionAsync(CreateTestLog(questionId: "q-2"));

        var logDir = Path.Combine(_tempDir, "hitl-decisions");
        var files = Directory.GetFiles(logDir, "*.json");
        Assert.Equal(2, files.Length);
    }

    #endregion

    #region GetDecisionsByRuleAsync Tests

    [Fact]
    public async Task GetDecisionsByRuleAsync_ReturnsMatchingDecisions()
    {
        await _logger.LogDecisionAsync(CreateTestLog(ruleId: "rule-A", questionId: "q1"));
        await Task.Delay(1100);
        await _logger.LogDecisionAsync(CreateTestLog(ruleId: "rule-B", questionId: "q2"));

        var results = await _logger.GetDecisionsByRuleAsync("rule-A");

        Assert.Single(results);
        Assert.Equal("rule-A", results[0].Question.RelatedRule.Id);
    }

    [Fact]
    public async Task GetDecisionsByRuleAsync_NoMatches_ReturnsEmpty()
    {
        await _logger.LogDecisionAsync(CreateTestLog(ruleId: "rule-A", questionId: "q1"));

        var results = await _logger.GetDecisionsByRuleAsync("nonexistent");

        Assert.Empty(results);
    }

    #endregion

    #region GetDecisionSummaryAsync Tests

    [Fact]
    public async Task GetDecisionSummaryAsync_NoDecisions_ReturnsEmptySummary()
    {
        // Use a fresh directory with no logs
        var freshDir = Path.Combine(_tempDir, "fresh");
        Directory.CreateDirectory(freshDir);
        var freshLogger = new HITLDecisionLogger(freshDir, NullLogger<HITLDecisionLogger>.Instance);

        var summary = await freshLogger.GetDecisionSummaryAsync();

        Assert.Equal(0, summary.TotalDecisions);
    }

    [Fact]
    public async Task GetDecisionSummaryAsync_WithDecisions_ReturnsCorrectCounts()
    {
        // Log decision that follows recommendation (selected = recommended = "A")
        await _logger.LogDecisionAsync(CreateTestLog(
            questionId: "q1", selectedOption: "A", recommendedOption: "A"));
        await Task.Delay(1100);

        // Log decision that overrides recommendation (selected B, recommended A)
        await _logger.LogDecisionAsync(CreateTestLog(
            questionId: "q2", selectedOption: "B", recommendedOption: "A"));

        var summary = await _logger.GetDecisionSummaryAsync();

        Assert.Equal(2, summary.TotalDecisions);
        Assert.Equal(1, summary.RecommendationsFollowed);
        Assert.Equal(1, summary.RecommendationsOverridden);
    }

    #endregion

    #region Directory Management Tests

    [Fact]
    public void Constructor_CreatesLogDirectory()
    {
        var newDir = Path.Combine(_tempDir, "new-base");
        _ = new HITLDecisionLogger(newDir, NullLogger<HITLDecisionLogger>.Instance);

        Assert.True(Directory.Exists(Path.Combine(newDir, "hitl-decisions")));
    }

    #endregion
}
