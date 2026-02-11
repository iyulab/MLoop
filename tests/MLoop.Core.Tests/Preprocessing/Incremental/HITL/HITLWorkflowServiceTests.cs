using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.Core.Preprocessing.Incremental.HITL;
using MLoop.Core.Preprocessing.Incremental.HITL.Contracts;
using MLoop.Core.Preprocessing.Incremental.HITL.Models;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.HITL;

public class HITLWorkflowServiceTests
{
    private readonly RecordingDecisionLogger _decisionLogger;
    private readonly HITLWorkflowService _service;

    public HITLWorkflowServiceTests()
    {
        _decisionLogger = new RecordingDecisionLogger();
        _service = new HITLWorkflowService(
            new FakeQuestionGenerator(),
            new AutoAcceptPromptBuilder(),
            _decisionLogger,
            NullLogger<HITLWorkflowService>.Instance);
    }

    private static PreprocessingRule CreateRule(string id, bool requiresHITL = true)
    {
        return new PreprocessingRule
        {
            Id = id,
            Type = PreprocessingRuleType.MissingValueStrategy,
            ColumnNames = new[] { "Col1" },
            Description = $"Rule {id}",
            PatternType = PatternType.MissingValue,
            RequiresHITL = requiresHITL,
            Priority = 5,
            DiscoveredInStage = 1
        };
    }

    private static DataFrame CreateSampleData()
    {
        var df = new DataFrame();
        df.Columns.Add(new PrimitiveDataFrameColumn<double>("Col1", new[] { 1.0, 2.0 }));
        return df;
    }

    private static SampleAnalysis CreateAnalysis()
    {
        return new SampleAnalysis
        {
            StageNumber = 1,
            SampleRatio = 0.1,
            Timestamp = DateTime.UtcNow,
            RowCount = 2,
            ColumnCount = 1,
            Columns = new List<ColumnAnalysis>(),
            QualityScore = 0.9
        };
    }

    #region ExecuteWorkflowAsync Tests

    [Fact]
    public async Task ExecuteWorkflowAsync_MultipleRules_AllShareSameSessionId()
    {
        var rules = new[] { CreateRule("r1"), CreateRule("r2"), CreateRule("r3") };

        await _service.ExecuteWorkflowAsync(rules, CreateSampleData(), CreateAnalysis());

        Assert.Equal(3, _decisionLogger.Logs.Count);

        var sessionIds = _decisionLogger.Logs.Select(l => l.SessionId).Distinct().ToList();
        Assert.Single(sessionIds); // All decisions share one session ID
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_NoRulesRequiringHITL_ReturnsEmpty()
    {
        var rules = new[] { CreateRule("r1", requiresHITL: false) };

        var decisions = await _service.ExecuteWorkflowAsync(rules, CreateSampleData(), CreateAnalysis());

        // FakeQuestionGenerator returns empty for non-HITL rules
        Assert.Empty(decisions);
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_ApprovedRules_MarkedAsApproved()
    {
        var rules = new[] { CreateRule("r1") };

        var decisions = await _service.ExecuteWorkflowAsync(rules, CreateSampleData(), CreateAnalysis());

        Assert.Single(decisions);
        Assert.True(decisions[0].ApprovedRule.IsApproved);
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_EachDecision_HasUniqueId()
    {
        var rules = new[] { CreateRule("r1"), CreateRule("r2") };

        await _service.ExecuteWorkflowAsync(rules, CreateSampleData(), CreateAnalysis());

        var ids = _decisionLogger.Logs.Select(l => l.Id).ToList();
        Assert.Equal(ids.Count, ids.Distinct().Count());
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_RecordsUserName()
    {
        var rules = new[] { CreateRule("r1") };

        var decisions = await _service.ExecuteWorkflowAsync(rules, CreateSampleData(), CreateAnalysis());

        Assert.All(decisions, d => Assert.False(string.IsNullOrEmpty(d.UserId)));
    }

    #endregion

    #region ExecuteSingleRuleWorkflowAsync Tests

    [Fact]
    public async Task ExecuteSingleRuleWorkflowAsync_ReturnsDecision()
    {
        var rule = CreateRule("r1");

        var decision = await _service.ExecuteSingleRuleWorkflowAsync(
            rule, CreateSampleData(), CreateAnalysis());

        Assert.NotNull(decision);
        Assert.True(decision.ApprovedRule.IsApproved);
    }

    [Fact]
    public async Task ExecuteSingleRuleWorkflowAsync_LogsDecision()
    {
        var rule = CreateRule("r1");

        await _service.ExecuteSingleRuleWorkflowAsync(
            rule, CreateSampleData(), CreateAnalysis());

        Assert.Single(_decisionLogger.Logs);
    }

    [Fact]
    public async Task ExecuteSingleRuleWorkflowAsync_HasOwnSessionId()
    {
        var rule = CreateRule("r1");

        var decision = await _service.ExecuteSingleRuleWorkflowAsync(
            rule, CreateSampleData(), CreateAnalysis());

        Assert.False(string.IsNullOrEmpty(decision.SessionId));
    }

    [Fact]
    public async Task ExecuteSingleRuleWorkflowAsync_DifferentFromWorkflowSession()
    {
        var rules = new[] { CreateRule("r1") };

        // Run workflow with single rule
        await _service.ExecuteWorkflowAsync(rules, CreateSampleData(), CreateAnalysis());
        var workflowSessionId = _decisionLogger.Logs[0].SessionId;

        // Run single rule
        var decision = await _service.ExecuteSingleRuleWorkflowAsync(
            CreateRule("r2"), CreateSampleData(), CreateAnalysis());

        // Single rule workflow should have its own session ID
        Assert.NotEqual(workflowSessionId, decision.SessionId);
    }

    #endregion

    #region Fake Implementations

    private sealed class FakeQuestionGenerator : IHITLQuestionGenerator
    {
        public HITLQuestion GenerateQuestion(PreprocessingRule rule, DataFrame sample, SampleAnalysis analysis)
        {
            return new HITLQuestion
            {
                Id = $"q-{rule.Id}",
                Type = HITLQuestionType.MultipleChoice,
                Context = "Test",
                Question = "What?",
                Options = new[]
                {
                    new HITLOption { Key = "A", Label = "Keep", Description = "Keep", Action = ActionType.KeepAsIs, IsRecommended = true }
                },
                RecommendedOption = "A",
                RelatedRule = rule
            };
        }

        public IReadOnlyList<HITLQuestion> GenerateAllQuestions(
            IReadOnlyList<PreprocessingRule> rules, DataFrame sample, SampleAnalysis analysis)
        {
            return rules.Where(r => r.RequiresHITL)
                .Select(r => GenerateQuestion(r, sample, analysis))
                .ToList();
        }
    }

    private sealed class AutoAcceptPromptBuilder : IHITLPromptBuilder
    {
        public string BuildPrompt(HITLQuestion question) => question.Id;

        public HITLAnswer CollectAnswer(HITLQuestion question) => new HITLAnswer
        {
            QuestionId = question.Id,
            SelectedOption = question.RecommendedOption ?? "A"
        };

        public void DisplayConfirmation(HITLAnswer answer) { }
    }

    private sealed class RecordingDecisionLogger : IHITLDecisionLogger
    {
        public List<HITLDecisionLog> Logs { get; } = new();

        public Task LogDecisionAsync(HITLDecisionLog log)
        {
            Logs.Add(log);
            return Task.CompletedTask;
        }

        public Task<IReadOnlyList<HITLDecisionLog>> GetDecisionsByRuleAsync(string ruleId)
            => Task.FromResult<IReadOnlyList<HITLDecisionLog>>(
                Logs.Where(l => l.ApprovedRule.Id == ruleId).ToList());

        public Task<IReadOnlyList<HITLDecisionLog>> GetDecisionsByTimeRangeAsync(DateTime startTime, DateTime endTime)
            => Task.FromResult<IReadOnlyList<HITLDecisionLog>>(new List<HITLDecisionLog>());

        public Task<HITLDecisionSummary> GetDecisionSummaryAsync()
            => Task.FromResult(new HITLDecisionSummary { TotalDecisions = Logs.Count });
    }

    #endregion
}
