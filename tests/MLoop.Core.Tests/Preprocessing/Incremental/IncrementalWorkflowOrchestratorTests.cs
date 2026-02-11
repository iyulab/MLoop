using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.Core.Preprocessing.Incremental;
using MLoop.Core.Preprocessing.Incremental.Contracts;
using MLoop.Core.Preprocessing.Incremental.Deliverables.Contracts;
using MLoop.Core.Preprocessing.Incremental.HITL;
using MLoop.Core.Preprocessing.Incremental.HITL.Contracts;
using MLoop.Core.Preprocessing.Incremental.HITL.Models;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleApplication.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleApplication.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental;

public class IncrementalWorkflowOrchestratorTests : IDisposable
{
    private readonly string _tempDir;
    private readonly string _csvPath;

    public IncrementalWorkflowOrchestratorTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"mloop-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);

        _csvPath = Path.Combine(_tempDir, "test.csv");
        File.WriteAllText(_csvPath, "Feature1,Feature2,Label\n1.0,2.0,A\n3.0,4.0,B\n5.0,6.0,A\n7.0,8.0,B\n9.0,10.0,A\n");
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            Directory.Delete(_tempDir, true);
        }
    }

    #region Test Helpers

    private static PreprocessingRule CreateTestRule(string id = "rule-1", bool requiresHITL = true)
    {
        return new PreprocessingRule
        {
            Id = id,
            Type = PreprocessingRuleType.MissingValueStrategy,
            ColumnNames = new[] { "Feature1" },
            Description = "Handle missing values in Feature1",
            PatternType = PatternType.MissingValue,
            RequiresHITL = requiresHITL,
            Priority = 5,
            DiscoveredInStage = 1
        };
    }

    private static SampleAnalysis CreateTestAnalysis(int stageNumber = 1, double qualityScore = 0.9)
    {
        return new SampleAnalysis
        {
            StageNumber = stageNumber,
            SampleRatio = 0.001,
            Timestamp = DateTime.UtcNow,
            RowCount = 5,
            ColumnCount = 3,
            Columns = new List<ColumnAnalysis>(),
            QualityScore = qualityScore
        };
    }

    private static IncrementalWorkflowConfig CreateDefaultConfig(string? outputDir = null, string? checkpointDir = null)
    {
        return new IncrementalWorkflowConfig
        {
            SkipHITL = true,
            EnableCheckpoints = false,
            OutputDirectory = outputDir ?? "./cleaned",
            CheckpointDirectory = checkpointDir ?? "./checkpoints"
        };
    }

    private IncrementalWorkflowOrchestrator CreateOrchestrator(
        ISamplingEngine? samplingEngine = null,
        ISampleAnalyzer? sampleAnalyzer = null,
        IRuleDiscoveryEngine? ruleDiscoveryEngine = null,
        HITLWorkflowService? hitlService = null,
        IRuleApplier? ruleApplier = null,
        IDeliverableGenerator? deliverableGenerator = null)
    {
        var se = samplingEngine ?? new FakeSamplingEngine();
        var sa = sampleAnalyzer ?? new FakeSampleAnalyzer();
        var rde = ruleDiscoveryEngine ?? new FakeRuleDiscoveryEngine();
        var hitl = hitlService ?? CreateFakeHITLService();

        return new IncrementalWorkflowOrchestrator(
            se, sa, rde, hitl,
            NullLogger<IncrementalWorkflowOrchestrator>.Instance,
            ruleApplier, deliverableGenerator);
    }

    private static HITLWorkflowService CreateFakeHITLService()
    {
        return new HITLWorkflowService(
            new FakeQuestionGenerator(),
            new FakePromptBuilder(),
            new FakeDecisionLogger(),
            NullLogger<HITLWorkflowService>.Instance);
    }

    #endregion

    #region ExecuteWorkflowAsync Tests

    [Fact]
    public async Task ExecuteWorkflowAsync_SkipHITL_CompletesAllStages()
    {
        var orchestrator = CreateOrchestrator();
        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        Assert.Equal(WorkflowStage.Completed, state.CurrentStage);
        Assert.NotNull(state.CompletedAt);
        Assert.Equal(5, state.CompletedStages.Count);
        Assert.True(state.CompletedStages.ContainsKey(WorkflowStage.InitialExploration));
        Assert.True(state.CompletedStages.ContainsKey(WorkflowStage.PatternExpansion));
        Assert.True(state.CompletedStages.ContainsKey(WorkflowStage.HITLDecision));
        Assert.True(state.CompletedStages.ContainsKey(WorkflowStage.ConfidenceCheckpoint));
        Assert.True(state.CompletedStages.ContainsKey(WorkflowStage.BulkProcessing));
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_RecordsCorrectDatasetPath()
    {
        var orchestrator = CreateOrchestrator();
        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        Assert.Equal(_csvPath, state.DatasetPath);
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_CountsRecordsFromCsv()
    {
        var orchestrator = CreateOrchestrator();
        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        // 5 data rows (header excluded)
        Assert.Equal(5, state.TotalRecords);
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_GeneratesSessionId()
    {
        var orchestrator = CreateOrchestrator();
        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        Assert.False(string.IsNullOrEmpty(state.SessionId));
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_DiscoversTwoRules_FromTwoStages()
    {
        // Stage 1 discovers rule-1, Stage 2 discovers rule-2
        var rde = new FakeRuleDiscoveryEngine(
            stage1Rules: new[] { CreateTestRule("rule-1") },
            stage2Rules: new[] { CreateTestRule("rule-1"), CreateTestRule("rule-2") });

        var orchestrator = CreateOrchestrator(ruleDiscoveryEngine: rde);
        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        Assert.Equal(2, state.DiscoveredRules.Count);
        Assert.Contains(state.DiscoveredRules, r => r.Id == "rule-1");
        Assert.Contains(state.DiscoveredRules, r => r.Id == "rule-2");
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_SkipHITL_AutoApprovesAllRules()
    {
        var rde = new FakeRuleDiscoveryEngine(
            stage1Rules: new[] { CreateTestRule("rule-1"), CreateTestRule("rule-2") });

        var orchestrator = CreateOrchestrator(ruleDiscoveryEngine: rde);
        var config = new IncrementalWorkflowConfig { SkipHITL = true, EnableCheckpoints = false };

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        Assert.Equal(2, state.ApprovedRules.Count);
        Assert.All(state.ApprovedRules, r => Assert.True(r.IsApproved));
        Assert.All(state.ApprovedRules, r => Assert.Contains("Auto-approved", r.UserFeedback!));
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_NoNewRulesInStage2_SetsConverged()
    {
        var rules = new[] { CreateTestRule("rule-1") };
        var rde = new FakeRuleDiscoveryEngine(
            stage1Rules: rules,
            stage2Rules: rules); // Same rules → no new rules

        var orchestrator = CreateOrchestrator(ruleDiscoveryEngine: rde);
        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        Assert.True(state.HasConverged);
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_ReportsProgress()
    {
        var orchestrator = CreateOrchestrator();
        var config = CreateDefaultConfig();
        var progressReports = new List<WorkflowProgress>();
        var progress = new Progress<WorkflowProgress>(p => progressReports.Add(p));

        await orchestrator.ExecuteWorkflowAsync(_csvPath, config, progress);

        // Allow async progress callbacks to complete
        await Task.Delay(100);

        Assert.NotEmpty(progressReports);
        Assert.Contains(progressReports, p => p.Stage == WorkflowStage.InitialExploration);
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_WithDeliverables_GeneratesOutput()
    {
        var rules = new[] { CreateTestRule("rule-1") };
        var rde = new FakeRuleDiscoveryEngine(stage1Rules: rules);
        var ruleApplier = new FakeRuleApplier();
        var deliverableGen = new FakeDeliverableGenerator();

        var orchestrator = CreateOrchestrator(
            ruleDiscoveryEngine: rde,
            ruleApplier: ruleApplier,
            deliverableGenerator: deliverableGen);

        var config = new IncrementalWorkflowConfig
        {
            SkipHITL = true,
            EnableCheckpoints = false,
            OutputDirectory = _tempDir
        };

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        Assert.True(ruleApplier.ApplyRulesCalled);
        Assert.True(deliverableGen.GenerateAllCalled);
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_NoApprovedRules_SkipsDeliverables()
    {
        // No rules discovered → no approved rules → skip deliverables
        var rde = new FakeRuleDiscoveryEngine();
        var ruleApplier = new FakeRuleApplier();
        var deliverableGen = new FakeDeliverableGenerator();

        var orchestrator = CreateOrchestrator(
            ruleDiscoveryEngine: rde,
            ruleApplier: ruleApplier,
            deliverableGenerator: deliverableGen);

        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        Assert.False(ruleApplier.ApplyRulesCalled);
        Assert.False(deliverableGen.GenerateAllCalled);
        Assert.Empty(state.ApprovedRules);
    }

    #endregion

    #region Checkpoint Tests

    [Fact]
    public async Task SaveCheckpointAsync_CreatesFile()
    {
        var orchestrator = CreateOrchestrator();
        var checkpointPath = Path.Combine(_tempDir, "checkpoint.json");

        var state = new IncrementalWorkflowState
        {
            SessionId = "test-session",
            CurrentStage = WorkflowStage.PatternExpansion,
            DatasetPath = _csvPath,
            TotalRecords = 100,
            CompletedStages = new Dictionary<WorkflowStage, StageResult>(),
            DiscoveredRules = new List<PreprocessingRule>(),
            ApprovedRules = new List<PreprocessingRule>(),
            ConfidenceScore = 0.5,
            Config = CreateDefaultConfig()
        };

        await orchestrator.SaveCheckpointAsync(state, checkpointPath);

        Assert.True(File.Exists(checkpointPath));
        var content = await File.ReadAllTextAsync(checkpointPath);
        Assert.Contains("test-session", content);
    }

    [Fact]
    public async Task LoadCheckpointAsync_ReturnsState()
    {
        var orchestrator = CreateOrchestrator();
        var checkpointPath = Path.Combine(_tempDir, "checkpoint.json");

        var originalState = new IncrementalWorkflowState
        {
            SessionId = "load-test",
            CurrentStage = WorkflowStage.HITLDecision,
            DatasetPath = _csvPath,
            TotalRecords = 200,
            CompletedStages = new Dictionary<WorkflowStage, StageResult>(),
            DiscoveredRules = new List<PreprocessingRule>(),
            ApprovedRules = new List<PreprocessingRule>(),
            ConfidenceScore = 0.75,
            Config = CreateDefaultConfig()
        };

        await orchestrator.SaveCheckpointAsync(originalState, checkpointPath);
        var loaded = await orchestrator.LoadCheckpointAsync(checkpointPath);

        Assert.Equal("load-test", loaded.SessionId);
        Assert.Equal(WorkflowStage.HITLDecision, loaded.CurrentStage);
        Assert.Equal(200, loaded.TotalRecords);
    }

    [Fact]
    public async Task LoadCheckpointAsync_ThrowsForMissingFile()
    {
        var orchestrator = CreateOrchestrator();
        var missingPath = Path.Combine(_tempDir, "nonexistent.json");

        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            orchestrator.LoadCheckpointAsync(missingPath));
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_WithCheckpoints_SavesAfterEachStage()
    {
        var checkpointDir = Path.Combine(_tempDir, "checkpoints");
        Directory.CreateDirectory(checkpointDir);

        var orchestrator = CreateOrchestrator();
        var config = new IncrementalWorkflowConfig
        {
            SkipHITL = true,
            EnableCheckpoints = true,
            CheckpointDirectory = checkpointDir
        };

        await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        var checkpointFiles = Directory.GetFiles(checkpointDir, "checkpoint-*.json");
        // 4 checkpoints (after stages 1-4, not after stage 5)
        Assert.Equal(4, checkpointFiles.Length);
    }

    #endregion

    #region ConfidenceScore Tests

    [Fact]
    public async Task ExecuteWorkflowAsync_CalculatesConfidenceScore()
    {
        var orchestrator = CreateOrchestrator();
        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        Assert.InRange(state.ConfidenceScore, 0.0, 1.0);
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_ConvergedRules_HigherConfidence()
    {
        var rules = new[] { CreateTestRule("rule-1") };
        var rde = new FakeRuleDiscoveryEngine(
            stage1Rules: rules,
            stage2Rules: rules); // Same rules → converged

        var orchestrator = CreateOrchestrator(ruleDiscoveryEngine: rde);
        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        // Converged → higher convergence factor (1.0 vs 0.8)
        Assert.True(state.HasConverged);
        Assert.True(state.ConfidenceScore > 0.3);
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public async Task ExecuteWorkflowAsync_InvalidPath_ThrowsInvalidOperation()
    {
        var orchestrator = CreateOrchestrator();
        var config = CreateDefaultConfig();

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            orchestrator.ExecuteWorkflowAsync(
                Path.Combine(_tempDir, "nonexistent.csv"), config));
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_EmptyCsv_HandlesGracefully()
    {
        var emptyPath = Path.Combine(_tempDir, "empty.csv");
        File.WriteAllText(emptyPath, "Col1,Col2\n");

        var orchestrator = CreateOrchestrator();
        var config = CreateDefaultConfig();

        // Empty CSV (0 data rows) should not crash
        var state = await orchestrator.ExecuteWorkflowAsync(emptyPath, config);

        Assert.Equal(WorkflowStage.Completed, state.CurrentStage);
        Assert.Equal(0, state.TotalRecords);
    }

    #endregion

    #region Stage Result Tests

    [Fact]
    public async Task ExecuteWorkflowAsync_Stage1_RecordsSampleSize()
    {
        var orchestrator = CreateOrchestrator();
        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        var stage1 = state.CompletedStages[WorkflowStage.InitialExploration];
        Assert.True(stage1.SampleSize > 0);
        Assert.Equal(config.Stage1Ratio, stage1.SampleRatio);
        Assert.NotNull(stage1.Analysis);
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_Stage5_RecordsBulkProcessing()
    {
        var orchestrator = CreateOrchestrator();
        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        var stage5 = state.CompletedStages[WorkflowStage.BulkProcessing];
        Assert.Equal(1.0, stage5.SampleRatio);
        Assert.Equal(WorkflowStage.BulkProcessing, stage5.Stage);
    }

    [Fact]
    public async Task ExecuteWorkflowAsync_EachStageHasDuration()
    {
        var orchestrator = CreateOrchestrator();
        var config = CreateDefaultConfig();

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        foreach (var stage in state.CompletedStages.Values)
        {
            Assert.True(stage.Duration >= TimeSpan.Zero);
        }
    }

    #endregion

    #region AutoApproval Tests

    [Fact]
    public async Task ExecuteWorkflowAsync_EnableAutoApproval_HighConfidence_ApprovesRemainingRules()
    {
        var rules = new[] { CreateTestRule("rule-1"), CreateTestRule("rule-2") };
        var rde = new FakeRuleDiscoveryEngine(
            stage1Rules: rules,
            stage2Rules: rules); // Converged → high confidence

        var highQualityAnalyzer = new FakeSampleAnalyzer(qualityScore: 1.0);

        var orchestrator = CreateOrchestrator(
            sampleAnalyzer: highQualityAnalyzer,
            ruleDiscoveryEngine: rde);

        var config = new IncrementalWorkflowConfig
        {
            SkipHITL = false, // Don't skip HITL, but use auto-approval
            EnableAutoApproval = true,
            MinConfidenceThreshold = 0.5,
            EnableCheckpoints = false
        };

        var state = await orchestrator.ExecuteWorkflowAsync(_csvPath, config);

        // All rules should be approved (either via HITL or auto-approval)
        Assert.True(state.ApprovedRules.Count > 0);
    }

    #endregion

    #region Fake Implementations

    private sealed class FakeSamplingEngine : ISamplingEngine
    {
        public ISamplingStrategy Strategy => throw new NotImplementedException();

        public Task<DataFrame> SampleAsync(
            DataFrame data,
            double sampleRatio,
            SamplingConfiguration? config = null,
            IProgress<double>? progress = null,
            CancellationToken cancellationToken = default)
        {
            // Return the data as-is (it's already small enough for testing)
            return Task.FromResult(data);
        }

        public bool ValidateSample(DataFrame source, DataFrame sample, double tolerance = 0.02)
            => true;
    }

    private sealed class FakeSampleAnalyzer : ISampleAnalyzer
    {
        private readonly double _qualityScore;

        public FakeSampleAnalyzer(double qualityScore = 0.9) => _qualityScore = qualityScore;

        public Task<SampleAnalysis> AnalyzeAsync(
            DataFrame sample,
            int stageNumber,
            AnalysisConfiguration? config = null,
            CancellationToken cancellationToken = default)
        {
            return Task.FromResult(CreateTestAnalysis(stageNumber, _qualityScore));
        }

        public ColumnAnalysis AnalyzeColumn(DataFrameColumn column, string columnName, int columnIndex)
            => throw new NotImplementedException();

        public bool HasConverged(SampleAnalysis previous, SampleAnalysis current, double threshold = 0.01)
            => false;
    }

    private sealed class FakeRuleDiscoveryEngine : IRuleDiscoveryEngine
    {
        private readonly IReadOnlyList<PreprocessingRule> _stage1Rules;
        private readonly IReadOnlyList<PreprocessingRule> _stage2Rules;
        private int _callCount;

        public FakeRuleDiscoveryEngine(
            IReadOnlyList<PreprocessingRule>? stage1Rules = null,
            IReadOnlyList<PreprocessingRule>? stage2Rules = null)
        {
            _stage1Rules = stage1Rules ?? Array.Empty<PreprocessingRule>();
            _stage2Rules = stage2Rules ?? _stage1Rules;
        }

        public Task<IReadOnlyList<PreprocessingRule>> DiscoverRulesAsync(
            DataFrame sample,
            SampleAnalysis analysis,
            CancellationToken cancellationToken = default)
        {
            _callCount++;
            var rules = _callCount == 1 ? _stage1Rules : _stage2Rules;
            return Task.FromResult(rules);
        }

        public Task<ConfidenceScore> CalculateConfidenceAsync(
            PreprocessingRule rule,
            DataFrame previousSample,
            DataFrame currentSample,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        public bool HasConverged(
            IReadOnlyList<PreprocessingRule> previousRules,
            IReadOnlyList<PreprocessingRule> currentRules,
            double threshold = 0.02)
            => false;
    }

    private sealed class FakeQuestionGenerator : IHITLQuestionGenerator
    {
        public HITLQuestion GenerateQuestion(
            PreprocessingRule rule,
            DataFrame sample,
            SampleAnalysis analysis)
        {
            return new HITLQuestion
            {
                Id = $"q-{rule.Id}",
                Type = HITLQuestionType.MultipleChoice,
                Context = "Test context",
                Question = "Test question?",
                Options = new[]
                {
                    new HITLOption { Key = "A", Label = "Option A", Description = "Desc A", Action = ActionType.KeepAsIs }
                },
                RecommendedOption = "A",
                RelatedRule = rule
            };
        }

        public IReadOnlyList<HITLQuestion> GenerateAllQuestions(
            IReadOnlyList<PreprocessingRule> rules,
            DataFrame sample,
            SampleAnalysis analysis)
        {
            return rules
                .Where(r => r.RequiresHITL)
                .Select(r => GenerateQuestion(r, sample, analysis))
                .ToList();
        }
    }

    private sealed class FakePromptBuilder : IHITLPromptBuilder
    {
        public string BuildPrompt(HITLQuestion question) => question.Id;

        public HITLAnswer CollectAnswer(HITLQuestion question) => new HITLAnswer
        {
            QuestionId = question.Id,
            SelectedOption = question.RecommendedOption ?? "A"
        };

        public void DisplayConfirmation(HITLAnswer answer) { }
    }

    private sealed class FakeDecisionLogger : IHITLDecisionLogger
    {
        private readonly List<HITLDecisionLog> _logs = new();

        public Task LogDecisionAsync(HITLDecisionLog log)
        {
            _logs.Add(log);
            return Task.CompletedTask;
        }

        public Task<IReadOnlyList<HITLDecisionLog>> GetDecisionsByRuleAsync(string ruleId)
            => Task.FromResult<IReadOnlyList<HITLDecisionLog>>(
                _logs.Where(l => l.ApprovedRule.Id == ruleId).ToList());

        public Task<IReadOnlyList<HITLDecisionLog>> GetDecisionsByTimeRangeAsync(DateTime startTime, DateTime endTime)
            => Task.FromResult<IReadOnlyList<HITLDecisionLog>>(
                _logs.Where(l => l.LoggedAt >= startTime && l.LoggedAt <= endTime).ToList());

        public Task<HITLDecisionSummary> GetDecisionSummaryAsync()
            => Task.FromResult(new HITLDecisionSummary { TotalDecisions = _logs.Count });
    }

    private sealed class FakeRuleApplier : IRuleApplier
    {
        public bool ApplyRulesCalled { get; private set; }

        public Task<RuleApplicationResult> ApplyRuleAsync(
            DataFrame data,
            PreprocessingRule rule,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        public Task<BulkApplicationResult> ApplyRulesAsync(
            DataFrame data,
            IReadOnlyList<PreprocessingRule> rules,
            IProgress<RuleApplicationProgress>? progress = null,
            CancellationToken cancellationToken = default)
        {
            ApplyRulesCalled = true;
            return Task.FromResult(new BulkApplicationResult
            {
                TotalRules = rules.Count,
                SuccessfulRules = rules.Count,
                FailedRules = 0,
                Results = rules.Select(r => new RuleApplicationResult
                {
                    Rule = r,
                    RowsAffected = 1,
                    RowsSkipped = 0,
                    Duration = TimeSpan.FromMilliseconds(10),
                    Success = true
                }).ToList(),
                TotalDuration = TimeSpan.FromMilliseconds(50)
            });
        }

        public bool ValidateRule(DataFrame data, PreprocessingRule rule) => true;
    }

    private sealed class FakeDeliverableGenerator : IDeliverableGenerator
    {
        public bool GenerateAllCalled { get; private set; }

        public Task<DeliverableManifest> GenerateAllAsync(
            IncrementalWorkflowState state,
            DataFrame cleanedData,
            string outputDirectory,
            CancellationToken cancellationToken = default)
        {
            GenerateAllCalled = true;
            return Task.FromResult(new DeliverableManifest
            {
                CleanedDataPath = Path.Combine(outputDirectory, "cleaned.csv")
            });
        }

        public Task SaveCleanedDataAsync(DataFrame data, string outputPath, CancellationToken cancellationToken = default)
            => Task.CompletedTask;

        public Task GenerateReportAsync(IncrementalWorkflowState state, string outputPath, CancellationToken cancellationToken = default)
            => Task.CompletedTask;

        public Task GenerateMetadataAsync(IncrementalWorkflowState state, string outputPath, CancellationToken cancellationToken = default)
            => Task.CompletedTask;
    }

    #endregion
}
