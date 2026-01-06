using System.Text.Json;
using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging;
using MLoop.Core.Preprocessing.Incremental.Contracts;
using MLoop.Core.Preprocessing.Incremental.HITL;
using MLoop.Core.Preprocessing.Incremental.HITL.Models;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental;

/// <summary>
/// Orchestrates the complete 5-stage incremental preprocessing workflow.
/// Coordinates sampling, rule discovery, HITL, and bulk processing.
/// </summary>
public sealed class IncrementalWorkflowOrchestrator : IWorkflowOrchestrator
{
    private readonly ISamplingEngine _samplingEngine;
    private readonly ISampleAnalyzer _sampleAnalyzer;
    private readonly IRuleDiscoveryEngine _ruleDiscoveryEngine;
    private readonly HITLWorkflowService _hitlWorkflowService;
    private readonly ILogger<IncrementalWorkflowOrchestrator> _logger;

    public IncrementalWorkflowOrchestrator(
        ISamplingEngine samplingEngine,
        ISampleAnalyzer sampleAnalyzer,
        IRuleDiscoveryEngine ruleDiscoveryEngine,
        HITLWorkflowService hitlWorkflowService,
        ILogger<IncrementalWorkflowOrchestrator> logger)
    {
        _samplingEngine = samplingEngine ?? throw new ArgumentNullException(nameof(samplingEngine));
        _sampleAnalyzer = sampleAnalyzer ?? throw new ArgumentNullException(nameof(sampleAnalyzer));
        _ruleDiscoveryEngine = ruleDiscoveryEngine ?? throw new ArgumentNullException(nameof(ruleDiscoveryEngine));
        _hitlWorkflowService = hitlWorkflowService ?? throw new ArgumentNullException(nameof(hitlWorkflowService));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <inheritdoc />
    public async Task<IncrementalWorkflowState> ExecuteWorkflowAsync(
        string datasetPath,
        IncrementalWorkflowConfig config,
        IProgress<WorkflowProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Starting incremental preprocessing workflow for dataset: {DatasetPath}", datasetPath);

        // Initialize workflow state
        var state = CreateInitialState(datasetPath, config);

        // Execute stages sequentially
        await ExecuteStage1Async(state, progress, cancellationToken);
        await SaveCheckpointIfEnabledAsync(state, config);

        await ExecuteStage2Async(state, progress, cancellationToken);
        await SaveCheckpointIfEnabledAsync(state, config);

        await ExecuteStage3Async(state, progress, cancellationToken);
        await SaveCheckpointIfEnabledAsync(state, config);

        await ExecuteStage4Async(state, progress, cancellationToken);
        await SaveCheckpointIfEnabledAsync(state, config);

        await ExecuteStage5Async(state, progress, cancellationToken);

        // Mark workflow as completed
        state.CompletedAt = DateTime.UtcNow;
        state.CurrentStage = WorkflowStage.Completed;

        _logger.LogInformation("Workflow completed successfully in {Duration}", state.TotalDuration);

        return state;
    }

    /// <inheritdoc />
    public async Task<IncrementalWorkflowState> ResumeWorkflowAsync(
        string checkpointPath,
        IProgress<WorkflowProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Resuming workflow from checkpoint: {CheckpointPath}", checkpointPath);

        var state = await LoadCheckpointAsync(checkpointPath, cancellationToken);

        // Resume from current stage
        switch (state.CurrentStage)
        {
            case WorkflowStage.InitialExploration:
                await ExecuteStage1Async(state, progress, cancellationToken);
                goto case WorkflowStage.PatternExpansion;

            case WorkflowStage.PatternExpansion:
                await ExecuteStage2Async(state, progress, cancellationToken);
                goto case WorkflowStage.HITLDecision;

            case WorkflowStage.HITLDecision:
                await ExecuteStage3Async(state, progress, cancellationToken);
                goto case WorkflowStage.ConfidenceCheckpoint;

            case WorkflowStage.ConfidenceCheckpoint:
                await ExecuteStage4Async(state, progress, cancellationToken);
                goto case WorkflowStage.BulkProcessing;

            case WorkflowStage.BulkProcessing:
                await ExecuteStage5Async(state, progress, cancellationToken);
                break;

            case WorkflowStage.Completed:
                _logger.LogInformation("Workflow already completed");
                break;
        }

        return state;
    }

    /// <inheritdoc />
    public async Task SaveCheckpointAsync(
        IncrementalWorkflowState state,
        string? checkpointPath = null,
        CancellationToken cancellationToken = default)
    {
        var path = checkpointPath ?? GetDefaultCheckpointPath(state);

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var json = JsonSerializer.Serialize(state, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        await File.WriteAllTextAsync(path, json, cancellationToken);

        _logger.LogInformation("Checkpoint saved to: {Path}", path);
    }

    /// <inheritdoc />
    public async Task<IncrementalWorkflowState> LoadCheckpointAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(checkpointPath))
        {
            throw new FileNotFoundException($"Checkpoint file not found: {checkpointPath}");
        }

        var json = await File.ReadAllTextAsync(checkpointPath, cancellationToken);
        var state = JsonSerializer.Deserialize<IncrementalWorkflowState>(json);

        if (state == null)
        {
            throw new InvalidOperationException($"Failed to deserialize checkpoint: {checkpointPath}");
        }

        _logger.LogInformation("Checkpoint loaded from: {Path}", checkpointPath);

        return state;
    }

    // ===== Stage Execution Methods =====

    private async Task ExecuteStage1Async(
        IncrementalWorkflowState state,
        IProgress<WorkflowProgress>? progress,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("Executing Stage 1: Initial Exploration (0.1%)");
        ReportProgress(progress, WorkflowStage.InitialExploration, 0.0, "Starting initial exploration", state);

        var startTime = DateTime.UtcNow;
        var sampleRatio = state.Config.Stage1Ratio;

        // Load full dataset
        var fullData = LoadDataFrame(state.DatasetPath);

        // Sample data
        var sample = await _samplingEngine.SampleAsync(fullData, sampleRatio, null, null, cancellationToken);

        // Analyze sample
        var analysis = await _sampleAnalyzer.AnalyzeAsync(sample, 1, null, cancellationToken);

        // Discover rules
        var rules = await _ruleDiscoveryEngine.DiscoverRulesAsync(sample, analysis, cancellationToken);

        // Update state
        var stageResult = new StageResult
        {
            Stage = WorkflowStage.InitialExploration,
            SampleSize = (int)sample.Rows.Count,
            SampleRatio = sampleRatio,
            Analysis = analysis,
            RulesDiscovered = rules.ToList(),
            Duration = DateTime.UtcNow - startTime
        };

        state.CompletedStages[WorkflowStage.InitialExploration] = stageResult;
        state.DiscoveredRules.AddRange(rules);
        state.CurrentStage = WorkflowStage.PatternExpansion;

        ReportProgress(progress, WorkflowStage.InitialExploration, 1.0,
            $"Stage 1 complete: {rules.Count} rules discovered", state);
    }

    private async Task ExecuteStage2Async(
        IncrementalWorkflowState state,
        IProgress<WorkflowProgress>? progress,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("Executing Stage 2: Pattern Expansion (0.5%)");
        ReportProgress(progress, WorkflowStage.PatternExpansion, 0.0, "Starting pattern expansion", state);

        var startTime = DateTime.UtcNow;
        var sampleRatio = state.Config.Stage2Ratio;

        // Load full dataset
        var fullData = LoadDataFrame(state.DatasetPath);

        // Sample data
        var sample = await _samplingEngine.SampleAsync(fullData, sampleRatio, null, null, cancellationToken);

        // Analyze sample
        var analysis = await _sampleAnalyzer.AnalyzeAsync(sample, 2, null, cancellationToken);

        // Discover additional rules
        var rules = await _ruleDiscoveryEngine.DiscoverRulesAsync(sample, analysis, cancellationToken);

        // Filter new rules (not in DiscoveredRules)
        var newRules = rules.Where(r => !state.DiscoveredRules.Any(dr => dr.Id == r.Id)).ToList();

        // Update state
        var stageResult = new StageResult
        {
            Stage = WorkflowStage.PatternExpansion,
            SampleSize = (int)sample.Rows.Count,
            SampleRatio = sampleRatio,
            Analysis = analysis,
            RulesDiscovered = newRules,
            Duration = DateTime.UtcNow - startTime
        };

        state.CompletedStages[WorkflowStage.PatternExpansion] = stageResult;
        state.DiscoveredRules.AddRange(newRules);
        state.CurrentStage = WorkflowStage.HITLDecision;

        // Check convergence
        if (newRules.Count == 0)
        {
            state.HasConverged = true;
            _logger.LogInformation("Rule discovery has converged (no new rules found)");
        }

        ReportProgress(progress, WorkflowStage.PatternExpansion, 1.0,
            $"Stage 2 complete: {newRules.Count} new rules discovered", state);
    }

    private async Task ExecuteStage3Async(
        IncrementalWorkflowState state,
        IProgress<WorkflowProgress>? progress,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("Executing Stage 3: HITL Decision (1.5%)");
        ReportProgress(progress, WorkflowStage.HITLDecision, 0.0, "Starting HITL decision process", state);

        var startTime = DateTime.UtcNow;
        var sampleRatio = state.Config.Stage3Ratio;

        // Load full dataset
        var fullData = LoadDataFrame(state.DatasetPath);

        // Sample data
        var sample = await _samplingEngine.SampleAsync(fullData, sampleRatio, null, null, cancellationToken);

        // Analyze sample
        var analysis = await _sampleAnalyzer.AnalyzeAsync(sample, 3, null, cancellationToken);

        // Execute HITL workflow (only for unapproved rules)
        var unapprovedRules = state.DiscoveredRules.Where(r => !r.IsApproved).ToList();

        IReadOnlyList<HITLDecisionLog> decisions;
        if (state.Config.SkipHITL)
        {
            _logger.LogInformation("Skipping HITL (SkipHITL=true), using recommended defaults");
            decisions = new List<HITLDecisionLog>();

            // Auto-approve all rules
            foreach (var rule in unapprovedRules)
            {
                rule.IsApproved = true;
                rule.UserFeedback = "Auto-approved (SkipHITL=true)";
                state.ApprovedRules.Add(rule);
            }
        }
        else
        {
            decisions = await _hitlWorkflowService.ExecuteWorkflowAsync(
                unapprovedRules, sample, analysis);

            // Update approved rules based on HITL decisions
            foreach (var decision in decisions)
            {
                var rule = decision.ApprovedRule;
                if (rule != null && rule.IsApproved && !state.ApprovedRules.Any(r => r.Id == rule.Id))
                {
                    state.ApprovedRules.Add(rule);
                }
            }
        }

        // Update state
        var stageResult = new StageResult
        {
            Stage = WorkflowStage.HITLDecision,
            SampleSize = (int)sample.Rows.Count,
            SampleRatio = sampleRatio,
            Analysis = analysis,
            RulesDiscovered = new List<PreprocessingRule>(), // No new rules in HITL stage
            Duration = DateTime.UtcNow - startTime,
            Notes = $"HITL decisions: {decisions.Count}, Approved rules: {state.ApprovedRules.Count}"
        };

        state.CompletedStages[WorkflowStage.HITLDecision] = stageResult;
        state.CurrentStage = WorkflowStage.ConfidenceCheckpoint;

        ReportProgress(progress, WorkflowStage.HITLDecision, 1.0,
            $"Stage 3 complete: {state.ApprovedRules.Count} rules approved", state);
    }

    private async Task ExecuteStage4Async(
        IncrementalWorkflowState state,
        IProgress<WorkflowProgress>? progress,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("Executing Stage 4: Confidence Checkpoint (2.5%)");
        ReportProgress(progress, WorkflowStage.ConfidenceCheckpoint, 0.0, "Starting confidence validation", state);

        var startTime = DateTime.UtcNow;
        var sampleRatio = state.Config.Stage4Ratio;

        // Load full dataset
        var fullData = LoadDataFrame(state.DatasetPath);

        // Sample data
        var sample = await _samplingEngine.SampleAsync(fullData, sampleRatio, null, null, cancellationToken);

        // Analyze sample
        var analysis = await _sampleAnalyzer.AnalyzeAsync(sample, 4, null, cancellationToken);

        // Calculate confidence score based on rule stability
        var confidenceScore = CalculateConfidenceScore(state, analysis);
        state.ConfidenceScore = confidenceScore;

        _logger.LogInformation("Confidence score: {Score:P2}", confidenceScore);

        // Auto-approval check
        if (state.Config.EnableAutoApproval && confidenceScore >= state.Config.MinConfidenceThreshold)
        {
            _logger.LogInformation("Auto-approving all rules (confidence threshold met)");

            foreach (var rule in state.DiscoveredRules.Where(r => !r.IsApproved))
            {
                rule.IsApproved = true;
                rule.UserFeedback = $"Auto-approved (confidence={confidenceScore:P2})";
                state.ApprovedRules.Add(rule);
            }
        }

        // Update state
        var stageResult = new StageResult
        {
            Stage = WorkflowStage.ConfidenceCheckpoint,
            SampleSize = (int)sample.Rows.Count,
            SampleRatio = sampleRatio,
            Analysis = analysis,
            RulesDiscovered = new List<PreprocessingRule>(),
            Duration = DateTime.UtcNow - startTime,
            Notes = $"Confidence: {confidenceScore:P2}, Converged: {state.HasConverged}"
        };

        state.CompletedStages[WorkflowStage.ConfidenceCheckpoint] = stageResult;
        state.CurrentStage = WorkflowStage.BulkProcessing;

        ReportProgress(progress, WorkflowStage.ConfidenceCheckpoint, 1.0,
            $"Stage 4 complete: Confidence {confidenceScore:P2}", state);
    }

    private async Task ExecuteStage5Async(
        IncrementalWorkflowState state,
        IProgress<WorkflowProgress>? progress,
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("Executing Stage 5: Bulk Processing (100%)");
        ReportProgress(progress, WorkflowStage.BulkProcessing, 0.0, "Starting bulk processing", state);

        var startTime = DateTime.UtcNow;

        // Note: Actual bulk processing will be implemented in Phase 2 (Rule Application Engine)
        // For now, we just record the stage completion

        var stageResult = new StageResult
        {
            Stage = WorkflowStage.BulkProcessing,
            SampleSize = (int)state.TotalRecords,
            SampleRatio = 1.0,
            Analysis = new SampleAnalysis
            {
                StageNumber = 5,
                RowCount = state.TotalRecords,
                ColumnCount = 0, // Will be updated in Phase 2
                Columns = new List<ColumnAnalysis>(),
                SampleRatio = 1.0,
                Timestamp = DateTime.UtcNow,
                QualityScore = state.ConfidenceScore
            },
            RulesDiscovered = new List<PreprocessingRule>(),
            Duration = DateTime.UtcNow - startTime,
            Notes = $"Applied {state.ApprovedRules.Count} rules to full dataset"
        };

        state.CompletedStages[WorkflowStage.BulkProcessing] = stageResult;

        ReportProgress(progress, WorkflowStage.BulkProcessing, 1.0,
            "Stage 5 complete: Bulk processing finished", state);

        await Task.CompletedTask; // Placeholder for actual bulk processing
    }

    // ===== Helper Methods =====

    private IncrementalWorkflowState CreateInitialState(string datasetPath, IncrementalWorkflowConfig config)
    {
        // Get total record count
        long totalRecords = GetTotalRecordCount(datasetPath);

        return new IncrementalWorkflowState
        {
            SessionId = Guid.NewGuid().ToString("N"),
            CurrentStage = WorkflowStage.InitialExploration,
            DatasetPath = datasetPath,
            TotalRecords = totalRecords,
            CompletedStages = new Dictionary<WorkflowStage, StageResult>(),
            DiscoveredRules = new List<PreprocessingRule>(),
            ApprovedRules = new List<PreprocessingRule>(),
            ConfidenceScore = 0.0,
            HasConverged = false,
            StartedAt = DateTime.UtcNow,
            Config = config
        };
    }

    private long GetTotalRecordCount(string datasetPath)
    {
        try
        {
            // Simple line count (CSV rows - header)
            return File.ReadLines(datasetPath).Count() - 1;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to count records in {Path}, using estimate", datasetPath);
            return 100000; // Default estimate
        }
    }

    private double CalculateConfidenceScore(IncrementalWorkflowState state, SampleAnalysis analysis)
    {
        // Confidence factors:
        // 1. Rule convergence (no new rules in recent stages)
        // 2. Sample quality score
        // 3. Number of approved vs discovered rules

        var convergenceFactor = state.HasConverged ? 1.0 : 0.8;
        var qualityFactor = analysis.QualityScore;
        var approvalRatio = state.DiscoveredRules.Count > 0
            ? (double)state.ApprovedRules.Count / state.DiscoveredRules.Count
            : 0.0;

        var confidence = (convergenceFactor * 0.4) + (qualityFactor * 0.3) + (approvalRatio * 0.3);

        return Math.Clamp(confidence, 0.0, 1.0);
    }

    private void ReportProgress(
        IProgress<WorkflowProgress>? progress,
        WorkflowStage stage,
        double percentage,
        string message,
        IncrementalWorkflowState state)
    {
        progress?.Report(new WorkflowProgress
        {
            Stage = stage,
            Percentage = percentage,
            Message = message,
            RulesDiscovered = state.DiscoveredRules.Count,
            ConfidenceScore = state.ConfidenceScore,
            HasConverged = state.HasConverged
        });
    }

    private async Task SaveCheckpointIfEnabledAsync(IncrementalWorkflowState state, IncrementalWorkflowConfig config)
    {
        if (config.EnableCheckpoints)
        {
            await SaveCheckpointAsync(state);
        }
    }

    private string GetDefaultCheckpointPath(IncrementalWorkflowState state)
    {
        var directory = state.Config.CheckpointDirectory;
        var filename = $"checkpoint-{state.SessionId}-{state.CurrentStage}.json";
        return Path.Combine(directory, filename);
    }

    private DataFrame LoadDataFrame(string csvPath)
    {
        try
        {
            return DataFrame.LoadCsv(csvPath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load DataFrame from {Path}", csvPath);
            throw new InvalidOperationException($"Failed to load dataset from: {csvPath}", ex);
        }
    }
}
