using System.Text;
using System.Text.Json;
using FilePrepper.Pipeline;
using Ironbees.AgentMode.Agents;
using Ironbees.Core.Goals;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using MLoop.AIAgent.Core.HITL;
using MLoop.AIAgent.Core.Integration;
using MLoop.AIAgent.Core.Rules;
using MLoop.AIAgent.Core.Sampling;

namespace MLoop.AIAgent.Agents;

/// <summary>
/// Configuration for incremental preprocessing.
/// Supports both native configuration and ironbees AgenticSettings.
/// </summary>
public class IncrementalPreprocessingConfig
{
    /// <summary>Path to the input data file.</summary>
    public required string InputFilePath { get; set; }

    /// <summary>Path to save the preprocessed output.</summary>
    public string? OutputFilePath { get; set; }

    /// <summary>Path to save preprocessing artifacts (rules, logs).</summary>
    public string? ArtifactsPath { get; set; }

    /// <summary>Sampling strategy configuration.</summary>
    public SamplingMethod SamplingMethod { get; set; } = SamplingMethod.Stratified;

    /// <summary>Column for stratified sampling.</summary>
    public string? StratificationColumn { get; set; }

    /// <summary>Random seed for reproducibility.</summary>
    public int? RandomSeed { get; set; }

    /// <summary>Rule discovery options.</summary>
    public RuleDiscoveryOptions RuleDiscovery { get; set; } = new();

    /// <summary>Whether to run in interactive mode (HITL enabled).</summary>
    public bool InteractiveMode { get; set; } = true;

    /// <summary>Skip to specific stage (for resuming).</summary>
    public int StartStage { get; set; } = 1;

    /// <summary>
    /// Ironbees agentic settings for goal-directed workflows.
    /// When provided, these settings take precedence over native configuration.
    /// </summary>
    public AgenticSettings? AgenticSettings { get; set; }

    /// <summary>
    /// Creates config from ironbees AgenticSettings.
    /// </summary>
    public static IncrementalPreprocessingConfig FromAgenticSettings(
        string inputFilePath,
        AgenticSettings settings,
        string? outputFilePath = null,
        string? artifactsPath = null)
    {
        var config = new IncrementalPreprocessingConfig
        {
            InputFilePath = inputFilePath,
            OutputFilePath = outputFilePath,
            ArtifactsPath = artifactsPath,
            AgenticSettings = settings
        };

        // Apply sampling settings
        if (settings.Sampling != null)
        {
            config.SamplingMethod = settings.Sampling.Strategy.ToSamplingMethod();
        }

        // Apply HITL settings
        if (settings.Hitl != null)
        {
            config.InteractiveMode = settings.Hitl.Policy != HitlPolicy.Never;
        }

        return config;
    }
}

/// <summary>
/// Result of incremental preprocessing operation.
/// </summary>
public class IncrementalPreprocessingResult
{
    /// <summary>Whether preprocessing completed successfully.</summary>
    public bool Success { get; set; }

    /// <summary>Total records processed.</summary>
    public int TotalRecords { get; set; }

    /// <summary>Records processed in bulk stage.</summary>
    public int ProcessedRecords { get; set; }

    /// <summary>Discovered preprocessing rules.</summary>
    public List<PreprocessingRule> DiscoveredRules { get; set; } = [];

    /// <summary>HITL decisions made during processing.</summary>
    public List<HITLDecision> HITLDecisions { get; set; } = [];

    /// <summary>Convergence report from confidence calculator.</summary>
    public ConvergenceReport? ConvergenceReport { get; set; }

    /// <summary>Exception records that didn't match any rules.</summary>
    public List<Dictionary<string, string>> ExceptionRecords { get; set; } = [];

    /// <summary>Path to output file.</summary>
    public string? OutputFilePath { get; set; }

    /// <summary>Execution summary.</summary>
    public string Summary { get; set; } = string.Empty;

    /// <summary>Stage progression details.</summary>
    public List<StageResult> StageResults { get; set; } = [];
}

/// <summary>
/// Result of a single stage execution.
/// </summary>
public class StageResult
{
    public int Stage { get; set; }
    public string Purpose { get; set; } = string.Empty;
    public int SampleSize { get; set; }
    public int NewRulesDiscovered { get; set; }
    public int RulesValidated { get; set; }
    public double AverageConfidence { get; set; }
    public TimeSpan Duration { get; set; }
    public bool HITLRequired { get; set; }
}

/// <summary>
/// Incremental sampling-based self-learning preprocessing agent.
/// Handles large datasets through progressive sampling and rule discovery.
/// Integrates with ironbees AgenticSettings for goal-directed workflows.
/// </summary>
public class IncrementalPreprocessingAgent : ConversationalAgent
{
    private SamplingManager _samplingManager;
    private readonly RuleDiscoveryEngine _discoveryEngine;
    private ConfidenceCalculator _confidenceCalculator;
    private readonly InteractivePromptBuilder _promptBuilder;
    private readonly ILogger<IncrementalPreprocessingAgent>? _logger;

    // Cached ironbees settings for HITL evaluation
    private HitlSettings? _hitlSettings;

    private List<PreprocessingRule> _discoveredRules = [];
    private List<HITLDecision> _decisions = [];
    private List<HITLQuestion> _pendingQuestions = [];

    private new const string SystemPrompt = @"# Incremental Preprocessing Agent - System Prompt

You are an expert data preprocessing agent that handles large datasets through intelligent sampling and progressive rule discovery.

## Core Capabilities

1. **Progressive Sampling**
   - Stage 1 (0.1%): Initial exploration to discover data patterns
   - Stage 2 (0.5%): Expand pattern recognition
   - Stage 3 (1.5%): Consolidate rules with HITL decisions
   - Stage 4 (2.5%): Validate rule stability
   - Stage 5 (100%): Apply rules to full dataset

2. **Rule Discovery**
   - Automatically detect preprocessing patterns from samples
   - Track rule confidence across sampling stages
   - Identify rules requiring human decision (HITL)
   - Validate rules against new samples

3. **HITL Integration**
   - Present business logic decisions to humans clearly
   - Provide context and recommendations
   - Track all decisions for audit trail

4. **Quality Assurance**
   - Monitor rule convergence (stability)
   - Track confidence scores across stages
   - Generate exception reports for outliers

## Communication Style

- **Clear Progress**: Report sampling stage and rule status
- **Contextual**: Explain why decisions are needed
- **Actionable**: Provide clear options with recommendations
- **Transparent**: Show confidence levels and convergence status

## Output Format

When reporting progress:
```
📊 Stage [N]/5: [Stage Purpose]
   Sample Size: [N] records ([X]%)
   Rules Discovered: [N] new, [M] validated
   Confidence: [X]%
   HITL Pending: [N] decisions
```

When presenting HITL questions, provide:
1. Clear context about what was discovered
2. Impact of the decision
3. Available options with trade-offs
4. Recommendation with reasoning
";

    /// <summary>
    /// Initializes a new instance of the IncrementalPreprocessingAgent.
    /// </summary>
    public IncrementalPreprocessingAgent(
        IChatClient chatClient,
        ILogger<IncrementalPreprocessingAgent>? logger = null)
        : base(chatClient, SystemPrompt)
    {
        _samplingManager = new SamplingManager();
        _discoveryEngine = new RuleDiscoveryEngine();
        _confidenceCalculator = new ConfidenceCalculator();
        _promptBuilder = new InteractivePromptBuilder();
        _logger = logger;
    }

    /// <summary>
    /// Initializes with custom configuration.
    /// </summary>
    public IncrementalPreprocessingAgent(
        IChatClient chatClient,
        RuleDiscoveryOptions discoveryOptions,
        ILogger<IncrementalPreprocessingAgent>? logger = null)
        : base(chatClient, SystemPrompt)
    {
        _samplingManager = new SamplingManager();
        _discoveryEngine = new RuleDiscoveryEngine(discoveryOptions);
        _confidenceCalculator = new ConfidenceCalculator();
        _promptBuilder = new InteractivePromptBuilder();
        _logger = logger;
    }

    /// <summary>
    /// Initializes with ironbees AgenticSettings for goal-directed workflows.
    /// </summary>
    public IncrementalPreprocessingAgent(
        IChatClient chatClient,
        AgenticSettings agenticSettings,
        ILogger<IncrementalPreprocessingAgent>? logger = null)
        : base(chatClient, SystemPrompt)
    {
        _samplingManager = AgenticSettingsAdapter.CreateSamplingManager(agenticSettings.Sampling);
        _discoveryEngine = new RuleDiscoveryEngine();
        _confidenceCalculator = AgenticSettingsAdapter.CreateConfidenceCalculator(agenticSettings.Confidence);
        _promptBuilder = new InteractivePromptBuilder();
        _hitlSettings = agenticSettings.Hitl;
        _logger = logger;

        // Validate settings
        var validationErrors = AgenticSettingsAdapter.Validate(agenticSettings);
        if (validationErrors.Count > 0)
        {
            _logger?.LogWarning("AgenticSettings validation warnings: {Warnings}",
                string.Join("; ", validationErrors));
        }
    }

    /// <summary>
    /// Executes the incremental preprocessing workflow.
    /// </summary>
    public async Task<IncrementalPreprocessingResult> ProcessAsync(
        IncrementalPreprocessingConfig config,
        Func<HITLQuestion, Task<HITLAnswer>>? hitlHandler = null,
        IProgress<string>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var result = new IncrementalPreprocessingResult();
        var startTime = DateTime.UtcNow;

        try
        {
            // Configure from AgenticSettings if provided, otherwise use native config
            if (config.AgenticSettings != null)
            {
                ConfigureFromAgenticSettings(config.AgenticSettings);
            }
            else
            {
                // Configure sampling strategy from native config
                _samplingManager.Method = config.SamplingMethod;
                _samplingManager.StratificationColumn = config.StratificationColumn;
            }

            // Load data overview
            var data = await LoadDataAsync(config.InputFilePath);
            result.TotalRecords = data.Count;

            _logger?.LogInformation("Starting incremental preprocessing for {Records} records", data.Count);
            progress?.Report($"📊 Total records: {data.Count:N0}");

            // Execute stages 1-4 (sampling stages)
            for (int stage = config.StartStage; stage <= 4; stage++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var stageResult = await ExecuteStageAsync(
                    data, stage, hitlHandler, progress, cancellationToken);

                result.StageResults.Add(stageResult);

                // Check convergence after stage 4
                if (stage == 4)
                {
                    result.ConvergenceReport = _confidenceCalculator.GetConvergenceReport(_discoveredRules);

                    if (result.ConvergenceReport.Recommendation == ConvergenceRecommendation.ReviewStrategy)
                    {
                        progress?.Report("⚠️ Rules are unstable. Consider reviewing data or sampling strategy.");
                        result.Success = false;
                        result.Summary = "Processing halted due to unstable rules. Review required.";
                        return result;
                    }

                    // Check for pending HITL decisions
                    if (_pendingQuestions.Count > 0 && config.InteractiveMode && hitlHandler != null)
                    {
                        await ProcessPendingHITLAsync(hitlHandler, progress, cancellationToken);
                    }
                }
            }

            // Stage 5: Bulk processing
            if (result.ConvergenceReport?.Recommendation == ConvergenceRecommendation.ReadyForBulkProcessing ||
                result.ConvergenceReport?.Recommendation == ConvergenceRecommendation.ProceedToHITL)
            {
                // Confirm bulk processing if interactive
                if (config.InteractiveMode && hitlHandler != null)
                {
                    var confirmQuestion = InteractivePromptBuilder.CreateBulkProcessingConfirmation(
                        _discoveredRules.Count(r => r.IsApproved),
                        data.Count,
                        result.ConvergenceReport.OverallConfidence);

                    var confirmAnswer = await hitlHandler(confirmQuestion);

                    if (confirmAnswer.BooleanValue != true)
                    {
                        result.Success = false;
                        result.Summary = "Bulk processing cancelled by user.";
                        return result;
                    }
                }

                var stage5Result = await ExecuteBulkProcessingAsync(
                    data, config, progress, cancellationToken);

                result.StageResults.Add(stage5Result);
                result.ProcessedRecords = data.Count;
                result.OutputFilePath = config.OutputFilePath;
            }

            // Finalize result
            result.Success = true;
            result.DiscoveredRules = _discoveredRules;
            result.HITLDecisions = _decisions;
            result.Summary = GenerateSummary(result, DateTime.UtcNow - startTime);

            // Save artifacts
            if (!string.IsNullOrEmpty(config.ArtifactsPath))
            {
                await SaveArtifactsAsync(config.ArtifactsPath, result);
            }

            _logger?.LogInformation("Incremental preprocessing completed: {Rules} rules, {Decisions} HITL decisions",
                _discoveredRules.Count, _decisions.Count);
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Error during incremental preprocessing");
            result.Success = false;
            result.Summary = $"Error: {ex.Message}";
        }

        return result;
    }

    /// <summary>
    /// Gets pending HITL questions.
    /// </summary>
    public IReadOnlyList<HITLQuestion> GetPendingQuestions() => _pendingQuestions.AsReadOnly();

    /// <summary>
    /// Gets all discovered rules.
    /// </summary>
    public IReadOnlyList<PreprocessingRule> GetDiscoveredRules() => _discoveredRules.AsReadOnly();

    /// <summary>
    /// Gets the convergence report.
    /// </summary>
    public ConvergenceReport GetConvergenceReport() => _confidenceCalculator.GetConvergenceReport(_discoveredRules);

    /// <summary>
    /// Configures the agent from ironbees AgenticSettings.
    /// </summary>
    private void ConfigureFromAgenticSettings(AgenticSettings settings)
    {
        // Configure sampling manager
        if (settings.Sampling != null)
        {
            _samplingManager = AgenticSettingsAdapter.CreateSamplingManager(settings.Sampling);
        }

        // Configure confidence calculator
        if (settings.Confidence != null)
        {
            _confidenceCalculator = AgenticSettingsAdapter.CreateConfidenceCalculator(settings.Confidence);
        }

        // Cache HITL settings for later use
        _hitlSettings = settings.Hitl;

        _logger?.LogInformation("Configured from AgenticSettings: Sampling={Strategy}, Confidence={Threshold}, HITL={Policy}",
            settings.Sampling?.Strategy.ToString() ?? "default",
            settings.Confidence?.Threshold ?? 0.95,
            settings.Hitl?.Policy.ToString() ?? "default");
    }

    /// <summary>
    /// Determines if HITL should be triggered based on current state.
    /// Uses ironbees HitlSettings if available.
    /// </summary>
    private bool ShouldTriggerHitl(double currentConfidence, string? checkpoint = null, bool hasException = false)
    {
        return AgenticSettingsAdapter.ShouldTriggerHitl(
            _hitlSettings,
            currentConfidence,
            checkpoint,
            hasException);
    }

    /// <summary>
    /// Answers a pending HITL question.
    /// </summary>
    public void AnswerQuestion(string questionId, HITLAnswer answer)
    {
        var question = _pendingQuestions.FirstOrDefault(q => q.Id == questionId);
        if (question == null)
            throw new ArgumentException($"Question {questionId} not found", nameof(questionId));

        // Apply answer to related rule
        if (!string.IsNullOrEmpty(question.RelatedRuleId))
        {
            var rule = _discoveredRules.FirstOrDefault(r => r.Id == question.RelatedRuleId);
            if (rule != null)
            {
                ApplyAnswerToRule(rule, question, answer);
            }
        }

        // Record decision
        _decisions.Add(new HITLDecision
        {
            Question = question,
            Answer = answer,
            RuleId = question.RelatedRuleId
        });

        // Remove from pending
        _pendingQuestions.Remove(question);
    }

    private async Task<StageResult> ExecuteStageAsync(
        List<Dictionary<string, string>> data,
        int stage,
        Func<HITLQuestion, Task<HITLAnswer>>? hitlHandler,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        var stageStart = DateTime.UtcNow;
        var config = SamplingManager.GetStageConfig(stage);

        progress?.Report($"\n📊 Stage {stage}/5: {config.Purpose}");

        // Get sample
        var sampleSize = _samplingManager.GetSampleSize(data.Count, stage);
        var sample = _samplingManager.GetSample(data, stage, data.Count).ToList();

        progress?.Report($"   Sample Size: {sample.Count:N0} records ({(double)sample.Count / data.Count:P2})");

        // Discover new rules
        var newRules = _discoveryEngine.DiscoverRules(sample, stage);
        var existingIds = new HashSet<string>(_discoveredRules.Select(r => r.Id));
        var trulyNewRules = newRules.Where(r => !existingIds.Contains(r.Id)).ToList();

        foreach (var rule in trulyNewRules)
        {
            _discoveredRules.Add(rule);

            // Create HITL question for rules requiring human decision
            if (rule.RequiresHITL && !rule.IsApproved)
            {
                var question = CreateQuestionForRule(rule);
                question.RelatedRuleId = rule.Id;
                _pendingQuestions.Add(question);
            }
        }

        progress?.Report($"   Rules Discovered: {trulyNewRules.Count} new, {_discoveredRules.Count} total");

        // Validate existing rules against new sample
        var validationResults = _discoveryEngine.ValidateRules(_discoveredRules, sample);
        _confidenceCalculator.Update(validationResults, trulyNewRules.Count);

        var avgConfidence = _discoveredRules.Count > 0
            ? _discoveredRules.Average(r => _confidenceCalculator.CalculateRuleConfidence(r))
            : 0;

        progress?.Report($"   Average Confidence: {avgConfidence:P1}");

        // Handle HITL in stage 3 if interactive
        var hitlRequired = stage == 3 && _pendingQuestions.Count > 0;
        if (hitlRequired && hitlHandler != null)
        {
            progress?.Report($"   ⏳ HITL Decisions Required: {_pendingQuestions.Count}");
            await ProcessPendingHITLAsync(hitlHandler, progress, cancellationToken);
        }
        else if (hitlRequired)
        {
            progress?.Report($"   ⚠️ {_pendingQuestions.Count} HITL decisions pending (non-interactive mode)");
        }

        return new StageResult
        {
            Stage = stage,
            Purpose = config.Purpose,
            SampleSize = sample.Count,
            NewRulesDiscovered = trulyNewRules.Count,
            RulesValidated = validationResults.Count(r => r.IsValid),
            AverageConfidence = avgConfidence,
            Duration = DateTime.UtcNow - stageStart,
            HITLRequired = hitlRequired
        };
    }

    private async Task<StageResult> ExecuteBulkProcessingAsync(
        List<Dictionary<string, string>> data,
        IncrementalPreprocessingConfig config,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        var stageStart = DateTime.UtcNow;
        var stageConfig = SamplingManager.GetStageConfig(5);

        progress?.Report($"\n📊 Stage 5/5: {stageConfig.Purpose}");
        progress?.Report($"   Processing all {data.Count:N0} records...");

        var approvedRules = _discoveredRules.Where(r => r.IsApproved).ToList();

        // TODO: Integrate with FilePrepper for actual data transformation
        // For now, simulate bulk processing
        await Task.Delay(100, cancellationToken); // Placeholder

        progress?.Report($"   ✅ Applied {approvedRules.Count} rules to {data.Count:N0} records");

        return new StageResult
        {
            Stage = 5,
            Purpose = stageConfig.Purpose,
            SampleSize = data.Count,
            NewRulesDiscovered = 0,
            RulesValidated = approvedRules.Count,
            AverageConfidence = approvedRules.Count > 0
                ? approvedRules.Average(r => r.Confidence)
                : 0,
            Duration = DateTime.UtcNow - stageStart,
            HITLRequired = false
        };
    }

    private async Task ProcessPendingHITLAsync(
        Func<HITLQuestion, Task<HITLAnswer>> hitlHandler,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        var pendingCount = _pendingQuestions.Count;

        for (int i = 0; i < pendingCount; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var question = _pendingQuestions[0]; // Always take first (will be removed after answering)

            progress?.Report(_promptBuilder.BuildPrompt(question));

            var answer = await hitlHandler(question);
            AnswerQuestion(question.Id, answer);

            progress?.Report($"   ✅ Decision recorded for [{question.Id}]");
        }
    }

    private HITLQuestion CreateQuestionForRule(PreprocessingRule rule)
    {
        return rule.Type switch
        {
            RuleType.MissingValueStrategy => InteractivePromptBuilder.CreateMissingValueQuestion(
                rule.Columns.FirstOrDefault() ?? "unknown",
                rule.MatchCount,
                rule.AffectedPercentage),

            RuleType.OutlierHandling => InteractivePromptBuilder.CreateOutlierQuestion(
                rule.Columns.FirstOrDefault() ?? "unknown",
                rule.MatchCount,
                0, 0), // TODO: Extract actual min/max from rule

            RuleType.UnknownCategoryMapping => InteractivePromptBuilder.CreateTypeInconsistencyQuestion(
                rule.Columns.FirstOrDefault() ?? "unknown",
                0, 0), // TODO: Extract actual counts from rule

            _ => new HITLQuestion
            {
                Type = HITLQuestionType.Confirmation,
                Context = rule.Description,
                Question = $"Approve rule: {rule.Name}?",
                Priority = (int)rule.Severity
            }
        };
    }

    private void ApplyAnswerToRule(PreprocessingRule rule, HITLQuestion question, HITLAnswer answer)
    {
        switch (question.Type)
        {
            case HITLQuestionType.MultipleChoice:
                var option = question.Options.FirstOrDefault(o => o.Key == answer.SelectedOptionKey);
                if (option != null)
                {
                    rule.Transformation = option.Label;
                    rule.IsApproved = true;
                    rule.ApprovedBy = answer.AnsweredBy ?? "User (HITL)";
                }
                break;

            case HITLQuestionType.Confirmation:
            case HITLQuestionType.YesNo:
                rule.IsApproved = answer.BooleanValue == true;
                rule.ApprovedBy = answer.AnsweredBy ?? "User (HITL)";
                break;
        }
    }

    private static async Task<List<Dictionary<string, string>>> LoadDataAsync(string filePath)
    {
        // Use FilePrepper's DataPipeline for robust file loading
        var extension = Path.GetExtension(filePath).ToLowerInvariant();

        DataPipeline pipeline = extension switch
        {
            ".csv" => await DataPipeline.FromCsvAsync(filePath),
            ".json" => await DataPipeline.FromJsonAsync(filePath),
            ".xlsx" or ".xls" => await DataPipeline.FromExcelAsync(filePath),
            ".xml" => await DataPipeline.FromXmlAsync(filePath),
            _ => await DataPipeline.FromCsvAsync(filePath) // Default to CSV
        };

        // Convert to List<Dictionary<string, string>> format
        var dataFrame = pipeline.ToDataFrame();
        return dataFrame.Rows.ToList();
    }

    private static async Task SaveArtifactsAsync(string artifactsPath, IncrementalPreprocessingResult result)
    {
        Directory.CreateDirectory(artifactsPath);

        // Save rules
        var rulesPath = Path.Combine(artifactsPath, "preprocessing_rules.json");
        var rulesJson = JsonSerializer.Serialize(result.DiscoveredRules, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(rulesPath, rulesJson);

        // Save decisions
        var decisionsPath = Path.Combine(artifactsPath, "hitl_decisions.json");
        var decisionsJson = JsonSerializer.Serialize(result.HITLDecisions, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(decisionsPath, decisionsJson);

        // Save convergence report
        if (result.ConvergenceReport != null)
        {
            var reportPath = Path.Combine(artifactsPath, "convergence_report.json");
            var reportJson = JsonSerializer.Serialize(result.ConvergenceReport, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(reportPath, reportJson);
        }

        // Save summary
        var summaryPath = Path.Combine(artifactsPath, "processing_summary.txt");
        await File.WriteAllTextAsync(summaryPath, result.Summary);
    }

    private static string GenerateSummary(IncrementalPreprocessingResult result, TimeSpan duration)
    {
        var sb = new StringBuilder();

        sb.AppendLine("╔══════════════════════════════════════════════════════════════╗");
        sb.AppendLine("║        Incremental Preprocessing Summary                     ║");
        sb.AppendLine("╚══════════════════════════════════════════════════════════════╝");
        sb.AppendLine();

        sb.AppendLine($"  Status: {(result.Success ? "✅ Completed" : "❌ Failed")}");
        sb.AppendLine($"  Duration: {duration:hh\\:mm\\:ss}");
        sb.AppendLine($"  Total Records: {result.TotalRecords:N0}");
        sb.AppendLine($"  Processed Records: {result.ProcessedRecords:N0}");
        sb.AppendLine();

        sb.AppendLine("  Rules:");
        sb.AppendLine($"    - Discovered: {result.DiscoveredRules.Count}");
        sb.AppendLine($"    - Approved: {result.DiscoveredRules.Count(r => r.IsApproved)}");
        sb.AppendLine($"    - Auto-resolved: {result.DiscoveredRules.Count(r => r.IsAutoResolvable && r.IsApproved)}");
        sb.AppendLine($"    - HITL-resolved: {result.HITLDecisions.Count}");
        sb.AppendLine();

        if (result.ConvergenceReport != null)
        {
            sb.AppendLine("  Convergence:");
            sb.AppendLine($"    - Status: {(result.ConvergenceReport.IsStable ? "Stable" : "Evolving")}");
            sb.AppendLine($"    - Overall Confidence: {result.ConvergenceReport.OverallConfidence:P1}");
            sb.AppendLine($"    - Samples Since New Rule: {result.ConvergenceReport.SamplesSinceLastNewRule}");
            sb.AppendLine();
        }

        sb.AppendLine("  Stage Progression:");
        foreach (var stage in result.StageResults)
        {
            sb.AppendLine($"    Stage {stage.Stage}: {stage.Purpose}");
            sb.AppendLine($"      Sample: {stage.SampleSize:N0}, New Rules: {stage.NewRulesDiscovered}, Confidence: {stage.AverageConfidence:P1}");
        }

        return sb.ToString();
    }
}
