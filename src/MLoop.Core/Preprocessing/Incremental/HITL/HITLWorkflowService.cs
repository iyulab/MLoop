using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging;
using MLoop.Core.Preprocessing.Incremental.HITL.Contracts;
using MLoop.Core.Preprocessing.Incremental.HITL.Models;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.HITL;

/// <summary>
/// Orchestrates the complete HITL workflow for preprocessing decisions.
/// </summary>
public sealed class HITLWorkflowService
{
    private readonly IHITLQuestionGenerator _questionGenerator;
    private readonly IHITLPromptBuilder _promptBuilder;
    private readonly IHITLDecisionLogger _decisionLogger;
    private readonly ILogger<HITLWorkflowService> _logger;

    public HITLWorkflowService(
        IHITLQuestionGenerator questionGenerator,
        IHITLPromptBuilder promptBuilder,
        IHITLDecisionLogger decisionLogger,
        ILogger<HITLWorkflowService> logger)
    {
        _questionGenerator = questionGenerator;
        _promptBuilder = promptBuilder;
        _decisionLogger = decisionLogger;
        _logger = logger;
    }

    /// <summary>
    /// Executes the complete HITL workflow for all rules requiring human input.
    /// </summary>
    /// <param name="rules">Preprocessing rules discovered.</param>
    /// <param name="sample">Data sample for context.</param>
    /// <param name="analysis">Sample analysis results.</param>
    /// <returns>List of approved decisions ready for application.</returns>
    public async Task<IReadOnlyList<HITLDecisionLog>> ExecuteWorkflowAsync(
        IReadOnlyList<PreprocessingRule> rules,
        DataFrame sample,
        SampleAnalysis analysis)
    {
        _logger.LogInformation("Starting HITL workflow for {RuleCount} rules", rules.Count);

        // Generate all questions
        var questions = _questionGenerator.GenerateAllQuestions(rules, sample, analysis);

        if (questions.Count == 0)
        {
            _logger.LogInformation("No HITL questions generated. All rules can be auto-applied.");
            return Array.Empty<HITLDecisionLog>();
        }

        _logger.LogInformation("Generated {QuestionCount} HITL questions", questions.Count);

        var decisions = new List<HITLDecisionLog>();
        var sessionId = Guid.NewGuid().ToString();

        // Process each question
        foreach (var question in questions)
        {
            try
            {
                // Display question
                _promptBuilder.BuildPrompt(question);

                // Collect answer
                var answer = _promptBuilder.CollectAnswer(question);

                // Show confirmation
                _promptBuilder.DisplayConfirmation(answer);

                // Determine selected action
                var selectedOption = question.Options
                    .FirstOrDefault(o => o.Key == answer.SelectedOption);

                var selectedAction = selectedOption?.Action ?? ActionType.KeepAsIs;

                // Mark rule as approved and update feedback
                var approvedRule = question.RelatedRule;
                approvedRule.IsApproved = true;
                approvedRule.UserFeedback = $"{selectedAction}: {selectedOption?.Label ?? "Unknown"}";

                // Create decision log
                var decisionLog = new HITLDecisionLog
                {
                    Id = Guid.NewGuid().ToString(),
                    SessionId = sessionId,
                    Question = question,
                    Answer = answer,
                    ApprovedRule = approvedRule,
                    UserId = Environment.UserName,
                    Notes = answer.SelectedOption == question.RecommendedOption
                        ? "Followed AI recommendation"
                        : "Overrode AI recommendation"
                };

                // Log decision
                await _decisionLogger.LogDecisionAsync(decisionLog);

                decisions.Add(decisionLog);

                _logger.LogInformation(
                    "Processed HITL decision for rule {RuleId}: {Action}",
                    question.RelatedRule.Id, selectedAction);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex,
                    "Failed to process HITL question {QuestionId}. Skipping.",
                    question.Id);
            }
        }

        _logger.LogInformation(
            "HITL workflow completed. Collected {DecisionCount} decisions.",
            decisions.Count);

        return decisions;
    }

    /// <summary>
    /// Executes HITL workflow for a single rule.
    /// </summary>
    /// <param name="rule">The rule requiring human input.</param>
    /// <param name="sample">Data sample for context.</param>
    /// <param name="analysis">Sample analysis results.</param>
    /// <returns>The approved decision.</returns>
    public async Task<HITLDecisionLog> ExecuteSingleRuleWorkflowAsync(
        PreprocessingRule rule,
        DataFrame sample,
        SampleAnalysis analysis)
    {
        _logger.LogInformation("Starting HITL workflow for rule {RuleId}", rule.Id);

        // Generate question
        var question = _questionGenerator.GenerateQuestion(rule, sample, analysis);

        // Display question
        _promptBuilder.BuildPrompt(question);

        // Collect answer
        var answer = _promptBuilder.CollectAnswer(question);

        // Show confirmation
        _promptBuilder.DisplayConfirmation(answer);

        // Determine selected action
        var selectedOption = question.Options
            .FirstOrDefault(o => o.Key == answer.SelectedOption);

        var selectedAction = selectedOption?.Action ?? ActionType.KeepAsIs;

        // Mark rule as approved and update feedback
        var approvedRule = rule;
        approvedRule.IsApproved = true;
        approvedRule.UserFeedback = $"{selectedAction}: {selectedOption?.Label ?? "Unknown"}";

        // Create decision log
        var sessionId = Guid.NewGuid().ToString();
        var decisionLog = new HITLDecisionLog
        {
            Id = Guid.NewGuid().ToString(),
            SessionId = sessionId,
            Question = question,
            Answer = answer,
            ApprovedRule = approvedRule,
            UserId = Environment.UserName,
            Notes = answer.SelectedOption == question.RecommendedOption
                ? "Followed AI recommendation"
                : "Overrode AI recommendation"
        };

        // Log decision
        await _decisionLogger.LogDecisionAsync(decisionLog);

        _logger.LogInformation(
            "HITL workflow completed for rule {RuleId}: {Action}",
            rule.Id, selectedAction);

        return decisionLog;
    }

    /// <summary>
    /// Gets a summary of all HITL decisions for reporting.
    /// </summary>
    /// <returns>Decision summary statistics.</returns>
    public async Task<HITLDecisionSummary> GetDecisionSummaryAsync()
    {
        return await _decisionLogger.GetDecisionSummaryAsync();
    }
}
