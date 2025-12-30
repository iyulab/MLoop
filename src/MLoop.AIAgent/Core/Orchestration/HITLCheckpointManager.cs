// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using Ironbees.Core.Goals;
using MLoop.AIAgent.Core.Integration;

namespace MLoop.AIAgent.Core.Orchestration;

/// <summary>
/// Manages HITL (Human-in-the-Loop) checkpoints throughout the orchestration workflow.
/// Defines 5 checkpoint types with their configurations and auto-approval logic.
/// </summary>
public class HITLCheckpointManager
{
    private readonly AgenticSettings? _agenticSettings;

    /// <summary>
    /// Checkpoint definitions for each HITL state.
    /// </summary>
    public static readonly Dictionary<OrchestrationState, HITLCheckpointDefinition> CheckpointDefinitions = new()
    {
        [OrchestrationState.DataAnalysisReview] = new HITLCheckpointDefinition
        {
            Id = "data-analysis-review",
            Name = "Data Analysis Review",
            Description = "Review data analysis results and confirm target variable and task type",
            Phase = 1,
            AutoApprovalThreshold = 0.85,
            RequiresExplicitApproval = false,
            DefaultOptions = [
                new HitlOption { Id = "approve", Label = "Approve", Description = "Accept the analysis and proceed", IsDefault = true, Shortcut = "Y" },
                new HitlOption { Id = "modify", Label = "Modify", Description = "Modify target or task type settings", Shortcut = "M" },
                new HitlOption { Id = "reanalyze", Label = "Re-analyze", Description = "Run analysis again with different parameters", Shortcut = "R" },
                new HitlOption { Id = "cancel", Label = "Cancel", Description = "Cancel orchestration", Shortcut = "C" }
            ]
        },

        [OrchestrationState.ModelSelectionReview] = new HITLCheckpointDefinition
        {
            Id = "model-selection-review",
            Name = "Model Selection Review",
            Description = "Review model recommendations and AutoML configuration",
            Phase = 2,
            AutoApprovalThreshold = 0.80,
            RequiresExplicitApproval = false,
            DefaultOptions = [
                new HitlOption { Id = "approve", Label = "Approve", Description = "Accept recommendations and proceed to preprocessing", IsDefault = true, Shortcut = "Y" },
                new HitlOption { Id = "modify", Label = "Modify", Description = "Adjust model selection or training parameters", Shortcut = "M" },
                new HitlOption { Id = "skip-training", Label = "Skip Training", Description = "Skip to evaluation with existing model", Shortcut = "S" },
                new HitlOption { Id = "cancel", Label = "Cancel", Description = "Cancel orchestration", Shortcut = "C" }
            ]
        },

        [OrchestrationState.PreprocessingReview] = new HITLCheckpointDefinition
        {
            Id = "preprocessing-review",
            Name = "Preprocessing Review",
            Description = "Review preprocessing pipeline and transformation steps",
            Phase = 3,
            AutoApprovalThreshold = 0.75,
            RequiresExplicitApproval = false,
            DefaultOptions = [
                new HitlOption { Id = "approve", Label = "Approve", Description = "Accept preprocessing and proceed to training", IsDefault = true, Shortcut = "Y" },
                new HitlOption { Id = "modify", Label = "Modify", Description = "Adjust preprocessing steps", Shortcut = "M" },
                new HitlOption { Id = "skip", Label = "Skip Preprocessing", Description = "Use raw data without preprocessing", Shortcut = "S" },
                new HitlOption { Id = "cancel", Label = "Cancel", Description = "Cancel orchestration", Shortcut = "C" }
            ]
        },

        [OrchestrationState.TrainingReview] = new HITLCheckpointDefinition
        {
            Id = "training-review",
            Name = "Training Results Review",
            Description = "Review training results and model performance",
            Phase = 4,
            AutoApprovalThreshold = double.MaxValue, // Always requires explicit approval
            RequiresExplicitApproval = true,
            DefaultOptions = [
                new HitlOption { Id = "approve", Label = "Approve", Description = "Accept model and proceed to evaluation", IsDefault = true, Shortcut = "Y" },
                new HitlOption { Id = "retrain", Label = "Retrain", Description = "Retrain with different parameters", Shortcut = "R" },
                new HitlOption { Id = "select-other", Label = "Select Other", Description = "Choose a different model from candidates", Shortcut = "O" },
                new HitlOption { Id = "cancel", Label = "Cancel", Description = "Cancel orchestration", Shortcut = "C" }
            ]
        },

        [OrchestrationState.DeploymentReview] = new HITLCheckpointDefinition
        {
            Id = "deployment-review",
            Name = "Deployment Approval",
            Description = "Final approval before deploying the model to production",
            Phase = 5,
            AutoApprovalThreshold = double.MaxValue, // Always requires explicit approval
            RequiresExplicitApproval = true,
            DefaultOptions = [
                new HitlOption { Id = "deploy", Label = "Deploy", Description = "Deploy model to production", IsDefault = true, Shortcut = "Y" },
                new HitlOption { Id = "export", Label = "Export Only", Description = "Export model without deploying", Shortcut = "E" },
                new HitlOption { Id = "save", Label = "Save for Later", Description = "Save model for later deployment", Shortcut = "S" },
                new HitlOption { Id = "cancel", Label = "Cancel", Description = "Cancel without deploying", Shortcut = "C" }
            ]
        }
    };

    public HITLCheckpointManager(AgenticSettings? agenticSettings = null)
    {
        _agenticSettings = agenticSettings;
    }

    /// <summary>
    /// Determines if HITL should be triggered for the given state.
    /// </summary>
    public bool ShouldTriggerHitl(OrchestrationState state, double confidence, OrchestrationOptions options)
    {
        // If HITL is globally skipped
        if (options.SkipHitl)
            return false;

        // If not a HITL checkpoint state
        if (!state.IsHitlCheckpoint())
            return false;

        // Get checkpoint definition
        if (!CheckpointDefinitions.TryGetValue(state, out var definition))
            return true; // Unknown checkpoint, trigger HITL

        // Always require explicit approval for certain checkpoints
        if (definition.RequiresExplicitApproval)
            return true;

        // Check auto-approval
        if (options.AutoApproveHighConfidence)
        {
            var threshold = Math.Min(definition.AutoApprovalThreshold, options.AutoApprovalThreshold);
            if (confidence >= threshold)
                return false; // Auto-approve
        }

        // Use ironbees HITL settings if available
        if (_agenticSettings?.Hitl != null)
        {
            return AgenticSettingsAdapter.ShouldTriggerHitl(
                _agenticSettings.Hitl,
                confidence,
                definition.Id,
                hasException: false);
        }

        return true; // Default to triggering HITL
    }

    /// <summary>
    /// Creates a HITL request event for the given state.
    /// </summary>
    public HitlRequestedEvent CreateHitlRequest(OrchestrationState state, OrchestrationContext context)
    {
        var definition = CheckpointDefinitions[state];
        var confidence = context.GetCurrentConfidence();

        return new HitlRequestedEvent
        {
            SessionId = context.SessionId,
            CheckpointId = definition.Id,
            CheckpointName = definition.Name,
            Question = BuildQuestion(state, context),
            Options = definition.DefaultOptions,
            Context = BuildHitlContext(state, context),
            Confidence = confidence,
            CanAutoApprove = !definition.RequiresExplicitApproval && confidence >= definition.AutoApprovalThreshold,
            Timeout = _agenticSettings?.Hitl?.ResponseTimeout
        };
    }

    /// <summary>
    /// Builds the question/prompt for a HITL checkpoint.
    /// </summary>
    private static string BuildQuestion(OrchestrationState state, OrchestrationContext context)
    {
        return state switch
        {
            OrchestrationState.DataAnalysisReview => BuildDataAnalysisQuestion(context),
            OrchestrationState.ModelSelectionReview => BuildModelSelectionQuestion(context),
            OrchestrationState.PreprocessingReview => BuildPreprocessingQuestion(context),
            OrchestrationState.TrainingReview => BuildTrainingQuestion(context),
            OrchestrationState.DeploymentReview => BuildDeploymentQuestion(context),
            _ => "Please review and approve to continue."
        };
    }

    private static string BuildDataAnalysisQuestion(OrchestrationContext context)
    {
        var analysis = context.DataAnalysis;
        if (analysis == null)
            return "Data analysis complete. Do you want to proceed?";

        return $"""
            Data Analysis Summary:
            - Rows: {analysis.RowCount:N0} | Columns: {analysis.ColumnCount}
            - Detected Target: '{analysis.DetectedTargetColumn ?? "Unknown"}' ({analysis.InferredTaskType ?? "Unknown"})
            - Data Quality: {analysis.DataQualityScore:P0}
            - Missing Values: {analysis.MissingValuePercentage:P1}

            Do you approve this analysis?
            """;
    }

    private static string BuildModelSelectionQuestion(OrchestrationContext context)
    {
        var rec = context.ModelRecommendation;
        if (rec == null)
            return "Model recommendation complete. Do you want to proceed?";

        var trainers = string.Join(", ", rec.RecommendedTrainers.Take(3).Select(t => t.TrainerName));
        return $"""
            Model Recommendation:
            - Task Type: {rec.TaskType}
            - Primary Metric: {rec.PrimaryMetric}
            - Recommended Trainers: {trainers}
            - Training Time: {rec.AutoMLConfig?.MaxTrainingTimeSeconds ?? 300}s

            Do you approve these settings?
            """;
    }

    private static string BuildPreprocessingQuestion(OrchestrationContext context)
    {
        var prep = context.Preprocessing;
        if (prep == null)
            return "Preprocessing complete. Do you want to proceed?";

        var steps = string.Join(", ", prep.Steps.Take(3).Select(s => s.Name));
        return $"""
            Preprocessing Summary:
            - Rows: {prep.RowsBefore:N0} → {prep.RowsAfter:N0}
            - Columns: {prep.ColumnsBefore} → {prep.ColumnsAfter}
            - Steps: {steps}{(prep.Steps.Count > 3 ? $" (+{prep.Steps.Count - 3} more)" : "")}
            - Used Incremental: {(prep.UsedIncrementalProcessing ? "Yes" : "No")}

            Do you approve this preprocessing pipeline?
            """;
    }

    private static string BuildTrainingQuestion(OrchestrationContext context)
    {
        var training = context.Training;
        if (training == null)
            return "Training complete. Do you want to proceed?";

        return $"""
            Training Results:
            - Best Model: {training.BestModelName}
            - {training.PrimaryMetricName}: {training.PrimaryMetricValue:F4}
            - Models Evaluated: {training.ModelsEvaluated}
            - Training Duration: {training.TrainingDuration.TotalMinutes:F1} minutes

            Do you approve this model?
            """;
    }

    private static string BuildDeploymentQuestion(OrchestrationContext context)
    {
        var training = context.Training;
        var eval = context.Evaluation;

        return $"""
            Ready for Deployment:
            - Model: {training?.BestModelName ?? "Unknown"}
            - Performance: {training?.PrimaryMetricName ?? "Metric"} = {training?.PrimaryMetricValue ?? 0:F4}
            {(eval?.Summary != null ? $"- Evaluation: {eval.Summary}" : "")}

            Do you want to deploy this model?
            """;
    }

    /// <summary>
    /// Builds context information for HITL decision making.
    /// </summary>
    private static Dictionary<string, object> BuildHitlContext(OrchestrationState state, OrchestrationContext context)
    {
        var ctx = new Dictionary<string, object>
        {
            ["session_id"] = context.SessionId,
            ["state"] = state.ToString(),
            ["elapsed_time"] = context.GetElapsedTime().ToString(@"hh\:mm\:ss"),
            ["progress"] = state.GetProgressPercentage()
        };

        switch (state)
        {
            case OrchestrationState.DataAnalysisReview when context.DataAnalysis != null:
                ctx["row_count"] = context.DataAnalysis.RowCount;
                ctx["column_count"] = context.DataAnalysis.ColumnCount;
                ctx["target_column"] = context.DataAnalysis.DetectedTargetColumn ?? "Unknown";
                ctx["task_type"] = context.DataAnalysis.InferredTaskType ?? "Unknown";
                ctx["data_quality"] = context.DataAnalysis.DataQualityScore;
                ctx["issues"] = context.DataAnalysis.Issues;
                break;

            case OrchestrationState.ModelSelectionReview when context.ModelRecommendation != null:
                ctx["task_type"] = context.ModelRecommendation.TaskType;
                ctx["primary_metric"] = context.ModelRecommendation.PrimaryMetric;
                ctx["trainers"] = context.ModelRecommendation.RecommendedTrainers.Select(t => t.TrainerName).ToList();
                ctx["rationale"] = context.ModelRecommendation.Rationale ?? "";
                break;

            case OrchestrationState.PreprocessingReview when context.Preprocessing != null:
                ctx["rows_before"] = context.Preprocessing.RowsBefore;
                ctx["rows_after"] = context.Preprocessing.RowsAfter;
                ctx["steps"] = context.Preprocessing.Steps.Select(s => s.Name).ToList();
                ctx["incremental"] = context.Preprocessing.UsedIncrementalProcessing;
                break;

            case OrchestrationState.TrainingReview when context.Training != null:
                ctx["best_model"] = context.Training.BestModelName;
                ctx["metrics"] = context.Training.Metrics;
                ctx["top_models"] = context.Training.TopModels.Select(m => new { m.ModelName, m.Score }).ToList();
                ctx["training_duration"] = context.Training.TrainingDuration.TotalSeconds;
                break;

            case OrchestrationState.DeploymentReview:
                ctx["model"] = context.Training?.BestModelName ?? "Unknown";
                ctx["performance"] = context.Training?.Metrics ?? new Dictionary<string, double>();
                ctx["model_path"] = context.Training?.ModelPath ?? "";
                break;
        }

        return ctx;
    }

    /// <summary>
    /// Processes a HITL response and determines next action.
    /// </summary>
    public HitlResponseAction ProcessResponse(OrchestrationState state, string selectedOptionId, OrchestrationContext context)
    {
        // Common cancel handling
        if (selectedOptionId == "cancel")
            return HitlResponseAction.Cancel;

        return state switch
        {
            OrchestrationState.DataAnalysisReview => selectedOptionId switch
            {
                "approve" => HitlResponseAction.Proceed,
                "modify" => HitlResponseAction.Modify,
                "reanalyze" => HitlResponseAction.Retry,
                _ => HitlResponseAction.Proceed
            },

            OrchestrationState.ModelSelectionReview => selectedOptionId switch
            {
                "approve" => HitlResponseAction.Proceed,
                "modify" => HitlResponseAction.Modify,
                "skip-training" => HitlResponseAction.Skip,
                _ => HitlResponseAction.Proceed
            },

            OrchestrationState.PreprocessingReview => selectedOptionId switch
            {
                "approve" => HitlResponseAction.Proceed,
                "modify" => HitlResponseAction.Modify,
                "skip" => HitlResponseAction.Skip,
                _ => HitlResponseAction.Proceed
            },

            OrchestrationState.TrainingReview => selectedOptionId switch
            {
                "approve" => HitlResponseAction.Proceed,
                "retrain" => HitlResponseAction.Retry,
                "select-other" => HitlResponseAction.Modify,
                _ => HitlResponseAction.Proceed
            },

            OrchestrationState.DeploymentReview => selectedOptionId switch
            {
                "deploy" => HitlResponseAction.Deploy,
                "export" => HitlResponseAction.Export,
                "save" => HitlResponseAction.Save,
                _ => HitlResponseAction.Proceed
            },

            _ => HitlResponseAction.Proceed
        };
    }

    /// <summary>
    /// Gets timeout action from ironbees settings.
    /// </summary>
    public string GetTimeoutActionDescription()
        => AgenticSettingsAdapter.GetTimeoutActionDescription(_agenticSettings?.Hitl);
}

/// <summary>
/// Definition of a HITL checkpoint.
/// </summary>
public record HITLCheckpointDefinition
{
    /// <summary>Unique identifier.</summary>
    public required string Id { get; init; }

    /// <summary>Display name.</summary>
    public required string Name { get; init; }

    /// <summary>Description of the checkpoint.</summary>
    public required string Description { get; init; }

    /// <summary>Phase number (1-5).</summary>
    public int Phase { get; init; }

    /// <summary>Confidence threshold for auto-approval.</summary>
    public double AutoApprovalThreshold { get; init; }

    /// <summary>Whether explicit approval is always required.</summary>
    public bool RequiresExplicitApproval { get; init; }

    /// <summary>Default options for this checkpoint.</summary>
    public required List<HitlOption> DefaultOptions { get; init; }
}

/// <summary>
/// Action to take after HITL response.
/// </summary>
public enum HitlResponseAction
{
    /// <summary>Proceed to next phase.</summary>
    Proceed,

    /// <summary>Modify current settings.</summary>
    Modify,

    /// <summary>Retry current phase.</summary>
    Retry,

    /// <summary>Skip current phase.</summary>
    Skip,

    /// <summary>Cancel orchestration.</summary>
    Cancel,

    /// <summary>Deploy to production.</summary>
    Deploy,

    /// <summary>Export model only.</summary>
    Export,

    /// <summary>Save for later.</summary>
    Save
}
