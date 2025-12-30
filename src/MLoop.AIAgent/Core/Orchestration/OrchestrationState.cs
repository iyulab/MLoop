// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

namespace MLoop.AIAgent.Core.Orchestration;

/// <summary>
/// Represents the states in the MLOps orchestration workflow.
/// State machine progression: START → DATA_ANALYSIS → MODEL_RECOMMENDATION → PREPROCESSING → TRAINING → EVALUATION → DEPLOYMENT → COMPLETED
/// </summary>
public enum OrchestrationState
{
    /// <summary>Initial state before orchestration begins.</summary>
    NotStarted = 0,

    /// <summary>Orchestration has started, initializing resources.</summary>
    Initializing = 1,

    /// <summary>Data analysis phase - DataAnalyst agent examines the dataset.</summary>
    DataAnalysis = 10,

    /// <summary>HITL checkpoint after data analysis - confirm data understanding.</summary>
    DataAnalysisReview = 11,

    /// <summary>Model recommendation phase - ModelArchitect suggests suitable models.</summary>
    ModelRecommendation = 20,

    /// <summary>HITL checkpoint after model recommendation - approve model selection.</summary>
    ModelSelectionReview = 21,

    /// <summary>Preprocessing phase - PreprocessingExpert/IncrementalPreprocessing handles data transformation.</summary>
    Preprocessing = 30,

    /// <summary>HITL checkpoint after preprocessing - approve transformation pipeline.</summary>
    PreprocessingReview = 31,

    /// <summary>Training phase - AutoML training with selected models.</summary>
    Training = 40,

    /// <summary>HITL checkpoint after training - review training results.</summary>
    TrainingReview = 41,

    /// <summary>Evaluation phase - comprehensive model evaluation.</summary>
    Evaluation = 50,

    /// <summary>HITL checkpoint after evaluation - approve deployment.</summary>
    DeploymentReview = 51,

    /// <summary>Deployment phase - MLOpsManager handles model deployment.</summary>
    Deployment = 60,

    /// <summary>Orchestration completed successfully.</summary>
    Completed = 100,

    /// <summary>Orchestration cancelled by user.</summary>
    Cancelled = 200,

    /// <summary>Orchestration failed with error.</summary>
    Failed = 201,

    /// <summary>Orchestration paused, waiting for HITL response.</summary>
    Paused = 202
}

/// <summary>
/// Extension methods for OrchestrationState.
/// </summary>
public static class OrchestrationStateExtensions
{
    /// <summary>
    /// Checks if the state is a terminal state (completed, cancelled, or failed).
    /// </summary>
    public static bool IsTerminal(this OrchestrationState state)
        => state is OrchestrationState.Completed
            or OrchestrationState.Cancelled
            or OrchestrationState.Failed;

    /// <summary>
    /// Checks if the state is a HITL checkpoint requiring user interaction.
    /// </summary>
    public static bool IsHitlCheckpoint(this OrchestrationState state)
        => state is OrchestrationState.DataAnalysisReview
            or OrchestrationState.ModelSelectionReview
            or OrchestrationState.PreprocessingReview
            or OrchestrationState.TrainingReview
            or OrchestrationState.DeploymentReview;

    /// <summary>
    /// Gets the display name for the state.
    /// </summary>
    public static string GetDisplayName(this OrchestrationState state) => state switch
    {
        OrchestrationState.NotStarted => "Not Started",
        OrchestrationState.Initializing => "Initializing",
        OrchestrationState.DataAnalysis => "Data Analysis",
        OrchestrationState.DataAnalysisReview => "Data Analysis Review",
        OrchestrationState.ModelRecommendation => "Model Recommendation",
        OrchestrationState.ModelSelectionReview => "Model Selection Review",
        OrchestrationState.Preprocessing => "Preprocessing",
        OrchestrationState.PreprocessingReview => "Preprocessing Review",
        OrchestrationState.Training => "Training",
        OrchestrationState.TrainingReview => "Training Review",
        OrchestrationState.Evaluation => "Evaluation",
        OrchestrationState.DeploymentReview => "Deployment Review",
        OrchestrationState.Deployment => "Deployment",
        OrchestrationState.Completed => "Completed",
        OrchestrationState.Cancelled => "Cancelled",
        OrchestrationState.Failed => "Failed",
        OrchestrationState.Paused => "Paused",
        _ => state.ToString()
    };

    /// <summary>
    /// Gets the next state in the workflow progression.
    /// </summary>
    public static OrchestrationState GetNextState(this OrchestrationState state) => state switch
    {
        OrchestrationState.NotStarted => OrchestrationState.Initializing,
        OrchestrationState.Initializing => OrchestrationState.DataAnalysis,
        OrchestrationState.DataAnalysis => OrchestrationState.DataAnalysisReview,
        OrchestrationState.DataAnalysisReview => OrchestrationState.ModelRecommendation,
        OrchestrationState.ModelRecommendation => OrchestrationState.ModelSelectionReview,
        OrchestrationState.ModelSelectionReview => OrchestrationState.Preprocessing,
        OrchestrationState.Preprocessing => OrchestrationState.PreprocessingReview,
        OrchestrationState.PreprocessingReview => OrchestrationState.Training,
        OrchestrationState.Training => OrchestrationState.TrainingReview,
        OrchestrationState.TrainingReview => OrchestrationState.Evaluation,
        OrchestrationState.Evaluation => OrchestrationState.DeploymentReview,
        OrchestrationState.DeploymentReview => OrchestrationState.Deployment,
        OrchestrationState.Deployment => OrchestrationState.Completed,
        _ => state // Terminal states return themselves
    };

    /// <summary>
    /// Gets the phase number (1-5) for progress tracking.
    /// </summary>
    public static int GetPhaseNumber(this OrchestrationState state) => state switch
    {
        OrchestrationState.DataAnalysis or OrchestrationState.DataAnalysisReview => 1,
        OrchestrationState.ModelRecommendation or OrchestrationState.ModelSelectionReview => 2,
        OrchestrationState.Preprocessing or OrchestrationState.PreprocessingReview => 3,
        OrchestrationState.Training or OrchestrationState.TrainingReview or OrchestrationState.Evaluation => 4,
        OrchestrationState.DeploymentReview or OrchestrationState.Deployment => 5,
        OrchestrationState.Completed => 5,
        _ => 0
    };

    /// <summary>
    /// Gets the progress percentage (0-100).
    /// </summary>
    public static int GetProgressPercentage(this OrchestrationState state) => state switch
    {
        OrchestrationState.NotStarted => 0,
        OrchestrationState.Initializing => 5,
        OrchestrationState.DataAnalysis => 10,
        OrchestrationState.DataAnalysisReview => 15,
        OrchestrationState.ModelRecommendation => 25,
        OrchestrationState.ModelSelectionReview => 30,
        OrchestrationState.Preprocessing => 40,
        OrchestrationState.PreprocessingReview => 50,
        OrchestrationState.Training => 60,
        OrchestrationState.TrainingReview => 75,
        OrchestrationState.Evaluation => 85,
        OrchestrationState.DeploymentReview => 90,
        OrchestrationState.Deployment => 95,
        OrchestrationState.Completed => 100,
        _ => 0
    };
}
