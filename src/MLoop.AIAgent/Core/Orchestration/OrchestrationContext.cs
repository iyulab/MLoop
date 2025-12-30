// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using System.Text.Json.Serialization;
using Ironbees.Core.Goals;

namespace MLoop.AIAgent.Core.Orchestration;

/// <summary>
/// Context maintained throughout the orchestration workflow.
/// Contains all data collected from agents and user decisions.
/// </summary>
public class OrchestrationContext
{
    /// <summary>Unique session identifier.</summary>
    public string SessionId { get; set; } = GenerateSessionId();

    /// <summary>Path to the input data file.</summary>
    public required string DataFilePath { get; set; }

    /// <summary>Current orchestration state.</summary>
    public OrchestrationState CurrentState { get; set; } = OrchestrationState.NotStarted;

    /// <summary>When orchestration started.</summary>
    public DateTimeOffset StartedAt { get; set; } = DateTimeOffset.UtcNow;

    /// <summary>When orchestration completed (if terminal).</summary>
    public DateTimeOffset? CompletedAt { get; set; }

    /// <summary>User-provided options.</summary>
    public OrchestrationOptions Options { get; set; } = new();

    /// <summary>ironbees AgenticSettings for HITL/Sampling/Confidence.</summary>
    public AgenticSettings? AgenticSettings { get; set; }

    // ========================================
    // Phase 1: Data Analysis Results
    // ========================================

    /// <summary>Results from DataAnalyst agent.</summary>
    public DataAnalysisResult? DataAnalysis { get; set; }

    // ========================================
    // Phase 2: Model Recommendation Results
    // ========================================

    /// <summary>Results from ModelArchitect agent.</summary>
    public ModelRecommendationResult? ModelRecommendation { get; set; }

    // ========================================
    // Phase 3: Preprocessing Results
    // ========================================

    /// <summary>Results from PreprocessingExpert/IncrementalPreprocessing agent.</summary>
    public PreprocessingResult? Preprocessing { get; set; }

    // ========================================
    // Phase 4: Training Results
    // ========================================

    /// <summary>Training results from AutoML execution.</summary>
    public TrainingResult? Training { get; set; }

    // ========================================
    // Phase 5: Evaluation & Deployment Results
    // ========================================

    /// <summary>Model evaluation results.</summary>
    public EvaluationResult? Evaluation { get; set; }

    /// <summary>Deployment results.</summary>
    public DeploymentResult? Deployment { get; set; }

    // ========================================
    // HITL Decisions
    // ========================================

    /// <summary>All HITL decisions made during orchestration.</summary>
    public List<HitlDecision> HitlDecisions { get; set; } = [];

    // ========================================
    // Artifacts
    // ========================================

    /// <summary>Generated artifacts with their paths.</summary>
    public Dictionary<string, string> Artifacts { get; set; } = [];

    // ========================================
    // Error Tracking
    // ========================================

    /// <summary>Errors encountered during orchestration.</summary>
    public List<OrchestrationError> Errors { get; set; } = [];

    /// <summary>Last error message if in failed state.</summary>
    public string? LastError { get; set; }

    // ========================================
    // Helpers
    // ========================================

    private static string GenerateSessionId()
        => $"orc-{DateTime.UtcNow:yyyyMMdd}-{Guid.NewGuid().ToString("N")[..6]}";

    /// <summary>Gets the current confidence level based on available results.</summary>
    public double GetCurrentConfidence()
    {
        return CurrentState switch
        {
            OrchestrationState.DataAnalysis or OrchestrationState.DataAnalysisReview
                => DataAnalysis?.Confidence ?? 0,
            OrchestrationState.ModelRecommendation or OrchestrationState.ModelSelectionReview
                => ModelRecommendation?.Confidence ?? 0,
            OrchestrationState.Preprocessing or OrchestrationState.PreprocessingReview
                => Preprocessing?.Confidence ?? 0,
            OrchestrationState.Training or OrchestrationState.TrainingReview
                => Training?.Confidence ?? 0,
            OrchestrationState.Evaluation or OrchestrationState.DeploymentReview
                => Evaluation?.Confidence ?? 0,
            _ => 0
        };
    }

    /// <summary>Gets total elapsed time.</summary>
    public TimeSpan GetElapsedTime()
        => (CompletedAt ?? DateTimeOffset.UtcNow) - StartedAt;
}

/// <summary>
/// Result from DataAnalyst agent.
/// </summary>
public record DataAnalysisResult
{
    /// <summary>Number of rows in dataset.</summary>
    public int RowCount { get; init; }

    /// <summary>Number of columns in dataset.</summary>
    public int ColumnCount { get; init; }

    /// <summary>Detected target column.</summary>
    public string? DetectedTargetColumn { get; init; }

    /// <summary>Inferred ML task type.</summary>
    public string? InferredTaskType { get; init; }

    /// <summary>Column information.</summary>
    public List<ColumnInfo> Columns { get; init; } = [];

    /// <summary>Data quality score (0-1).</summary>
    public double DataQualityScore { get; init; }

    /// <summary>Missing value percentage.</summary>
    public double MissingValuePercentage { get; init; }

    /// <summary>Detected issues.</summary>
    public List<string> Issues { get; init; } = [];

    /// <summary>Recommendations.</summary>
    public List<string> Recommendations { get; init; } = [];

    /// <summary>Confidence in analysis (0-1).</summary>
    public double Confidence { get; init; }

    /// <summary>Raw analysis text from LLM.</summary>
    public string? RawAnalysis { get; init; }
}

/// <summary>
/// Column information from data analysis.
/// </summary>
public record ColumnInfo
{
    public required string Name { get; init; }
    public required string DataType { get; init; }
    public int MissingCount { get; init; }
    public double MissingPercentage { get; init; }
    public int UniqueCount { get; init; }
    public bool IsCategorical { get; init; }
    public bool IsNumeric { get; init; }
    public bool IsTarget { get; init; }
    public string? Role { get; init; }
}

/// <summary>
/// Result from ModelArchitect agent.
/// </summary>
public record ModelRecommendationResult
{
    /// <summary>Recommended ML task type.</summary>
    public required string TaskType { get; init; }

    /// <summary>Recommended primary metric.</summary>
    public required string PrimaryMetric { get; init; }

    /// <summary>Recommended trainers/algorithms.</summary>
    public List<TrainerRecommendation> RecommendedTrainers { get; init; } = [];

    /// <summary>AutoML configuration.</summary>
    public AutoMLConfiguration? AutoMLConfig { get; init; }

    /// <summary>Rationale for recommendations.</summary>
    public string? Rationale { get; init; }

    /// <summary>Confidence in recommendation (0-1).</summary>
    public double Confidence { get; init; }
}

/// <summary>
/// Individual trainer recommendation.
/// </summary>
public record TrainerRecommendation
{
    public required string TrainerName { get; init; }
    public string? Reason { get; init; }
    public int Priority { get; init; }
    public Dictionary<string, object>? SuggestedParameters { get; init; }
}

/// <summary>
/// AutoML configuration.
/// </summary>
public record AutoMLConfiguration
{
    public int MaxTrainingTimeSeconds { get; init; } = 300;
    public int? MaxModels { get; init; }
    public List<string>? AllowedTrainers { get; init; }
    public List<string>? BlockedTrainers { get; init; }
    public int? Seed { get; init; }
}

/// <summary>
/// Result from preprocessing agent.
/// </summary>
public record PreprocessingResult
{
    /// <summary>Path to preprocessed data file.</summary>
    public string? OutputFilePath { get; init; }

    /// <summary>Preprocessing steps applied.</summary>
    public List<PreprocessingStep> Steps { get; init; } = [];

    /// <summary>Rows before preprocessing.</summary>
    public int RowsBefore { get; init; }

    /// <summary>Rows after preprocessing.</summary>
    public int RowsAfter { get; init; }

    /// <summary>Columns before preprocessing.</summary>
    public int ColumnsBefore { get; init; }

    /// <summary>Columns after preprocessing.</summary>
    public int ColumnsAfter { get; init; }

    /// <summary>Whether incremental preprocessing was used.</summary>
    public bool UsedIncrementalProcessing { get; init; }

    /// <summary>Confidence in preprocessing (0-1).</summary>
    public double Confidence { get; init; }

    /// <summary>Generated preprocessing script/pipeline.</summary>
    public string? PipelineDefinition { get; init; }
}

/// <summary>
/// Individual preprocessing step.
/// </summary>
public record PreprocessingStep
{
    public required string Name { get; init; }
    public required string Description { get; init; }
    public string? TargetColumn { get; init; }
    public Dictionary<string, object>? Parameters { get; init; }
}

/// <summary>
/// Training result from AutoML.
/// </summary>
public record TrainingResult
{
    /// <summary>Best model name/algorithm.</summary>
    public required string BestModelName { get; init; }

    /// <summary>Primary metric name.</summary>
    public required string PrimaryMetricName { get; init; }

    /// <summary>Primary metric value.</summary>
    public double PrimaryMetricValue { get; init; }

    /// <summary>All metrics for best model.</summary>
    public Dictionary<string, double> Metrics { get; init; } = [];

    /// <summary>Training duration.</summary>
    public TimeSpan TrainingDuration { get; init; }

    /// <summary>Number of models evaluated.</summary>
    public int ModelsEvaluated { get; init; }

    /// <summary>Path to saved model.</summary>
    public string? ModelPath { get; init; }

    /// <summary>Cross-validation results.</summary>
    public CrossValidationResult? CrossValidation { get; init; }

    /// <summary>Top N models evaluated.</summary>
    public List<ModelSummary> TopModels { get; init; } = [];

    /// <summary>Confidence in training result (0-1).</summary>
    public double Confidence { get; init; }

    /// <summary>MLoop experiment ID.</summary>
    public string? ExperimentId { get; init; }
}

/// <summary>
/// Cross-validation result.
/// </summary>
public record CrossValidationResult
{
    public int Folds { get; init; }
    public double[] Scores { get; init; } = [];
    public double Mean { get; init; }
    public double StandardDeviation { get; init; }
}

/// <summary>
/// Summary of an evaluated model.
/// </summary>
public record ModelSummary
{
    public required string ModelName { get; init; }
    public double Score { get; init; }
    public TimeSpan TrainingTime { get; init; }
    public int Rank { get; init; }
}

/// <summary>
/// Evaluation result.
/// </summary>
public record EvaluationResult
{
    /// <summary>Final metrics on test set.</summary>
    public Dictionary<string, double> TestMetrics { get; init; } = [];

    /// <summary>Confusion matrix for classification.</summary>
    public int[,]? ConfusionMatrix { get; init; }

    /// <summary>Feature importance if available.</summary>
    public Dictionary<string, double>? FeatureImportance { get; init; }

    /// <summary>Evaluation summary.</summary>
    public string? Summary { get; init; }

    /// <summary>Recommendations for improvement.</summary>
    public List<string> Recommendations { get; init; } = [];

    /// <summary>Confidence in evaluation (0-1).</summary>
    public double Confidence { get; init; }

    /// <summary>Path to evaluation report.</summary>
    public string? ReportPath { get; init; }
}

/// <summary>
/// Deployment result.
/// </summary>
public record DeploymentResult
{
    /// <summary>Whether deployment was successful.</summary>
    public bool Success { get; init; }

    /// <summary>Deployment target (e.g., local, API endpoint).</summary>
    public string? DeploymentTarget { get; init; }

    /// <summary>Model endpoint URL if applicable.</summary>
    public string? EndpointUrl { get; init; }

    /// <summary>Deployment timestamp.</summary>
    public DateTimeOffset DeployedAt { get; init; }

    /// <summary>Model version.</summary>
    public string? ModelVersion { get; init; }

    /// <summary>Additional deployment info.</summary>
    public Dictionary<string, string>? Metadata { get; init; }
}

/// <summary>
/// Record of a HITL decision.
/// </summary>
public record HitlDecision
{
    /// <summary>Checkpoint where decision was made.</summary>
    public required string CheckpointId { get; init; }

    /// <summary>State when decision was made.</summary>
    public OrchestrationState State { get; init; }

    /// <summary>Selected option ID.</summary>
    public required string SelectedOptionId { get; init; }

    /// <summary>Whether auto-approved.</summary>
    public bool IsAutoApproval { get; init; }

    /// <summary>Confidence at time of decision.</summary>
    public double Confidence { get; init; }

    /// <summary>User comment if any.</summary>
    public string? UserComment { get; init; }

    /// <summary>When decision was made.</summary>
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>Time taken to respond.</summary>
    public TimeSpan ResponseTime { get; init; }
}

/// <summary>
/// Error encountered during orchestration.
/// </summary>
public record OrchestrationError
{
    /// <summary>State when error occurred.</summary>
    public OrchestrationState State { get; init; }

    /// <summary>Error message.</summary>
    public required string Message { get; init; }

    /// <summary>Error details/stack trace.</summary>
    public string? Details { get; init; }

    /// <summary>When error occurred.</summary>
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>Whether error is recoverable.</summary>
    public bool IsRecoverable { get; init; }

    /// <summary>Agent that caused the error if applicable.</summary>
    public string? SourceAgent { get; init; }
}
