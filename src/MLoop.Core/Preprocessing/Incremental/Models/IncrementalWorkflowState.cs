using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Represents the complete state of an incremental preprocessing workflow.
/// Can be serialized for checkpointing and resume capability.
/// </summary>
public sealed class IncrementalWorkflowState
{
    /// <summary>
    /// Unique identifier for this workflow session.
    /// </summary>
    public required string SessionId { get; init; }

    /// <summary>
    /// Current stage of the workflow.
    /// </summary>
    public required WorkflowStage CurrentStage { get; set; }

    /// <summary>
    /// Path to the dataset being processed.
    /// </summary>
    public required string DatasetPath { get; init; }

    /// <summary>
    /// Total number of records in the full dataset.
    /// </summary>
    public required long TotalRecords { get; init; }

    /// <summary>
    /// Results from completed workflow stages.
    /// Key: WorkflowStage, Value: StageResult
    /// </summary>
    public required Dictionary<WorkflowStage, StageResult> CompletedStages { get; init; }

    /// <summary>
    /// All preprocessing rules discovered across all stages.
    /// </summary>
    public required List<PreprocessingRule> DiscoveredRules { get; init; }

    /// <summary>
    /// Rules approved by user or auto-approved at checkpoint.
    /// These will be applied during bulk processing.
    /// </summary>
    public required List<PreprocessingRule> ApprovedRules { get; init; }

    /// <summary>
    /// Overall confidence score for the discovered rules (0.0 to 1.0).
    /// Based on rule stability across samples.
    /// </summary>
    public required double ConfidenceScore { get; set; }

    /// <summary>
    /// Whether rule discovery has converged (no new patterns found).
    /// </summary>
    public bool HasConverged { get; set; }

    /// <summary>
    /// When this workflow was started.
    /// </summary>
    public DateTime StartedAt { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// When this workflow was completed (null if still in progress).
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Configuration used for this workflow.
    /// </summary>
    public required IncrementalWorkflowConfig Config { get; init; }

    /// <summary>
    /// Optional user notes or observations.
    /// </summary>
    public string? Notes { get; set; }

    /// <summary>
    /// Total duration of the workflow (calculated).
    /// </summary>
    public TimeSpan TotalDuration =>
        CompletedAt.HasValue
            ? CompletedAt.Value - StartedAt
            : DateTime.UtcNow - StartedAt;

    /// <summary>
    /// Gets the result for a specific stage if completed.
    /// </summary>
    public StageResult? GetStageResult(WorkflowStage stage)
    {
        return CompletedStages.TryGetValue(stage, out var result) ? result : null;
    }

    /// <summary>
    /// Checks if a stage has been completed.
    /// </summary>
    public bool IsStageCompleted(WorkflowStage stage)
    {
        return CompletedStages.ContainsKey(stage);
    }
}
