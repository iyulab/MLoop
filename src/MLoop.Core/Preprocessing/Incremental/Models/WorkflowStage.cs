namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Represents the current stage in the incremental preprocessing workflow.
/// </summary>
public enum WorkflowStage
{
    /// <summary>
    /// Workflow has not started yet.
    /// </summary>
    NotStarted = 0,

    /// <summary>
    /// Stage 1: Initial exploration with 0.1% sample.
    /// Discover schema and obvious patterns.
    /// </summary>
    InitialExploration = 1,

    /// <summary>
    /// Stage 2: Pattern expansion with 0.5% sample.
    /// Validate patterns and apply auto-fixable rules.
    /// </summary>
    PatternExpansion = 2,

    /// <summary>
    /// Stage 3: HITL decision with 1.5% sample.
    /// Get human decisions for business logic.
    /// </summary>
    HITLDecision = 3,

    /// <summary>
    /// Stage 4: Confidence checkpoint with 2.5% sample.
    /// Validate rule stability and get final approval.
    /// </summary>
    ConfidenceCheckpoint = 4,

    /// <summary>
    /// Stage 5: Bulk processing of remaining data (100%).
    /// Apply all approved rules to full dataset.
    /// </summary>
    BulkProcessing = 5,

    /// <summary>
    /// Workflow completed successfully.
    /// </summary>
    Completed = 6
}
