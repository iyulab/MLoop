using MLoop.Core.Preprocessing.Incremental.Models;

namespace MLoop.Core.Preprocessing.Incremental.Contracts;

/// <summary>
/// Orchestrates the complete 5-stage incremental preprocessing workflow.
/// </summary>
public interface IWorkflowOrchestrator
{
    /// <summary>
    /// Starts a new incremental preprocessing workflow.
    /// </summary>
    /// <param name="datasetPath">Path to the dataset CSV file.</param>
    /// <param name="config">Workflow configuration.</param>
    /// <param name="progress">Optional progress reporting callback.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The completed workflow state with all results.</returns>
    Task<IncrementalWorkflowState> ExecuteWorkflowAsync(
        string datasetPath,
        IncrementalWorkflowConfig config,
        IProgress<WorkflowProgress>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Resumes a workflow from a saved checkpoint.
    /// </summary>
    /// <param name="checkpointPath">Path to the checkpoint file.</param>
    /// <param name="progress">Optional progress reporting callback.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The completed workflow state.</returns>
    Task<IncrementalWorkflowState> ResumeWorkflowAsync(
        string checkpointPath,
        IProgress<WorkflowProgress>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Saves the current workflow state as a checkpoint.
    /// </summary>
    /// <param name="state">Workflow state to save.</param>
    /// <param name="checkpointPath">Optional custom checkpoint path.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task SaveCheckpointAsync(
        IncrementalWorkflowState state,
        string? checkpointPath = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Loads a workflow state from a checkpoint file.
    /// </summary>
    /// <param name="checkpointPath">Path to the checkpoint file.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The loaded workflow state.</returns>
    Task<IncrementalWorkflowState> LoadCheckpointAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Progress information for workflow execution.
/// </summary>
public sealed class WorkflowProgress
{
    /// <summary>
    /// Current workflow stage.
    /// </summary>
    public required WorkflowStage Stage { get; init; }

    /// <summary>
    /// Progress percentage (0.0 to 1.0).
    /// </summary>
    public required double Percentage { get; init; }

    /// <summary>
    /// Current status message.
    /// </summary>
    public required string Message { get; init; }

    /// <summary>
    /// Number of rules discovered so far.
    /// </summary>
    public int RulesDiscovered { get; init; }

    /// <summary>
    /// Current confidence score (0.0 to 1.0).
    /// </summary>
    public double ConfidenceScore { get; init; }

    /// <summary>
    /// Whether rule discovery has converged.
    /// </summary>
    public bool HasConverged { get; init; }
}
