namespace MLoop.Core.Pipeline;

/// <summary>
/// Interface for pipeline step execution handlers
/// </summary>
public interface IPipelineStepHandler
{
    /// <summary>
    /// Step type this handler supports (e.g., "train", "preprocess")
    /// </summary>
    string StepType { get; }

    /// <summary>
    /// Execute the pipeline step with the given parameters
    /// </summary>
    Task<Dictionary<string, object>> ExecuteAsync(
        Dictionary<string, object> parameters,
        Action<string>? logger,
        CancellationToken cancellationToken);
}
