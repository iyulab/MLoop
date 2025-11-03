namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Manages model promotion to staging and production
/// </summary>
public interface IModelRegistry
{
    /// <summary>
    /// Promotes an experiment model to staging or production
    /// </summary>
    Task PromoteAsync(
        string experimentId,
        ModelStage stage,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the current model for a stage
    /// </summary>
    Task<ModelInfo?> GetAsync(
        ModelStage stage,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Lists all promoted models
    /// </summary>
    Task<IEnumerable<ModelInfo>> ListAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the model directory path for a stage
    /// </summary>
    string GetModelPath(ModelStage stage);

    /// <summary>
    /// Checks if a new experiment should be promoted to production
    /// </summary>
    Task<bool> ShouldPromoteToProductionAsync(
        string experimentId,
        string primaryMetric,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Automatically promotes experiment to production if it performs better
    /// </summary>
    Task<bool> AutoPromoteAsync(
        string experimentId,
        string primaryMetric,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Model deployment stages
/// </summary>
public enum ModelStage
{
    Staging,
    Production
}

/// <summary>
/// Model information
/// </summary>
public class ModelInfo
{
    public required ModelStage Stage { get; init; }
    public required string ExperimentId { get; init; }
    public required DateTime PromotedAt { get; init; }
    public Dictionary<string, double>? Metrics { get; init; }
    public required string ModelPath { get; init; }
}
