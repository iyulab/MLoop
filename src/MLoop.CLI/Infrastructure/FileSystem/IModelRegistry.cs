namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Manages model promotion to production for multi-model projects.
/// Each model has its own production slot at models/{modelName}/production/.
/// </summary>
public interface IModelRegistry
{
    /// <summary>
    /// Promotes an experiment model to production for a specific model.
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <param name="experimentId">The experiment ID to promote</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task PromoteAsync(
        string modelName,
        string experimentId,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the current production model for a specific model name.
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Model info or null if no production model</returns>
    Task<ModelInfo?> GetProductionAsync(
        string modelName,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Lists all production models across all model names.
    /// </summary>
    /// <param name="modelName">Optional model name filter, null for all models</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task<IEnumerable<ModelInfo>> ListAsync(
        string? modelName = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the production model directory path for a model.
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <returns>Absolute path to production directory (models/{modelName}/production/)</returns>
    string GetProductionPath(string modelName);

    /// <summary>
    /// Checks if a new experiment should be promoted to production.
    /// Compares against the current production model for the same model name.
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <param name="experimentId">The experiment ID to evaluate</param>
    /// <param name="primaryMetric">The metric to compare</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task<bool> ShouldPromoteAsync(
        string modelName,
        string experimentId,
        string primaryMetric,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Automatically promotes experiment to production if it performs better
    /// than the current production model for the same model name.
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <param name="experimentId">The experiment ID to potentially promote</param>
    /// <param name="primaryMetric">The metric to compare</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>True if promoted, false otherwise</returns>
    Task<bool> AutoPromoteAsync(
        string modelName,
        string experimentId,
        string primaryMetric,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if a production model exists for the given model name.
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <returns>True if production model exists</returns>
    bool HasProduction(string modelName);

    /// <summary>
    /// Gets the path to the model.zip file for production.
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <returns>Absolute path to model.zip in production</returns>
    string GetProductionModelFile(string modelName);
}

/// <summary>
/// Production model information
/// </summary>
public class ModelInfo
{
    /// <summary>
    /// Model name this production model belongs to
    /// </summary>
    public required string ModelName { get; init; }

    /// <summary>
    /// Experiment ID that was promoted
    /// </summary>
    public required string ExperimentId { get; init; }

    /// <summary>
    /// When the model was promoted to production
    /// </summary>
    public required DateTime PromotedAt { get; init; }

    /// <summary>
    /// Metrics at time of promotion
    /// </summary>
    public Dictionary<string, double>? Metrics { get; init; }

    /// <summary>
    /// Absolute path to the production model directory
    /// </summary>
    public required string ModelPath { get; init; }

    /// <summary>
    /// ML task type (binary-classification, regression, etc.)
    /// </summary>
    public string? Task { get; init; }

    /// <summary>
    /// Best trainer algorithm used
    /// </summary>
    public string? BestTrainer { get; init; }
}
