using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Resolves and manages model names within an MLoop project.
/// Handles the mapping between model names and their filesystem paths.
/// </summary>
public interface IModelNameResolver
{
    /// <summary>
    /// Resolves a model name, returning "default" if null or empty.
    /// </summary>
    /// <param name="name">The model name to resolve, or null for default</param>
    /// <returns>The resolved model name</returns>
    string Resolve(string? name);

    /// <summary>
    /// Checks if a model with the given name exists in the project.
    /// </summary>
    bool Exists(string name);

    /// <summary>
    /// Gets the root directory path for a model.
    /// </summary>
    /// <param name="name">The model name</param>
    /// <returns>Absolute path to the model's root directory (models/{name}/)</returns>
    string GetModelPath(string name);

    /// <summary>
    /// Lists all models in the project.
    /// </summary>
    Task<IEnumerable<ModelSummary>> ListAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Creates a new model with the given definition.
    /// </summary>
    Task CreateAsync(string name, ModelDefinition definition, CancellationToken cancellationToken = default);

    /// <summary>
    /// Removes a model and all its experiments.
    /// </summary>
    Task RemoveAsync(string name, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the model definition from configuration.
    /// Returns null if not defined in config (but may exist on filesystem).
    /// </summary>
    Task<ModelDefinition?> GetDefinitionAsync(string name, CancellationToken cancellationToken = default);

    /// <summary>
    /// Ensures the model directory structure exists.
    /// Creates staging/ and production/ subdirectories if needed.
    /// </summary>
    Task EnsureModelDirectoryAsync(string name, CancellationToken cancellationToken = default);

    /// <summary>
    /// Validates a model name according to naming rules.
    /// </summary>
    /// <param name="name">The model name to validate</param>
    /// <returns>True if valid, false otherwise</returns>
    bool IsValidName(string name);
}

/// <summary>
/// Summary information about a model
/// </summary>
public class ModelSummary
{
    /// <summary>
    /// Model name (e.g., "default", "churn-predictor")
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// ML task type (regression, binary-classification, etc.)
    /// </summary>
    public required string Task { get; init; }

    /// <summary>
    /// Label column name
    /// </summary>
    public required string Label { get; init; }

    /// <summary>
    /// When the model was first created
    /// </summary>
    public required DateTime CreatedAt { get; init; }

    /// <summary>
    /// Number of experiments for this model
    /// </summary>
    public int ExperimentCount { get; init; }

    /// <summary>
    /// Currently promoted production experiment ID (null if none)
    /// </summary>
    public string? ProductionExperiment { get; init; }

    /// <summary>
    /// Model description
    /// </summary>
    public string? Description { get; init; }
}

/// <summary>
/// Model index persisted to .mloop/models.json
/// </summary>
public class ModelIndex
{
    /// <summary>
    /// Map of model name to model metadata
    /// </summary>
    public Dictionary<string, ModelMetadata> Models { get; set; } = new();
}

/// <summary>
/// Metadata for a single model (stored in index)
/// </summary>
public class ModelMetadata
{
    public required DateTime CreatedAt { get; set; }
    public required string Task { get; set; }
    public required string Label { get; set; }
    public string? Description { get; set; }
}
