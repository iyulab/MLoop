namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Manages experiment storage and ID generation for multi-model projects.
/// Each model has its own experiment namespace and index.
/// </summary>
public interface IExperimentStore
{
    /// <summary>
    /// Generates a unique experiment ID for a model (atomic operation).
    /// IDs are unique within a model namespace (e.g., default/exp-001, churn/exp-001).
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Generated experiment ID (e.g., "exp-001")</returns>
    Task<string> GenerateIdAsync(string modelName, CancellationToken cancellationToken = default);

    /// <summary>
    /// Saves experiment metadata and results.
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <param name="experiment">Experiment data to save</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task SaveAsync(string modelName, ExperimentData experiment, CancellationToken cancellationToken = default);

    /// <summary>
    /// Loads experiment metadata.
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <param name="experimentId">The experiment ID</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Experiment data</returns>
    Task<ExperimentData> LoadAsync(string modelName, string experimentId, CancellationToken cancellationToken = default);

    /// <summary>
    /// Lists experiments. If modelName is null, lists across all models.
    /// </summary>
    /// <param name="modelName">Model name to filter by, or null for all models</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Experiment summaries</returns>
    Task<IEnumerable<ExperimentSummary>> ListAsync(string? modelName = null, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the experiment directory path.
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <param name="experimentId">The experiment ID</param>
    /// <returns>Absolute path to experiment directory</returns>
    string GetExperimentPath(string modelName, string experimentId);

    /// <summary>
    /// Checks if an experiment exists.
    /// </summary>
    /// <param name="modelName">The model name</param>
    /// <param name="experimentId">The experiment ID</param>
    /// <returns>True if experiment exists</returns>
    bool ExperimentExists(string modelName, string experimentId);
}

/// <summary>
/// Full experiment data with metadata, metrics, and configuration
/// </summary>
public class ExperimentData
{
    /// <summary>
    /// Model name this experiment belongs to
    /// </summary>
    public required string ModelName { get; init; }

    /// <summary>
    /// Unique experiment ID within the model namespace
    /// </summary>
    public required string ExperimentId { get; init; }

    /// <summary>
    /// When the experiment was created
    /// </summary>
    public required DateTime Timestamp { get; init; }

    /// <summary>
    /// Experiment status: Running, Completed, Failed
    /// </summary>
    public required string Status { get; init; }

    /// <summary>
    /// ML task type
    /// </summary>
    public required string Task { get; init; }

    /// <summary>
    /// Experiment configuration
    /// </summary>
    public required ExperimentConfig Config { get; init; }

    /// <summary>
    /// Training result (null if not completed)
    /// </summary>
    public ExperimentResult? Result { get; init; }

    /// <summary>
    /// Evaluation metrics (null if not available)
    /// </summary>
    public Dictionary<string, double>? Metrics { get; init; }
}

/// <summary>
/// Experiment configuration
/// </summary>
public class ExperimentConfig
{
    public required string DataFile { get; init; }
    public required string LabelColumn { get; init; }
    public required int TimeLimitSeconds { get; init; }
    public required string Metric { get; init; }
    public required double TestSplit { get; init; }
    public InputSchemaInfo? InputSchema { get; init; }
}

/// <summary>
/// Input schema information captured during training
/// </summary>
public class InputSchemaInfo
{
    public required List<ColumnSchema> Columns { get; init; }
    public required DateTime CapturedAt { get; init; }
}

/// <summary>
/// Individual column schema information
/// </summary>
public class ColumnSchema
{
    public required string Name { get; init; }
    public required string DataType { get; init; }
    public required string Purpose { get; init; } // "Label", "Feature", "Ignore"

    /// <summary>
    /// For categorical columns: list of all unique values seen during training
    /// This is critical for preventing dimension mismatch during prediction
    /// </summary>
    public List<string>? CategoricalValues { get; init; }

    /// <summary>
    /// Total number of unique values (for validation)
    /// </summary>
    public int? UniqueValueCount { get; init; }
}

/// <summary>
/// Experiment result
/// </summary>
public class ExperimentResult
{
    public required string BestTrainer { get; init; }
    public required double TrainingTimeSeconds { get; init; }
}

/// <summary>
/// Summary information for experiment listing
/// </summary>
public class ExperimentSummary
{
    /// <summary>
    /// Model name this experiment belongs to
    /// </summary>
    public required string ModelName { get; init; }

    /// <summary>
    /// Experiment ID
    /// </summary>
    public required string ExperimentId { get; init; }

    /// <summary>
    /// When the experiment was created
    /// </summary>
    public required DateTime Timestamp { get; init; }

    /// <summary>
    /// Experiment status
    /// </summary>
    public required string Status { get; init; }

    /// <summary>
    /// Best metric value achieved
    /// </summary>
    public double? BestMetric { get; init; }

    /// <summary>
    /// Label column used in training
    /// </summary>
    public string? LabelColumn { get; init; }
}

/// <summary>
/// Per-model experiment index stored in models/{name}/experiment-index.json
/// </summary>
public class ExperimentIndex
{
    /// <summary>
    /// Next experiment ID number
    /// </summary>
    public int NextId { get; set; } = 1;

    /// <summary>
    /// List of experiment summaries
    /// </summary>
    public List<ExperimentSummary> Experiments { get; set; } = new();
}
