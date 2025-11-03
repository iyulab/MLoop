namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Manages experiment storage and ID generation
/// </summary>
public interface IExperimentStore
{
    /// <summary>
    /// Generates a unique experiment ID (atomic operation)
    /// </summary>
    Task<string> GenerateIdAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Saves experiment metadata and results
    /// </summary>
    Task SaveAsync(ExperimentData experiment, CancellationToken cancellationToken = default);

    /// <summary>
    /// Loads experiment metadata
    /// </summary>
    Task<ExperimentData> LoadAsync(string experimentId, CancellationToken cancellationToken = default);

    /// <summary>
    /// Lists all experiments
    /// </summary>
    Task<IEnumerable<ExperimentSummary>> ListAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the experiment directory path
    /// </summary>
    string GetExperimentPath(string experimentId);

    /// <summary>
    /// Checks if an experiment exists
    /// </summary>
    bool ExperimentExists(string experimentId);
}

/// <summary>
/// Full experiment data with metadata, metrics, and configuration
/// </summary>
public class ExperimentData
{
    public required string ExperimentId { get; init; }
    public required DateTime Timestamp { get; init; }
    public required string Status { get; init; }
    public required string Task { get; init; }
    public required ExperimentConfig Config { get; init; }
    public ExperimentResult? Result { get; init; }
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
    public required string ExperimentId { get; init; }
    public required DateTime Timestamp { get; init; }
    public required string Status { get; init; }
    public double? BestMetric { get; init; }
}
