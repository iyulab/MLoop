using MLoop.Core.Models;

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

    /// <summary>
    /// Every completed trial of the search, in completion order. Empty for the task paths that fit a
    /// single pipeline instead of searching — nothing is written in that case.
    /// </summary>
    public IReadOnlyList<TrialRecord> Trials { get; init; } = [];

    /// <summary>
    /// Canonical name of the metric the search ranked trials by — what the leaderboard is sorted on.
    /// Can differ from <see cref="ExperimentConfig.Metric"/>: binary classification switches to
    /// <c>f1_score</c> when AUC turns out to be undefined.
    /// </summary>
    public string? RankingMetric { get; init; }
}

/// <summary>
/// Experiment configuration
/// </summary>
public class ExperimentConfig
{
    public required string DataFile { get; init; }

    /// <summary>
    /// The SHA-256 of <see cref="DataFile"/>'s content as the user had it
    /// (<c>sha256:&lt;hex&gt;</c>, see <c>MLoop.Core.Data.DataFingerprint</c>), so the experiment
    /// can say which data it saw rather than only where it looked. Null on experiments recorded
    /// before the field existed and on directory-based tasks, whose input is not one file.
    /// </summary>
    public string? DataFileHash { get; init; }

    public required string LabelColumn { get; init; }
    /// <summary>
    /// The training budget, in seconds, the run was granted. For an auto-time run
    /// (<see cref="AutoTime"/>) this is the sum of the budgets it chose — the probe, plus the main
    /// pass when one ran — not the configured limit auto-time replaced.
    /// </summary>
    public required int TimeLimitSeconds { get; init; }

    /// <summary>
    /// True when auto-time chose the budget. Null for a fixed budget and on experiments recorded
    /// before the field existed, which stored the configured limit even when auto-time ran.
    /// </summary>
    public bool? AutoTime { get; init; }

    public required string Metric { get; init; }
    public required double TestSplit { get; init; }
    public InputSchemaInfo? InputSchema { get; init; }
    public string? GroupColumn { get; init; }
    // Recommendation user/item columns — persisted so predict can keep them individually
    // addressable for the model's key transforms (the same as GroupColumn for ranking).
    public string? UserColumn { get; init; }
    public string? ItemColumn { get; init; }
}

/// <summary>
/// Experiment result
/// </summary>
public class ExperimentResult
{
    public required string BestTrainer { get; init; }

    /// <summary>
    /// The same trainer in parts — name, hyperparameters, fallback reason — so a reader
    /// (<c>list</c>, <c>compare</c>, the API) gets an identifier without parsing
    /// <see cref="BestTrainer"/>'s display form. Null on experiments written before this field
    /// existed, and on the failure record where no trainer ran.
    /// </summary>
    public TrainerDescriptor? Trainer { get; init; }

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

    /// <summary>
    /// Best trainer/algorithm selected by AutoML
    /// </summary>
    public string? BestTrainer { get; init; }

    /// <summary>
    /// The same trainer in parts, so a listing can show the trainer without the transforms in front
    /// of it, and name the reason a fallback ran, without taking <see cref="BestTrainer"/> apart.
    /// Null on experiments recorded before the descriptor existed.
    /// </summary>
    public TrainerDescriptor? Trainer { get; init; }

    /// <summary>
    /// Optimization metric name (e.g., "MacroAccuracy", "RSquared")
    /// </summary>
    public string? MetricName { get; init; }

    /// <summary>
    /// Training duration in seconds
    /// </summary>
    public double? TrainingTimeSeconds { get; init; }
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
