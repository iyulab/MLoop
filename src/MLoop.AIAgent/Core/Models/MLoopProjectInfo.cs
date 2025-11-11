namespace MLoop.AIAgent.Core.Models;

/// <summary>
/// MLoop project initialization configuration
/// </summary>
public class MLoopProjectConfig
{
    /// <summary>
    /// Project name
    /// </summary>
    public required string ProjectName { get; init; }

    /// <summary>
    /// Path to training data CSV file
    /// </summary>
    public required string DataPath { get; init; }

    /// <summary>
    /// Target column name for ML training
    /// </summary>
    public required string LabelColumn { get; init; }

    /// <summary>
    /// ML problem type (binary-classification, multiclass-classification, regression)
    /// </summary>
    public required string TaskType { get; init; }

    /// <summary>
    /// Project directory path (defaults to current directory)
    /// </summary>
    public string? ProjectDirectory { get; init; }
}

/// <summary>
/// MLoop training configuration
/// </summary>
public class MLoopTrainingConfig
{
    /// <summary>
    /// Training time budget in seconds
    /// </summary>
    public required int TimeSeconds { get; init; }

    /// <summary>
    /// Optimization metric (e.g., Accuracy, F1Score, RSquared)
    /// </summary>
    public required string Metric { get; init; }

    /// <summary>
    /// Test data split ratio (0.0-1.0)
    /// </summary>
    public double TestSplit { get; init; } = 0.2;

    /// <summary>
    /// Path to training data (optional, uses project data if not specified)
    /// </summary>
    public string? DataPath { get; init; }

    /// <summary>
    /// Experiment name (optional)
    /// </summary>
    public string? ExperimentName { get; init; }
}

/// <summary>
/// MLoop experiment information
/// </summary>
public class MLoopExperiment
{
    /// <summary>
    /// Experiment unique identifier
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Experiment name
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Training timestamp
    /// </summary>
    public required DateTime Timestamp { get; init; }

    /// <summary>
    /// Trainer algorithm used
    /// </summary>
    public required string Trainer { get; init; }

    /// <summary>
    /// Primary metric value
    /// </summary>
    public required double MetricValue { get; init; }

    /// <summary>
    /// Metric name
    /// </summary>
    public required string MetricName { get; init; }

    /// <summary>
    /// Is this experiment promoted to production?
    /// </summary>
    public bool IsProduction { get; init; }
}

/// <summary>
/// Result of MLoop operation
/// </summary>
public class MLoopOperationResult
{
    /// <summary>
    /// Was the operation successful?
    /// </summary>
    public required bool Success { get; init; }

    /// <summary>
    /// Exit code from CLI command
    /// </summary>
    public required int ExitCode { get; init; }

    /// <summary>
    /// Standard output from command
    /// </summary>
    public required string Output { get; init; }

    /// <summary>
    /// Standard error from command (if any)
    /// </summary>
    public string? Error { get; init; }

    /// <summary>
    /// Parsed data from output (e.g., experiment ID, metric values)
    /// </summary>
    public Dictionary<string, object> Data { get; init; } = new();
}
