namespace MLoop.Core.Models;

/// <summary>
/// Configuration for model training
/// </summary>
public class TrainingConfig
{
    /// <summary>
    /// Model name for multi-model support (defaults to "default")
    /// </summary>
    public required string ModelName { get; init; }

    /// <summary>
    /// Path to training data file
    /// </summary>
    public required string DataFile { get; init; }

    /// <summary>
    /// Label column name in the data
    /// </summary>
    public required string LabelColumn { get; init; }

    /// <summary>
    /// ML task type (binary-classification, multiclass-classification, regression)
    /// </summary>
    public required string Task { get; init; }

    /// <summary>
    /// Training time limit in seconds
    /// </summary>
    public int TimeLimitSeconds { get; init; } = 300;

    /// <summary>
    /// Optimization metric
    /// </summary>
    public string Metric { get; init; } = "accuracy";

    /// <summary>
    /// Test data split ratio
    /// </summary>
    public double TestSplit { get; init; } = 0.2;

    /// <summary>
    /// Path to separate test data file (when pre-split, e.g. for balanced training)
    /// </summary>
    public string? TestDataFile { get; init; }

    /// <summary>
    /// Whether to use automatic time estimation (true when --time not specified and YAML has no time_limit_seconds)
    /// </summary>
    public bool UseAutoTime { get; init; } = false;
}
