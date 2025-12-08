namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// MLoop project configuration supporting multiple models
/// </summary>
public class MLoopConfig
{
    /// <summary>
    /// Project name
    /// </summary>
    public string? Project { get; set; }

    /// <summary>
    /// Model definitions keyed by model name.
    /// "default" is used when --name is omitted from CLI commands.
    /// </summary>
    public Dictionary<string, ModelDefinition> Models { get; set; } = new();

    /// <summary>
    /// Shared data path settings
    /// </summary>
    public DataSettings? Data { get; set; }
}

/// <summary>
/// Definition of a single ML model within the project
/// </summary>
public class ModelDefinition
{
    /// <summary>
    /// ML task type: regression, binary-classification, multiclass-classification
    /// </summary>
    public required string Task { get; set; }

    /// <summary>
    /// Label column name in the dataset
    /// </summary>
    public required string Label { get; set; }

    /// <summary>
    /// Model-specific training settings (overrides project defaults)
    /// </summary>
    public TrainingSettings? Training { get; set; }

    /// <summary>
    /// Model description for documentation purposes
    /// </summary>
    public string? Description { get; set; }
}

/// <summary>
/// Training configuration settings
/// </summary>
public class TrainingSettings
{
    /// <summary>
    /// Maximum training time in seconds (default: 300)
    /// </summary>
    public int? TimeLimitSeconds { get; set; }

    /// <summary>
    /// Optimization metric (e.g., accuracy, auc, r2, rmse)
    /// </summary>
    public string? Metric { get; set; }

    /// <summary>
    /// Fraction of data to use for testing (default: 0.2)
    /// </summary>
    public double? TestSplit { get; set; }
}

/// <summary>
/// Data path settings
/// </summary>
public class DataSettings
{
    /// <summary>
    /// Path to training data file
    /// </summary>
    public string? Train { get; set; }

    /// <summary>
    /// Path to test data file
    /// </summary>
    public string? Test { get; set; }

    /// <summary>
    /// Path to prediction input file
    /// </summary>
    public string? Predict { get; set; }
}

/// <summary>
/// Default values and constants for configuration
/// </summary>
public static class ConfigDefaults
{
    public const string DefaultModelName = "default";
    public const int DefaultTimeLimitSeconds = 300;
    public const double DefaultTestSplit = 0.2;
    public const string DefaultMetric = "auto";

    /// <summary>
    /// Creates default training settings
    /// </summary>
    public static TrainingSettings CreateDefaultTrainingSettings() => new()
    {
        TimeLimitSeconds = DefaultTimeLimitSeconds,
        Metric = DefaultMetric,
        TestSplit = DefaultTestSplit
    };
}
