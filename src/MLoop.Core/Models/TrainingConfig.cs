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

    /// <summary>
    /// Column type overrides from mloop.yaml.
    /// Key: column name, Value: type string (text, categorical, numeric, ignore)
    /// </summary>
    public Dictionary<string, string>? ColumnOverrides { get; init; }

    /// <summary>
    /// Number of clusters for clustering task (0 = auto-select via silhouette search)
    /// </summary>
    public int NumClusters { get; init; } = 0;

    /// <summary>
    /// Group column name for ranking task (groups rows into query contexts)
    /// </summary>
    public string? GroupColumn { get; init; }

    /// <summary>
    /// Forecast horizon — number of future time steps to predict (forecasting task)
    /// </summary>
    public int Horizon { get; init; } = 0;

    /// <summary>
    /// SSA window size for time series decomposition (0 = auto: series_length / 4)
    /// </summary>
    public int WindowSize { get; init; } = 0;

    /// <summary>
    /// Number of past data points to consider in SSA model (0 = auto: total rows)
    /// </summary>
    public int SeriesLength { get; init; } = 0;

    /// <summary>
    /// User column name for recommendation task
    /// </summary>
    public string? UserColumn { get; init; }

    /// <summary>
    /// Item column name for recommendation task
    /// </summary>
    public string? ItemColumn { get; init; }
}
