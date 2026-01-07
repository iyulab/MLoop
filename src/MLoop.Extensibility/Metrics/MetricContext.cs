using Microsoft.ML;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Extensibility.Metrics;

/// <summary>
/// Execution context for custom business metric calculation.
/// Provides model predictions, ground truth labels, and metadata.
/// </summary>
public class MetricContext
{
    /// <summary>
    /// ML.NET context for accessing ML operations.
    /// </summary>
    public required MLContext MLContext { get; init; }

    /// <summary>
    /// Model predictions for the evaluation dataset.
    /// For binary classification: bool[] (true/false predictions)
    /// For multiclass: int[] (predicted class indices)
    /// For regression: float[] (predicted values)
    /// </summary>
    public required Array Predictions { get; init; }

    /// <summary>
    /// Ground truth labels from the evaluation dataset.
    /// Same format as Predictions based on task type.
    /// </summary>
    public required Array Labels { get; init; }

    /// <summary>
    /// ML task type: BinaryClassification, MulticlassClassification, Regression
    /// </summary>
    public required string TaskType { get; init; }

    /// <summary>
    /// Model name for identification and logging.
    /// </summary>
    public string? ModelName { get; init; }

    /// <summary>
    /// Experiment ID if available.
    /// </summary>
    public string? ExperimentId { get; init; }

    /// <summary>
    /// Logger for progress reporting and debugging.
    /// </summary>
    public required ILogger Logger { get; init; }

    /// <summary>
    /// Additional metadata from configuration or pipeline context.
    /// </summary>
    private readonly Dictionary<string, object> _metadata = new();

    /// <summary>
    /// Sets metadata value by key.
    /// </summary>
    public void SetMetadata<T>(string key, T value) where T : notnull
    {
        _metadata[key] = value;
    }

    /// <summary>
    /// Gets metadata value by key with optional default.
    /// </summary>
    public T GetMetadata<T>(string key, T defaultValue = default!) where T : notnull
    {
        if (_metadata.TryGetValue(key, out var value) && value is T typedValue)
        {
            return typedValue;
        }
        return defaultValue;
    }
}
