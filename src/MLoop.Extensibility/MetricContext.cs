using Microsoft.ML;

namespace MLoop.Extensibility;

/// <summary>
/// Provides context and data access for custom metric calculation.
/// Contains predictions and ML context needed to evaluate model performance.
/// </summary>
public class MetricContext
{
    /// <summary>
    /// Gets the ML.NET context for ML operations.
    /// </summary>
    public required MLContext MLContext { get; init; }

    /// <summary>
    /// Gets the predictions data view to evaluate.
    /// Contains both actual labels and predicted scores/labels.
    /// </summary>
    public required IDataView Predictions { get; init; }

    /// <summary>
    /// Gets the name of the label column in the predictions data.
    /// </summary>
    public required string LabelColumn { get; init; }

    /// <summary>
    /// Gets the name of the score column in the predictions data.
    /// For binary classification, this is typically "Score".
    /// For multiclass, this might be "PredictedLabel" or "Score".
    /// </summary>
    public required string ScoreColumn { get; init; }

    /// <summary>
    /// Gets the logger for outputting metric calculation details.
    /// Use this to provide transparency about how the metric is calculated.
    /// </summary>
    public required ILogger Logger { get; init; }

    /// <summary>
    /// Gets optional metadata about the evaluation context.
    /// May include experiment ID, model information, etc.
    /// </summary>
    public IReadOnlyDictionary<string, object> Metadata { get; init; } =
        new Dictionary<string, object>();
}
