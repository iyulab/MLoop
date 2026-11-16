using Microsoft.ML;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Extensibility.Hooks;

/// <summary>
/// Execution context for hook scripts providing access to training data, configuration, and metadata.
/// </summary>
public class HookContext
{
    /// <summary>
    /// Type of hook being executed (PreTrain, PostTrain, PrePredict, PostEvaluate).
    /// </summary>
    public required HookType HookType { get; init; }

    /// <summary>
    /// Name of the hook script (e.g., "pre-train-validation.cs").
    /// Useful for logging and debugging.
    /// </summary>
    public required string HookName { get; init; }

    /// <summary>
    /// ML.NET context for data operations and transformations.
    /// </summary>
    public required MLContext MLContext { get; init; }

    /// <summary>
    /// Training data as ML.NET IDataView, when the loader can read it.
    /// Available for: PreTrain
    /// Null for: the other stages (PrePredict and PostEvaluate read <c>Metadata["DataFile"]</c>; PostTrain uses Model and ExperimentResult)
    /// </summary>
    public IDataView? DataView { get; init; }

    /// <summary>
    /// Trained ML model.
    /// Available for: PostTrain
    /// Null for: the other stages (<c>Metadata["ModelPath"]</c> names the model file)
    /// </summary>
    public ITransformer? Model { get; init; }

    /// <summary>
    /// The training run's result — its trials, metrics and the search it ran (<c>MLoop.Core.Training.TrainedModel</c>).
    /// Available for: PostTrain
    /// Null for: Other hook types
    /// </summary>
    public object? ExperimentResult { get; init; }

    /// <summary>
    /// Model evaluation metrics.
    /// Available for: PostTrain, PostEvaluate (an <c>IReadOnlyDictionary&lt;string, double&gt;</c>, name → value)
    /// Null for: PreTrain, PrePredict
    /// </summary>
    public object? Metrics { get; init; }  // Typed metrics (BinaryClassificationMetrics, etc.)

    /// <summary>
    /// Project root directory (location of `.mloop/` folder).
    /// Use for accessing configuration files, additional data, or outputs.
    /// </summary>
    public required string ProjectRoot { get; init; }

    /// <summary>
    /// Logger for progress reporting, warnings, and errors.
    /// </summary>
    public required ILogger Logger { get; init; }

    /// <summary>
    /// Metadata from training configuration and previous hooks.
    /// Common keys:
    /// - "LabelColumn": Name of the target label column
    /// - "TaskType": ML task type (BinaryClassification, Regression, etc.)
    /// - "ModelName": Name of the model being trained
    /// - "ExperimentId": Unique experiment identifier
    /// - "TimeLimit": Training time limit in seconds
    /// </summary>
    public required Dictionary<string, object> Metadata { get; init; }

    /// <summary>
    /// Retrieves typed metadata value.
    /// </summary>
    /// <typeparam name="T">Expected type of metadata value</typeparam>
    /// <param name="key">Metadata key</param>
    /// <param name="defaultValue">Default value if key not found</param>
    /// <returns>Typed metadata value or default</returns>
    public T GetMetadata<T>(string key, T defaultValue = default!)
    {
        if (Metadata.TryGetValue(key, out var value) && value is T typedValue)
        {
            return typedValue;
        }
        return defaultValue;
    }

    /// <summary>
    /// Checks if metadata key exists.
    /// </summary>
    public bool HasMetadata(string key) => Metadata.ContainsKey(key);
}
