using Microsoft.ML;

namespace MLoop.Extensibility;

/// <summary>
/// Provides context and data access for hook execution.
/// Contains all information needed by hooks to make decisions and perform operations.
/// </summary>
public class HookContext
{
    /// <summary>
    /// Gets the ML.NET context for ML operations.
    /// </summary>
    public required MLContext MLContext { get; init; }

    /// <summary>
    /// Gets the current data view being processed.
    /// For pre-train hooks, this is the training data.
    /// For post-train hooks, this may be predictions or evaluation results.
    /// </summary>
    public required IDataView DataView { get; init; }

    /// <summary>
    /// Gets the logger for outputting messages to the user.
    /// Use this to provide feedback about hook execution.
    /// </summary>
    public required ILogger Logger { get; init; }

    /// <summary>
    /// Gets read-only metadata about the current operation.
    /// Common keys include:
    /// - "ExperimentId": The current experiment identifier
    /// - "LabelColumn": The name of the label column
    /// - "Metrics": Training metrics (post-train only)
    /// - "BestTrainer": The selected trainer name (post-train only)
    /// - "ModelPath": Path to saved model (post-train only)
    /// </summary>
    public IReadOnlyDictionary<string, object> Metadata { get; private set; } =
        new Dictionary<string, object>();

    private readonly Dictionary<string, object> _mutableMetadata = new();

    /// <summary>
    /// Initializes the metadata dictionary.
    /// </summary>
    /// <param name="metadata">Initial metadata values.</param>
    public void InitializeMetadata(Dictionary<string, object> metadata)
    {
        _mutableMetadata.Clear();
        foreach (var kvp in metadata)
        {
            _mutableMetadata[kvp.Key] = kvp.Value;
        }
        Metadata = _mutableMetadata;
    }

    /// <summary>
    /// Sets or updates a metadata value.
    /// Useful for passing data between hooks or storing context for later hooks.
    /// </summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="value">The metadata value.</param>
    public void SetMetadata(string key, object value)
    {
        _mutableMetadata[key] = value;
    }

    /// <summary>
    /// Gets a metadata value with type casting.
    /// </summary>
    /// <typeparam name="T">The expected type of the metadata value.</typeparam>
    /// <param name="key">The metadata key.</param>
    /// <returns>The metadata value cast to type T, or default(T) if not found.</returns>
    public T? GetMetadata<T>(string key)
    {
        if (Metadata.TryGetValue(key, out var value) && value is T typedValue)
        {
            return typedValue;
        }
        return default;
    }
}

/// <summary>
/// Logger interface for hook output.
/// Abstracts the logging implementation to allow different logging backends.
/// </summary>
public interface ILogger
{
    /// <summary>
    /// Logs an informational message.
    /// </summary>
    void Info(string message);

    /// <summary>
    /// Logs a warning message.
    /// </summary>
    void Warning(string message);

    /// <summary>
    /// Logs an error message.
    /// </summary>
    void Error(string message);

    /// <summary>
    /// Logs a debug message (only shown in verbose mode).
    /// </summary>
    void Debug(string message);
}
