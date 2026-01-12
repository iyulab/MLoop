namespace MLoop.DataStore.Interfaces;

/// <summary>
/// Logs predictions for monitoring, analysis, and potential retraining triggers.
/// </summary>
public interface IPredictionLogger
{
    /// <summary>
    /// Logs a single prediction with its input features and output.
    /// </summary>
    /// <param name="modelName">Name of the model used</param>
    /// <param name="experimentId">Experiment identifier</param>
    /// <param name="input">Input features as key-value pairs</param>
    /// <param name="output">Prediction output</param>
    /// <param name="confidence">Optional confidence score</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task LogPredictionAsync(
        string modelName,
        string experimentId,
        IDictionary<string, object> input,
        object output,
        double? confidence = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Logs a batch of predictions.
    /// </summary>
    Task LogBatchAsync(
        string modelName,
        string experimentId,
        IEnumerable<PredictionLogEntry> entries,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Retrieves prediction logs for analysis.
    /// </summary>
    /// <param name="modelName">Filter by model name</param>
    /// <param name="from">Start date</param>
    /// <param name="to">End date</param>
    /// <param name="limit">Maximum entries to return</param>
    /// <param name="cancellationToken">Cancellation token</param>
    Task<IReadOnlyList<PredictionLogEntry>> GetLogsAsync(
        string? modelName = null,
        DateTimeOffset? from = null,
        DateTimeOffset? to = null,
        int limit = 1000,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Represents a single prediction log entry.
/// </summary>
public record PredictionLogEntry(
    string ModelName,
    string ExperimentId,
    IDictionary<string, object> Input,
    object Output,
    double? Confidence,
    DateTimeOffset Timestamp);
