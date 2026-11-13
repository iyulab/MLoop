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
/// <param name="ModelName">The model name the prediction is logged under.</param>
/// <param name="ExperimentId">The experiment that answered.</param>
/// <param name="Input">What the answer was made from.</param>
/// <param name="Output">The answer (<c>PredictionLogOutput</c>).</param>
/// <param name="Confidence">The answer's per-row confidence, when it has one.</param>
/// <param name="Timestamp">When it was logged.</param>
/// <param name="Id">The id feedback names it by; made when none is given.</param>
/// <param name="Interval">The interval the answer stated around its value (a regression or forecast answer), or null when it
/// stated none — what lets a later reader judge whether the stated level held (<see cref="FeedbackMetrics.IntervalCoverage"/>).</param>
public record PredictionLogEntry(
    string ModelName,
    string ExperimentId,
    IDictionary<string, object> Input,
    object Output,
    double? Confidence,
    DateTimeOffset Timestamp,
    string? Id = null,
    PredictionInterval? Interval = null);

/// <summary>
/// The interval a prediction stated around its value: <paramref name="Lower"/> to <paramref name="Upper"/> at the level
/// <paramref name="Confidence"/> (a share, 0.9 for "90 %"). Kept as stated — what the answer said, not a recomputation.
/// </summary>
public sealed record PredictionInterval(double Lower, double Upper, double Confidence)
{
    /// <summary>
    /// Whether this interval can be judged against an actual value: finite bounds the right way round and a level that is a
    /// share. (A method, not a property — a property would be written into every log line.)
    /// </summary>
    public bool CanJudge() =>
        double.IsFinite(Lower) && double.IsFinite(Upper) && Lower <= Upper
        && double.IsFinite(Confidence) && Confidence is > 0 and < 1;

    /// <summary>Whether <paramref name="actual"/> lies in the interval, its bounds included.</summary>
    public bool Contains(double actual) => actual >= Lower && actual <= Upper;
}

/// <summary>
/// The identifier a logged prediction is known by — what feedback names when it reports the actual
/// value. One format, made in one place, so an id handed back to a caller before the entry is written
/// is the id the log file holds.
/// </summary>
public static class PredictionIds
{
    /// <summary>A new prediction id: 16 lowercase hex characters.</summary>
    public static string New() => Guid.NewGuid().ToString("N")[..16];
}
