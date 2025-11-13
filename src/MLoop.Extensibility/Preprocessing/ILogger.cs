namespace MLoop.Extensibility.Preprocessing;

/// <summary>
/// Logger interface for preprocessing scripts.
/// </summary>
public interface ILogger
{
    /// <summary>
    /// Logs informational message.
    /// </summary>
    void Info(string message);

    /// <summary>
    /// Logs warning message.
    /// </summary>
    void Warning(string message);

    /// <summary>
    /// Logs error message.
    /// </summary>
    void Error(string message);

    /// <summary>
    /// Logs error with exception details.
    /// </summary>
    void Error(string message, Exception exception);

    /// <summary>
    /// Logs debug message (only visible with verbose logging).
    /// </summary>
    void Debug(string message);
}
