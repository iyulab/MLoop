namespace MLoop.Extensibility;

/// <summary>
/// Represents the result of hook execution.
/// Indicates whether the operation should continue or be aborted.
/// </summary>
public class HookResult
{
    /// <summary>
    /// Gets a value indicating whether the operation should continue.
    /// </summary>
    /// <value>
    /// <c>true</c> if the operation should continue normally;
    /// <c>false</c> if the operation should be aborted.
    /// </value>
    public bool ShouldContinue { get; init; }

    /// <summary>
    /// Gets an optional message explaining the result.
    /// Required when <see cref="ShouldContinue"/> is <c>false</c> to explain why the operation was aborted.
    /// </summary>
    public string? Message { get; init; }

    /// <summary>
    /// Creates a result indicating the operation should continue.
    /// </summary>
    /// <returns>A <see cref="HookResult"/> with <see cref="ShouldContinue"/> set to <c>true</c>.</returns>
    /// <example>
    /// <code>
    /// if (dataIsValid)
    /// {
    ///     return HookResult.Continue();
    /// }
    /// </code>
    /// </example>
    public static HookResult Continue() =>
        new() { ShouldContinue = true };

    /// <summary>
    /// Creates a result indicating the operation should continue with an informational message.
    /// </summary>
    /// <param name="message">Informational message about the continuation.</param>
    /// <returns>A <see cref="HookResult"/> with <see cref="ShouldContinue"/> set to <c>true</c>.</returns>
    public static HookResult Continue(string message) =>
        new() { ShouldContinue = true, Message = message };

    /// <summary>
    /// Creates a result indicating the operation should be aborted.
    /// </summary>
    /// <param name="message">
    /// A message explaining why the operation was aborted.
    /// This message will be displayed to the user.
    /// </param>
    /// <returns>A <see cref="HookResult"/> with <see cref="ShouldContinue"/> set to <c>false</c>.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="message"/> is null or empty.
    /// </exception>
    /// <example>
    /// <code>
    /// if (rowCount &lt; 100)
    /// {
    ///     return HookResult.Abort($"Insufficient data: {rowCount} rows, need at least 100");
    /// }
    /// </code>
    /// </example>
    public static HookResult Abort(string message)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            throw new ArgumentException("Abort message cannot be null or empty", nameof(message));
        }

        return new() { ShouldContinue = false, Message = message };
    }
}
