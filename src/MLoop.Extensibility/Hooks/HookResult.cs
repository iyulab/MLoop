namespace MLoop.Extensibility.Hooks;

/// <summary>
/// Result of hook execution indicating what action the pipeline should take.
/// </summary>
public class HookResult
{
    /// <summary>
    /// Action the pipeline should take after hook execution.
    /// </summary>
    public HookAction Action { get; private init; }

    /// <summary>
    /// Optional message explaining the action (e.g., reason for abort).
    /// </summary>
    public string? Message { get; private init; }

    /// <summary>
    /// Optional configuration modifications to apply.
    /// </summary>
    public Dictionary<string, object>? ConfigModifications { get; private init; }

    private HookResult(HookAction action, string? message = null, Dictionary<string, object>? configModifications = null)
    {
        Action = action;
        Message = message;
        ConfigModifications = configModifications;
    }

    /// <summary>
    /// Continue pipeline execution normally (default behavior).
    /// </summary>
    public static HookResult Continue(string? message = null) =>
        new(HookAction.Continue, message);

    /// <summary>
    /// Abort pipeline execution with error message.
    /// Use for critical validation failures or unrecoverable conditions.
    /// </summary>
    /// <param name="reason">Clear explanation of why execution was aborted</param>
    public static HookResult Abort(string reason) =>
        new(HookAction.Abort, reason);

    /// <summary>
    /// Modify pipeline configuration and continue execution.
    /// Use for dynamic configuration adjustments based on data analysis.
    /// </summary>
    /// <param name="modifications">Configuration key-value pairs to modify</param>
    /// <param name="message">Optional explanation of modifications</param>
    public static HookResult ModifyConfig(Dictionary<string, object> modifications, string? message = null) =>
        new(HookAction.ModifyConfig, message, modifications);
}

/// <summary>
/// Action to take after hook execution.
/// </summary>
public enum HookAction
{
    /// <summary>
    /// Continue normal pipeline execution (default).
    /// </summary>
    Continue,

    /// <summary>
    /// Abort pipeline execution with error.
    /// </summary>
    Abort,

    /// <summary>
    /// Modify configuration and continue.
    /// </summary>
    ModifyConfig
}
