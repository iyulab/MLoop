namespace MLoop.Extensibility.Preprocessing;

/// <summary>
/// Exception thrown when preprocessing script execution fails.
/// </summary>
public class PreprocessingException : Exception
{
    /// <summary>
    /// Name of the script that failed.
    /// </summary>
    public string? ScriptName { get; init; }

    /// <summary>
    /// Index of the script in execution sequence.
    /// </summary>
    public int? ScriptIndex { get; init; }

    public PreprocessingException(string message) : base(message)
    {
    }

    public PreprocessingException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    public PreprocessingException(string message, string scriptName, int scriptIndex)
        : base(message)
    {
        ScriptName = scriptName;
        ScriptIndex = scriptIndex;
    }

    public PreprocessingException(
        string message,
        Exception innerException,
        string scriptName,
        int scriptIndex)
        : base(message, innerException)
    {
        ScriptName = scriptName;
        ScriptIndex = scriptIndex;
    }
}
