namespace MLoop.Core.Storage;

/// <summary>
/// A file the project keeps its state in exists and cannot be read - invalid JSON, a truncated archive. That is a different
/// fact from the file being absent, and the two used to share a message: an unreadable <c>production/metadata.json</c> was
/// reported as "No production model found... Train and promote a model first", which sends the user to retrain over a model
/// that was intact. The message names the file, the reason, and the way back.
/// </summary>
public sealed class UnreadableStateFileException : IOException
{
    public UnreadableStateFileException(string path, string reason, string remedy, Exception? inner = null)
        : base($"{path} exists but cannot be read ({reason}). {remedy}", inner)
    {
        Path = path;
        Reason = reason;
    }

    /// <summary>The file that could not be read.</summary>
    public string Path { get; }

    /// <summary>What was wrong with it, in a few words.</summary>
    public string Reason { get; }
}
