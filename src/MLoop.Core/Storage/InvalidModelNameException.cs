namespace MLoop.Core.Storage;

/// <summary>
/// A model name that cannot be a model directory — it holds a path separator or <c>..</c>, is reserved by
/// the layout, or breaks the name rule in some other way. Thrown by <see cref="ModelName.Resolve"/>, the one
/// place a name a caller supplied becomes a name the file system sees, so that no endpoint or command has to
/// remember to check. The message is <see cref="ModelName.DescribeViolation"/>'s sentence.
/// </summary>
public sealed class InvalidModelNameException : ArgumentException
{
    public InvalidModelNameException(string name, string violation)
        : base($"Invalid model name '{Printable(name)}'. {violation}")
    {
        Name = name;
    }

    /// <summary>The name as it was given after normalizing — what the file system would have been shown.</summary>
    public string Name { get; }

    // A name that failed may hold a control character (a NUL, a newline); the message is shown in a console and a log.
    private static string Printable(string text) => new(text.Select(c => char.IsControl(c) ? '?' : c).ToArray());
}
