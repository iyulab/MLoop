namespace MLoop.Core.Diagnostics;

/// <summary>
/// How a line MLoop.Core hands to a caller's <c>Action&lt;string&gt;</c> sink says that it is a warning rather than
/// narration: it begins with <see cref="WarningMarker"/>. Core's loaders and the trainer search write their
/// findings through one string sink, so the marker is the severity — declared here, once, for the code that writes
/// it and the code that reads it, instead of a prefix each side spells for itself.
/// </summary>
/// <remarks>
/// A caller that only prints lines prints the marker with the rest. One that keeps warnings apart — the CLI puts
/// them in a <c>--json</c> document's <c>warnings</c> and in a training run's <c>warning</c> events — reads them
/// with <see cref="TryReadWarning"/>.
/// </remarks>
public static class CoreMessage
{
    /// <summary>The start of a line that is a warning.</summary>
    public const string WarningMarker = "[Warning] ";

    /// <summary>A warning line for <paramref name="text"/>.</summary>
    public static string Warning(string text) => WarningMarker + text;

    /// <summary>Whether <paramref name="line"/> is a warning, and its text without the marker.</summary>
    public static bool TryReadWarning(string line, out string text)
    {
        var isWarning = line.StartsWith(WarningMarker, StringComparison.Ordinal);
        text = isWarning ? line[WarningMarker.Length..] : line;
        return isWarning;
    }
}
