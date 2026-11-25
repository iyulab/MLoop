using MLoop.Core.Embedding;

namespace MLoop.Core.Data;

/// <summary>
/// Whether the picture files a command is about to score can be read. A trainer's own loader reads a file it cannot decode as
/// nothing at all — image classification answered such a file with a class and zero probabilities, and object detection failed in
/// the middle of the run without saying which file it was — so the files are read first, and one that cannot be is named.
/// </summary>
public static class PictureFiles
{
    private const int Named = 5;

    /// <summary>The refusal naming the files among <paramref name="paths"/> that cannot be read as pictures, or <c>null</c> when all can.</summary>
    public static string? Unreadable(IEnumerable<string> paths)
    {
        ArgumentNullException.ThrowIfNull(paths);
        var bad = new List<string>();
        var count = 0;
        foreach (var path in paths)
        {
            var why = File.Exists(path) ? ImagePreprocessor.WhyUnreadable(File.ReadAllBytes(path)) : "no such file";
            if (why is null)
                continue;
            if (++count <= Named)
                bad.Add($"'{path}' ({why})");
        }

        return count switch
        {
            0 => null,
            1 => $"A picture cannot be read: {bad[0]}.",
            _ => $"{count} pictures cannot be read: {string.Join("; ", bad)}{(count > Named ? $"; and {count - Named} more" : "")}.",
        };
    }

    /// <summary>What to do about <see cref="Unreadable"/>.</summary>
    public const string UnreadableHint = "Replace each file named with a picture that opens, or take it out of the folder, then run the command again.";
}
