using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.Display;
using MLoop.Core.Data;
using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// <c>mloop predict --save-images &lt;folder&gt;</c>: where the copy of each picture goes, checked before anything is scored, and
/// the writing of them once it has been.
/// </summary>
internal sealed class PictureCopies
{
    private readonly string _projectRoot;
    private readonly string _folder;
    private readonly IReadOnlyList<string> _targets;

    private PictureCopies(string projectRoot, string folder, IReadOnlyList<string> targets)
    {
        _projectRoot = projectRoot;
        _folder = folder;
        _targets = targets;
    }

    /// <summary>
    /// The copies asked for by <paramref name="saveImages"/> of <paramref name="images"/>, read from <paramref name="input"/>
    /// (a folder, one picture, or an annotations file) — null when none were asked for. False after an error was shown.
    /// </summary>
    public static bool TryResolve(string? saveImages, string projectRoot, string input, IReadOnlyList<string> images, out PictureCopies? copies)
    {
        copies = null;
        if (saveImages is null)
            return true;
        if (string.IsNullOrWhiteSpace(saveImages))
        {
            ErrorConsole.Error("--save-images needs a folder to write the pictures into.", "Name one: --save-images <folder>");
            return false;
        }

        var folder = ProjectPaths.Resolve(saveImages, projectRoot);
        // A whole folder is read when a folder is named, and when an annotations file stands for the folder it is in.
        var readFolder = Directory.Exists(input) ? Path.GetFullPath(input)
            : input.EndsWith(".json", StringComparison.OrdinalIgnoreCase) ? Path.GetDirectoryName(Path.GetFullPath(input))
            : null;
        var sourceRoot = readFolder ?? Path.GetDirectoryName(Path.GetFullPath(input))!;
        if (PictureOverlay.TargetRefusal(images, readFolder, sourceRoot, folder) is { } refusal)
        {
            ErrorConsole.Error(Markup.Escape(refusal));
            return false;
        }

        copies = new PictureCopies(projectRoot, folder, PictureOverlay.Targets(images, sourceRoot, folder));
        return true;
    }

    /// <summary>Writes each copy with <paramref name="draw"/> (the picture's position, where to write it) and says where they are.</summary>
    public void Save(string what, Action<int, string> draw)
    {
        for (var i = 0; i < _targets.Count; i++)
            draw(i, _targets[i]);
        ValueLine.Write($"[green]>[/] Pictures with {what} saved to: ", $"{Path.GetRelativePath(_projectRoot, _folder)} ({_targets.Count:N0})");
    }
}
