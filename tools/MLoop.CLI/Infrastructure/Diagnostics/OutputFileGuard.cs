using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.CLI.Infrastructure.Diagnostics;

/// <summary>
/// An <c>--output</c> that names the data file the command reads. The output is opened for writing before the input has been
/// read to its end, so <c>mloop predict data.csv -o data.csv</c> truncated the file and then failed reading it ("Stream reading
/// encountered exception"), leaving a header-only <c>data.csv</c>; <c>sample create</c> and <c>detect</c> read first and then replaced
/// the data with the sample or the per-point result. A command that writes a file made from another says so and stops before it writes.
/// </summary>
internal static class OutputFileGuard
{
    /// <summary>
    /// The refusal when <paramref name="output"/> is the file <paramref name="input"/> (same place, a link to it included); null when it is not.
    /// <paramref name="input"/> is a path the caller has already placed; <paramref name="output"/> is read the way the command writes it —
    /// under <paramref name="outputRoot"/>, or under the working directory when that is null.
    /// </summary>
    public static (string Cause, string Tip)? Refusal(string? input, string? output, string? outputRoot)
    {
        if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output))
            return null;

        var inputPath = Path.GetFullPath(input);
        var outputPath = ProjectPaths.Resolve(output, outputRoot);
        if (!File.Exists(inputPath) || !ProjectPaths.IsSame(ProjectPaths.RealPath(inputPath), ProjectPaths.RealPath(outputPath)))
            return null;

        return ($"--output is the input file ({Markup.Escape(inputPath)}): writing it would replace the data it is made from.",
            "Give --output another path, or leave it out for the default name.");
    }

    /// <summary>Prints the refusal and returns true when <paramref name="output"/> is the input file.</summary>
    public static bool Refuses(string? input, string? output, string? outputRoot)
    {
        if (Refusal(input, output, outputRoot) is not { } refusal)
            return false;
        ErrorConsole.Error(refusal.Cause, refusal.Tip);
        return true;
    }
}
