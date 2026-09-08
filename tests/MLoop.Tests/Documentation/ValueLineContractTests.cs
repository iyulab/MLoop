using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A markup line whose last interpolation is a filesystem path goes through <c>ValueLine</c>, not
/// through <c>AnsiConsole.MarkupLine</c> — the latter folds the path at the console width, and a
/// folded path is not a path.
/// </summary>
/// <remarks>
/// <para>
/// Thirty-one call sites printed a path this way, and every one of them broke the same way at any
/// width the value did not fit: Spectre inserts a real newline at the fold, inside the token when
/// the token has nowhere to break. Fixing the one that was reported would have left thirty behind,
/// each waiting for a path long enough to reach the edge — which, at the 80 columns Spectre assumes
/// for a redirected stream, is most of them.
/// </para>
/// <para>
/// The rule is about <i>values</i>, not about paths specifically; paths are what this codebase
/// prints. It does not extend to prose — a sentence reflows without harm, and a machine consumer
/// reads the unfolded text from the event stream regardless. It does catch a path inside a list
/// item (<c>• {filename}</c>) as well as one at the end of a sentence, which is intended: what
/// makes the fold cost something is the value, not its position in the line.
/// </para>
/// </remarks>
public class ValueLineContractTests
{
    /// <summary>
    /// A markup line whose last interpolation looks like a filesystem path — the shape whose fold
    /// costs the reader the value.
    /// </summary>
    private static readonly Regex PathBearingMarkupLine =
        new(@"AnsiConsole\.MarkupLine\(\$""[^""{}]*"
          + @"(?:\[(?:cyan|grey|green|yellow|blue|white)\])?"
          + @"\{(?<expr>[^{}""]*(?:[Pp]ath|Directory|FileName|\.zip)[^{}""]*)\}"
          + @"(?:\[/\])?""\);",
            RegexOptions.Compiled);

    [Fact]
    public void NoCommandPrintsAPathThroughAFoldingLine()
    {
        var offenders =
            (from file in RepoSourceTree.ProductionSourceFiles(Path.Combine("tools", "MLoop.CLI"))
             from line in File.ReadAllLines(file).Select((Text, Number) => (Text, Number))
             let match = PathBearingMarkupLine.Match(line.Text)
             where match.Success
             select $"{Path.GetRelativePath(RepoSourceTree.RepoRoot, file)}:{line.Number + 1} — "
                  + $"prints {match.Groups["expr"].Value.Trim()} through MarkupLine, which folds it; "
                  + "use ValueLine.Write(label, value)")
            .ToArray();

        Assert.True(
            offenders.Length == 0,
            "A path is written whole or not at all:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    /// <summary>
    /// The guard has to be able to see the shape it forbids. If the pattern stopped matching — a
    /// reformat, a rename — this test would keep passing while checking nothing, so it asserts
    /// against a sample of the exact code it is meant to reject.
    /// </summary>
    [Fact]
    public void TheGuardRecognisesTheShapeItForbids()
    {
        Assert.Matches(
            PathBearingMarkupLine,
            """AnsiConsole.MarkupLine($"[green]>[/] Backup: [cyan]{backupPath}[/]");""");

        Assert.DoesNotMatch(
            PathBearingMarkupLine,
            """AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{modelName}[/]");""");
    }
}
