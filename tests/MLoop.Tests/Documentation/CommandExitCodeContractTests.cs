using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A command's handler returns an exit code. A handler declared to return nothing has no way to
/// report failure, so every failure it prints exits <c>0</c>.
/// </summary>
/// <remarks>
/// <para>
/// This is not a style rule — it is the half of the CLI-wide contract that
/// <c>ErrorSuggestions.DisplayError</c> already states from the other side: "every command's
/// top-level catch funnels here before returning a non-zero exit code". Three commands did not.
/// Measured before this guard: <c>mloop pipeline nope.yaml</c> outside a project printed
/// <c>Error: Not inside a MLoop project…</c> and exited <c>0</c>. <c>mloop serve</c> did the same
/// when the API assembly was missing, and again when the API it launched crashed; <c>mloop update</c>
/// did it after printing "Update failed."
/// </para>
/// <para>
/// The failure is invisible to everything except a caller that reads the exit code — which is the
/// caller these three commands are written for. <c>pipeline</c> is what a scheduler runs,
/// <c>serve</c> is what a supervisor starts, <c>update</c> is what a provisioning script calls. A
/// person watching the terminal sees the red line and is fine; the automation is told the run was
/// clean.
/// </para>
/// <para>
/// The guard keys on the naming convention every command in this tree follows — the handler a
/// <c>SetAction</c> delegates to is named <c>Execute</c>/<c>ExecuteAsync</c>. That is its limit and
/// also the point: a handler that follows the convention and returns nothing is exactly the mistake
/// that was made three times, and it is not visible in a diff that reads as a normal command.
/// </para>
/// </remarks>
public class CommandExitCodeContractTests
{
    /// <summary>
    /// A handler declaration that cannot carry an exit code: <c>void</c>, or a bare <c>Task</c>
    /// rather than <c>Task&lt;int&gt;</c>. Matched on the declaration, not the call, so an
    /// <c>await</c> of some other <c>Task</c>-returning helper is not caught by it.
    /// </summary>
    private static readonly Regex ExitCodelessHandler =
        new(@"^\s*(?:private|public|internal|protected)(?:\s+static)?(?:\s+async)?"
          + @"\s+(?<returns>void|Task)\s+(?<name>Execute\w*)\s*\(",
            RegexOptions.Compiled | RegexOptions.Multiline);

    [Fact]
    public void EveryCommandHandlerCanReportFailure()
    {
        var offenders =
            (from file in RepoSourceTree.ProductionSourceFiles(
                 Path.Combine("tools", "MLoop.CLI", "Commands"))
             from match in ExitCodelessHandler.Matches(File.ReadAllText(file)).Cast<Match>()
             select $"{Path.GetRelativePath(RepoSourceTree.RepoRoot, file)} — "
                  + $"{match.Groups["name"].Value} returns {match.Groups["returns"].Value}, so every "
                  + "failure it reports still exits 0")
            .ToArray();

        Assert.True(
            offenders.Length == 0,
            "A command handler must return an exit code (int / Task<int>):"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    /// <summary>The severity a block claims and the exit code that follows it are one statement.</summary>
    private static readonly Regex FailureBlock = new(
        @"Schema Validation Failed[\s\S]*?return 1;", RegexOptions.Compiled);

    private static readonly Regex WarningWord = new(@"\[yellow\]Warning:", RegexOptions.Compiled);

    [Fact]
    public void AMessagePrintedOnTheWayToANonZeroExitDoesNotCallItselfAWarning()
    {
        // Measured on the real surface: `predict` with a feature column missing printed a red
        // "Schema Validation Failed:" heading, then a yellow "Warning: ..." line naming the actual
        // problem, then exited 1. The word contradicted both the heading above it and the code
        // below it, and a reader deciding whether the run had succeeded had three answers.
        var offenders = RepoSourceTree.ProductionSourceFiles(Path.Combine("tools", "MLoop.CLI"))
            .Where(f => FailureBlock.Matches(File.ReadAllText(f)).Any(m => WarningWord.IsMatch(m.Value)))
            .Select(RepoSourceTree.RelativeToRoot)
            .Order(StringComparer.Ordinal)
            .ToList();

        Assert.True(offenders.Count == 0,
            "A block that ends in `return 1` states a failure; its message may not be labelled a "
            + "warning:\n" + string.Join('\n', offenders));
    }

    [Fact]
    public void ThatCheckSeesTheShapeItForbids()
    {
        // Companion: the matcher runs from the heading to the exit, and the first version bounded
        // that span to 600 characters -- which the real blocks exceed, so it matched nothing and
        // reported both commands clean while one of them held the exact shape it forbids. The
        // enumeration assertion at the end of this test is what said so.
        const string offending = """
            AnsiConsole.MarkupLine("[red]Schema Validation Failed:[/]");
            AnsiConsole.MarkupLine($"[yellow]Warning: {result.ErrorMessage}[/]");
            return 1;
            """;
        const string corrected = """
            AnsiConsole.MarkupLine("[red]Schema Validation Failed:[/]");
            AnsiConsole.MarkupLine($"[red]Error: {result.ErrorMessage}[/]");
            return 1;
            """;

        Assert.Contains(FailureBlock.Matches(offending), m => WarningWord.IsMatch(m.Value));
        Assert.DoesNotContain(FailureBlock.Matches(corrected), m => WarningWord.IsMatch(m.Value));

        // And the guard must actually reach the two commands it exists for.
        var scanned = RepoSourceTree.ProductionSourceFiles(Path.Combine("tools", "MLoop.CLI"))
            .Where(f => FailureBlock.IsMatch(File.ReadAllText(f)))
            .Select(Path.GetFileName)
            .ToList();
        Assert.Contains("PredictCommand.cs", scanned);
        Assert.Contains("EvaluateCommand.cs", scanned);
    }
}
