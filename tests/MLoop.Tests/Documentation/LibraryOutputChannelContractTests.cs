using System.Text.RegularExpressions;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A library does not write to the process's console. <c>MLoop.Core</c> narrates through the sink
/// its host injects, and where none was injected it says nothing.
/// </summary>
/// <remarks>
/// <para>
/// The rule exists because the host's output rules are installed at the host's layer and a library
/// write goes underneath them. Every <c>--json</c> command replaces the console it owns and promises
/// its consumer a parseable document on stdout; a <c>Console.WriteLine</c> from inside the data
/// loader is not the console the command replaced, so it landed on stdout ahead of the document.
/// Measured before this guard: <c>mloop info --json</c> and <c>mloop analyze … --json</c> both
/// emitted <c>[Info] Removed index column(s): Unnamed: 0</c> as their first line and exited 0 —
/// success reported, output unparseable. The CLI's own <c>--json</c> audit had passed, and could
/// not have caught this: it can only check the streams the CLI controls.
/// </para>
/// <para>
/// The trap was a default, not a call: <c>log ?? Console.WriteLine</c> reads as a courtesy and means
/// "when the host says nothing, write to its stdout anyway". The default is now a sink that
/// discards, so a host that wants these lines passes one and decides where they go. This guard is
/// what keeps the next such default from being written.
/// </para>
/// <para>
/// Scoped to <c>MLoop.Core</c>: the CLI and the API <i>are</i> hosts and own their streams, and
/// <c>MLoop.Core.Tests</c> legitimately captures console output to assert on it.
/// </para>
/// </remarks>
public class LibraryOutputChannelContractTests
{
    /// <summary>
    /// Writes that reach a stream the library does not own. Matched on <c>System.Console</c> only —
    /// the leading boundary is what keeps <c>AnsiConsole.WriteLine</c> out, and that exclusion is the
    /// rule rather than an escape from it: the ambient Spectre console <i>is</i> the one a host's
    /// scope replaces, so a write through it already obeys whatever the host installed.
    /// </summary>
    private static readonly (Regex Pattern, string What)[] ProcessConsoleWrites =
    [
        (ConsoleMember("WriteLine"), "writes to the process's stdout"),
        (ConsoleMember("Write"), "writes to the process's stdout"),
        (ConsoleMember("Error"), "writes to the process's stderr"),
        (ConsoleMember("Out"), "reaches for the process's stdout writer"),
    ];

    private static Regex ConsoleMember(string member) =>
        new($@"(?<![\w.]){Regex.Escape("Console." + member)}\b", RegexOptions.Compiled);

    [Fact]
    public void CoreNarratesThroughAnInjectedSinkRatherThanTheProcessConsole()
    {
        var offenders =
            (from file in RepoSourceTree.ProductionSourceFiles(Path.Combine("src", "MLoop.Core"))
             from line in File.ReadAllLines(file).Select((Text, Number) => (Text, Number))
             where !IsDocumentation(line.Text)
             from write in ProcessConsoleWrites
             where write.Pattern.IsMatch(line.Text)
             select $"{Path.GetRelativePath(RepoSourceTree.RepoRoot, file)}:{line.Number + 1} — "
                  + $"System.Console {write.What}")
            .ToList();

        Assert.True(
            offenders.Count == 0,
            "MLoop.Core must narrate through the Action<string> sink its host injects, defaulting to "
            + "CsvDataLoader.NoLog when the host passes none. A write here goes underneath whatever "
            + "channel the host installed — which is how [Info] lines landed on the stdout that "
            + "--json reserves for its document:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    [Fact]
    public void TheCheckReadsCoreAndWouldSeeAWriteInIt()
    {
        // Negative control: the assertion above passes if the scan finds no files at all, so pin
        // that it reads a real tree and that its tokens match the shape they are meant to catch.
        var files = RepoSourceTree.ProductionSourceFiles(Path.Combine("src", "MLoop.Core")).ToList();

        Assert.Contains(files, f => Path.GetFileName(f) == "CsvDataLoader.cs");
        Assert.True(files.Count > 20, $"Expected the Core source tree, found {files.Count} files.");

        const string wouldOffend = "        var write = log ?? Console.WriteLine;";
        Assert.Contains(ProcessConsoleWrites, w => w.Pattern.IsMatch(wouldOffend));
        Assert.False(IsDocumentation(wouldOffend));

        // …and that the boundary holds: Core's interactive HITL prompt writes through the ambient
        // Spectre console, which a host scope owns and redirects. That is a dependency-direction
        // question, not a stream the library stole.
        const string wouldNotOffend = "        AnsiConsole.WriteLine();";
        Assert.DoesNotContain(ProcessConsoleWrites, w => w.Pattern.IsMatch(wouldNotOffend));
    }

    /// <summary>
    /// A doc comment may name <c>Console.WriteLine</c> — the remarks explaining this very rule do.
    /// Only executable lines are the subject.
    /// </summary>
    private static bool IsDocumentation(string line)
    {
        var trimmed = line.TrimStart();
        return trimmed.StartsWith("///", StringComparison.Ordinal)
            || trimmed.StartsWith("//", StringComparison.Ordinal)
            || trimmed.StartsWith("*", StringComparison.Ordinal);
    }
}
