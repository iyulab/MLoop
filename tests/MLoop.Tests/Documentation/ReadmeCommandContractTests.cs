using System.CommandLine;
using MLoop.CLI;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Parses every concrete example command in README.md and docs/GUIDE.md against the real command
/// tree (<see cref="Program.BuildRootCommand"/> — the same one <c>Main</c> runs) and asserts it
/// parses without error.
/// </summary>
/// <remarks>
/// <para>
/// A new user's first quickstart command used to fail outright — README's representative train
/// example used a positional label argument the CLI had never accepted, across multiple sites in
/// both README.md and docs/GUIDE.md. The fix was document-only, and a *newly added* example
/// reproduced the same defect immediately afterward — proving a one-time correction doesn't stay
/// correct. This test is the mechanical safeguard that gap called for: every example in both files
/// is re-checked against the actual argument grammar on every build, not just re-read by a human
/// once.
/// </para>
/// <para>
/// This checks the <b>syntax contract</b> only — that System.CommandLine accepts the tokens
/// (right option names, right arity, valid enum-like choices where the parser itself validates
/// them). It cannot check that a referenced file exists or that training actually succeeds; that
/// still requires running the quickstart by hand. Catching "this option doesn't exist" / "this used
/// to be positional and no longer is" automatically is the gap that mattered — the original defect
/// was exactly that class, not a missing-file error.
/// </para>
/// <para>
/// Lines that are usage-grammar summaries rather than literal examples — the CLI Commands
/// Reference table, which intentionally uses <c>&lt;placeholder&gt;</c> and <c>[optional]</c>
/// notation — are excluded by the same marker: any line containing <c>&lt;</c> or <c>[</c> isn't a
/// command a user could paste and run, so parsing it as one would fail for a reason unrelated to
/// the CLI's actual grammar.
/// </para>
/// </remarks>
public class ReadmeCommandContractTests
{
    private static string RepoRoot => TestSourceTree.RepoRoot;
    private static readonly string[] DocFiles = { "README.md", "docs/GUIDE.md" };
    private const string CommandTableHeading = "### Every command";

    public static IEnumerable<object[]> ConcreteExampleCommands() =>
        DocFiles
            .SelectMany(relativePath => ExtractCommands(File.ReadAllLines(Path.Combine(RepoRoot, relativePath)))
                .Select(cmd => (relativePath, cmd)))
            // Distinct: the same example (e.g. "mloop update") legitimately appears more than once
            // within or across these files — xUnit identifies [Theory] cases by parameter value, so
            // an un-deduplicated list produces "duplicate test ID, skipping" noise rather than a
            // second real case. Only the command matters for dedup; which file it also lives in
            // doesn't change what gets parsed.
            .DistinctBy(x => x.cmd)
            .Select(x => new object[] { x.relativePath, x.cmd });

    [Theory]
    [MemberData(nameof(ConcreteExampleCommands))]
    public void Example_parses_without_error(string sourceFile, string commandLine)
    {
        var args = Tokenize(commandLine).Skip(1).ToArray(); // drop the leading "mloop"
        var result = Program.BuildRootCommand().Parse(args);

        Assert.True(result.Errors.Count == 0,
            $"{sourceFile} example \"{commandLine}\" failed to parse: " +
            string.Join("; ", result.Errors.Select(e => e.Message)));
    }

    [Fact]
    public void The_check_actually_catches_a_broken_example()
    {
        // Self-check: the original defect was README showing
        // `mloop train datasets/train.csv price --time 15` — a second positional token where
        // --label is the only way to pass a label. If this ever parses clean, Example_parses_
        // without_error above would rubber-stamp any doc regardless of what it says, so this pins
        // the check has teeth against the exact class of bug it exists to catch.
        var args = Tokenize("mloop train datasets/train.csv price --time 15").Skip(1).ToArray();
        var result = Program.BuildRootCommand().Parse(args);

        Assert.NotEmpty(result.Errors);
    }

    [Fact]
    public void At_least_one_example_was_found_in_each_doc()
    {
        // A change to a doc's formatting (e.g. every example moved into fenced code blocks with no
        // bare `mloop ` prefix left at line-start) could silently make ExtractCommands find
        // nothing for that file — every [Theory] case for it would then vacuously "pass" by not
        // existing. This guards the guard, per file rather than in aggregate, so one doc going
        // quiet isn't masked by the other still having examples.
        foreach (var relativePath in DocFiles)
        {
            var found = ExtractCommands(File.ReadAllLines(Path.Combine(RepoRoot, relativePath)));
            Assert.True(found.Count > 0, $"{relativePath}: no concrete `mloop ...` example lines found.");
        }
    }

    /// <summary>
    /// A line is a literal, runnable example iff it starts a line with <c>mloop </c> and contains
    /// neither <c>&lt;</c> nor <c>[</c> — the two markers these docs use for placeholder/optional
    /// notation in their grammar-summary sections. Trailing <c># comment</c> text is stripped, as is
    /// any shell redirection or pipe: <c>mloop list --json &gt; result.json</c> shows a reader how
    /// the output is meant to be consumed, and the part after the redirection belongs to the shell,
    /// not to this CLI's grammar.
    /// <para>
    /// An example whose comment opens with <c>Error:</c> is <b>documenting a failure</b> — what the
    /// CLI rejects and what it says — so it is excluded rather than asserted to parse. The check
    /// exists to keep a doc's promises true, and for these lines the promise is that they fail; a
    /// reader copying one is meant to see the error. Excluding them is what lets the docs show an
    /// error contract at all instead of only the happy path.
    /// </para>
    /// </summary>
    internal static List<string> ExtractCommands(IEnumerable<string> lines)
    {
        var commands = new List<string>();
        foreach (var raw in lines)
        {
            var line = raw.Trim();
            if (!line.StartsWith("mloop ") || DocumentsAFailure(line))
                continue;

            var withoutComment = StripShellPlumbing(StripTrailingComment(line));
            if (withoutComment.Contains('<') || withoutComment.Contains('['))
                continue;

            commands.Add(withoutComment);
        }
        return commands;
    }

    private static bool DocumentsAFailure(string line)
    {
        var hashIndex = line.IndexOf('#');
        return hashIndex >= 0
            && line[(hashIndex + 1)..].TrimStart().StartsWith("Error:", StringComparison.Ordinal);
    }

    private static string StripTrailingComment(string line)
    {
        var hashIndex = line.IndexOf('#');
        return (hashIndex < 0 ? line : line[..hashIndex]).TrimEnd();
    }

    /// <summary>
    /// Cuts the line at the first shell redirection or pipe. Checked outside quotes so a value that
    /// legitimately contains one of these characters is left intact.
    /// </summary>
    private static string StripShellPlumbing(string line)
    {
        char? quote = null;
        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];

            if (quote is char q)
            {
                if (c == q) quote = null;
                continue;
            }

            if (c is '\'' or '"') { quote = c; continue; }
            if (c is '>' or '|') return line[..i].TrimEnd();
        }

        return line;
    }

    /// <summary>
    /// Shell-like tokenizer: splits on whitespace, but a single- or double-quoted run (e.g. the
    /// JSON payload in <c>--vars '{"a": 1}'</c>) stays one token with its quotes stripped, matching
    /// how a real shell would hand the example to the process. A naive <c>Split(' ')</c> would
    /// break that payload into several tokens and report a false parse failure unrelated to the
    /// CLI's actual grammar.
    /// </summary>
    private static IEnumerable<string> Tokenize(string commandLine)
    {
        var tokens = new List<string>();
        var current = new System.Text.StringBuilder();
        char? quote = null;
        var inToken = false;

        foreach (var c in commandLine)
        {
            if (quote is char q)
            {
                if (c == q) { quote = null; }
                else { current.Append(c); }
                continue;
            }

            if (c is '\'' or '"')
            {
                quote = c;
                inToken = true;
                continue;
            }

            if (char.IsWhiteSpace(c))
            {
                if (inToken) { tokens.Add(current.ToString()); current.Clear(); inToken = false; }
                continue;
            }

            current.Append(c);
            inToken = true;
        }

        if (inToken)
            tokens.Add(current.ToString());

        return tokens;
    }

    [Fact]
    public void The_command_table_lists_exactly_the_commands_that_exist()
    {
        // Eight commands were absent from every user-facing doc — among them `status` and `logs`,
        // which the CLI's own release notes advertise as machine-readable. A reader had no way to
        // learn they exist. A hand-kept list drifts the moment a command is added, which is how the
        // gap opened, so the list is pinned to the tree that actually answers `mloop --help`.
        // Bounded to the table's own section: README has other tables whose first cell is also
        // backticked, and a scan wide enough to catch them would report exit codes as commands.
        var documented = File.ReadAllLines(Path.Combine(RepoRoot, "README.md"))
            .SkipWhile(line => !line.StartsWith(CommandTableHeading, StringComparison.Ordinal))
            .Skip(1)
            .TakeWhile(line => !line.StartsWith("#", StringComparison.Ordinal))
            .Select(line => line.Trim())
            .Where(line => line.StartsWith("| `", StringComparison.Ordinal))
            .Select(line => line[3..line.IndexOf('`', 3)])
            .ToHashSet(StringComparer.Ordinal);

        Assert.True(documented.Count > 0, $"No command table found under \"{CommandTableHeading}\" in README.md.");

        var actual = Program.BuildRootCommand().Subcommands
            .Select(c => c.Name)
            .ToHashSet(StringComparer.Ordinal);

        var missing = actual.Except(documented).Order().ToList();
        var stale = documented.Except(actual).Order().ToList();

        Assert.True(
            missing.Count == 0 && stale.Count == 0,
            $"README's command table is out of step with the command tree.{Environment.NewLine}" +
            $"  Undocumented: {(missing.Count == 0 ? "(none)" : string.Join(", ", missing))}{Environment.NewLine}" +
            $"  Documented but gone: {(stale.Count == 0 ? "(none)" : string.Join(", ", stale))}");
    }

}
