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
/// notation — are not parsed as written: any line containing <c>&lt;</c> or <c>[</c> isn't a
/// command a user could paste and run. They are held to the grammar a second way
/// (<see cref="Grammar_summary_parses_with_every_optional_present"/>): written out in full, with
/// every optional present and a value of the right kind in each placeholder. That is how a summary
/// listing a positional the command does not have (<c>predict [model] [data]</c>,
/// <c>train &lt;data-file&gt; &lt;label-column&gt;</c>) stopped passing as correct.
/// </para>
/// </remarks>
public class ReadmeCommandContractTests
{
    // Declared first: the static initializers below read them, and run in the order written.
    /// <summary>The names of the commands <c>mloop</c> has, for a job written as a quoted command without the program's name.</summary>
    private static readonly Lazy<HashSet<string>> RootVerbs = new(() =>
        [.. Program.BuildRootCommand().Subcommands.Select(c => c.Name)]);

    private static readonly System.Text.RegularExpressions.Regex WrappedCommand = new(
        @"^(?:nice\s+-n\s+\d+|taskset\s+-c\s+\S+|systemd-run\s+.*?|sudo|time|env\s+\S+=\S+)\s+(mloop\s.*)$",
        System.Text.RegularExpressions.RegexOptions.Compiled);

    private static readonly System.Text.RegularExpressions.Regex QuotedJob = new(
        "^\"(?<job>[a-z][a-z-]*\\s[^\"]*)\"\\s*\\\\?$",
        System.Text.RegularExpressions.RegexOptions.Compiled);

    private static string RepoRoot => RepoSourceTree.RepoRoot;
    /// <summary>The two documents a new user reads first: they must exist and must yield examples.</summary>
    private static readonly string[] CoreDocFiles = { "README.md", "docs/GUIDE.md" };
    private const string CommandTableHeading = "### Every command";

    /// <summary>
    /// Every markdown file of the repository that shows <c>mloop</c> command lines: the two core documents and any other
    /// guide, architecture note or example README. A scan of 408 lines in seventeen files found fourteen that failed as written,
    /// all outside the two files this test used to read - the same class of defect, in the files nobody was checking.
    /// The change log is history (it quotes commands as they were) and is not read.
    /// </summary>
    private static readonly string[] DocFiles = CoreDocFiles
        .Concat(Directory.EnumerateFiles(RepoSourceTree.RepoRoot, "*.md", SearchOption.AllDirectories)
            .Select(path => Path.GetRelativePath(RepoSourceTree.RepoRoot, path).Replace('\\', '/'))
            .Where(path => !path.Split('/').Any(part => part is "bin" or "obj" or "node_modules" or "TestResults" or ".git")
                           && !path.Equals("CHANGELOG.md", StringComparison.OrdinalIgnoreCase))
            .Except(CoreDocFiles)
            .Where(path => ExtractCommands(File.ReadAllLines(Path.Combine(RepoSourceTree.RepoRoot, path))).Count > 0
                           || ExtractGrammarSummaries(File.ReadAllLines(Path.Combine(RepoSourceTree.RepoRoot, path))).Count > 0)
            .Order(StringComparer.Ordinal))
        .ToArray();

    /// <summary>
    /// Lines that show what the CLI does not accept, on purpose: the "bad" half of a good/bad pair, a command of a design
    /// that is not built yet, a two-column terminal diagram. Each carries its reason, and
    /// <see cref="Every_exemption_still_names_a_line_that_fails_to_parse"/> removes the entry the day the line is fixed or the CLI grows it.
    /// </summary>
    private static readonly (string File, string Command, string Why)[] DeliberatelyNotParsable =
    {
        ("CONTRIBUTING.md", "mloop train --config complex.yaml --schema schema.json --mappings map.yaml", "the anti-pattern its principle warns against"),
        ("CONTRIBUTING.md", "mloop train --pipeline transform-train-evaluate --config pipeline.yaml", "the anti-pattern its principle warns against"),
        ("docs/ARCHITECTURE.md", "mloop train data.csv --label target --detach", "section 13.4, a design marked as future"),
        ("docs/ARCHITECTURE.md", "mloop experiment show exp-005", "section 13.4, a design marked as future"),
        ("docs/ARCHITECTURE.md", "mloop train data.csv         $ mloop train data.csv", "two terminals drawn side by side"),
    };

    private static bool IsExempt(string file, string command)
        => DeliberatelyNotParsable.Any(e => e.File == file && e.Command == command);

    public static IEnumerable<object[]> ConcreteExampleCommands() =>
        DocFiles
            .SelectMany(relativePath => ExtractCommands(File.ReadAllLines(Path.Combine(RepoRoot, relativePath)))
                .Select(cmd => (relativePath, cmd)))
            // Distinct: the same example (e.g. "mloop update") legitimately appears more than once
            // within or across these files — xUnit identifies [Theory] cases by parameter value, so
            // an un-deduplicated list produces "duplicate test ID, skipping" noise rather than a
            // second real case. Only the command matters for dedup; which file it also lives in
            // doesn't change what gets parsed.
            .Where(x => !IsExempt(x.relativePath, x.cmd))
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
    public void A_command_after_a_wrapper_or_quoted_as_a_job_is_found_and_held_to_the_grammar()
    {
        // The shapes the line-start scan missed: a job handed to a queue script, and a command run under nice. Both carried
        // the retired `train data.csv price` form past this test.
        var found = ExtractCommands(
        [
            "  \"train dataset1.csv price --time 1800\" \\",
            "nice -n 19 mloop train dataset.csv price --time 1800",
            "  \"hello world\" \\",
            "  \"not-a-command x --y\"",
        ]);

        Assert.Equal(["mloop train dataset1.csv price --time 1800", "mloop train dataset.csv price --time 1800"], found);
        Assert.All(found, c => Assert.NotEmpty(Program.BuildRootCommand().Parse(Tokenize(c).Skip(1).ToArray()).Errors));
        Assert.Equal("mloop train d.csv --label price", ExtractCommands(["  \"train d.csv --label price\" \\"]).Single());
    }

    [Fact]
    public void The_scripts_readme_shows_jobs_this_scan_now_reads()
    {
        // The guard sees: the queue examples are quoted jobs, and the scan finds them.
        var jobs = ExtractCommands(File.ReadAllLines(Path.Combine(RepoRoot, "examples/scripts/README.md")));

        Assert.Contains(jobs, c => c.StartsWith("mloop train ", StringComparison.Ordinal));
        Assert.Contains(jobs, c => c.StartsWith("mloop evaluate exp-", StringComparison.Ordinal));
    }

    [Fact]
    public void At_least_one_example_was_found_in_each_doc()
    {
        // A change to a doc's formatting (e.g. every example moved into fenced code blocks with no
        // bare `mloop ` prefix left at line-start) could silently make ExtractCommands find
        // nothing for that file — every [Theory] case for it would then vacuously "pass" by not
        // existing. This guards the guard, per file rather than in aggregate, so one doc going
        // quiet isn't masked by the other still having examples.
        foreach (var relativePath in CoreDocFiles)
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
            var line = AsMloopCommand(StripPrompt(raw.Trim()));
            if (line is null || DocumentsAFailure(line))
                continue;

            var withoutComment = StripShellPlumbing(StripTrailingComment(line));
            if (withoutComment.Contains('<') || withoutComment.Contains('[') || HasTemplateBrace(withoutComment))
                continue;

            commands.Add(withoutComment);
        }
        return commands;
    }

    /// <summary>
    /// The grammar-summary lines the literal-example scan leaves out: a command with <c>&lt;placeholder&gt;</c> or
    /// <c>[optional]</c> notation. They state the grammar a reader will copy from, so they are held to it too.
    /// </summary>
    internal static List<string> ExtractGrammarSummaries(IEnumerable<string> lines)
    {
        var summaries = new List<string>();
        foreach (var raw in lines)
        {
            var line = AsMloopCommand(StripPrompt(raw.Trim()));
            if (line is null || DocumentsAFailure(line))
                continue;

            // Only the comment is cut: in a summary ">" closes a placeholder, it is not a redirection.
            var withoutComment = StripTrailingComment(line);
            if (withoutComment.Contains('<') || withoutComment.Contains('['))
                summaries.Add(withoutComment);
        }
        return summaries;
    }

    public static IEnumerable<object[]> GrammarSummaryCommands() =>
        DocFiles
            .SelectMany(relativePath => ExtractGrammarSummaries(File.ReadAllLines(Path.Combine(RepoRoot, relativePath)))
                .Select(cmd => (relativePath, cmd)))
            .DistinctBy(x => x.cmd)
            .Select(x => new object[] { x.relativePath, x.cmd });

    /// <summary>
    /// Writes a grammar summary as one command that uses everything it offers: the first of a set of alternatives,
    /// every optional group present, and each placeholder given a value of the kind its option takes. A summary that
    /// lists a positional the command does not have, or an option it does not know, fails to parse in that form —
    /// which the literal-example scan above, skipping these lines, never saw.
    /// </summary>
    internal static string Instantiate(string summary)
    {
        // A set of alternatives — "[--latest|--best]", "<a|b>" — is its first member.
        var text = System.Text.RegularExpressions.Regex.Replace(summary, @"([\[<])([^\[\]<>|]+)\|[^\[\]<>]*([\]>])", "$1$2$3");
        // "--drop/--keep" offers two options; "[options]" stands for the options the line does not list.
        text = System.Text.RegularExpressions.Regex.Replace(text, @"(--[a-z-]+)(?:/--[a-z-]+)+", "$1");
        text = text.Replace("[options]", "");
        text = text.Replace("[", "").Replace("]", "");

        var tokens = Tokenize(text).ToList();
        for (var i = 0; i < tokens.Count; i++)
        {
            var after = i > 0 && tokens[i - 1].StartsWith('-');
            // "--bank-size N": a bare capital letter after an option stands for a number.
            if (after && tokens[i].Length == 1 && char.IsUpper(tokens[i][0]))
            {
                tokens[i] = "1";
                continue;
            }

            if (!tokens[i].StartsWith('<') || !tokens[i].EndsWith('>'))
                continue;

            tokens[i] = tokens[i] switch
            {
                "<type>" or "<task>" => "regression",
                "<port>" or "<n>" or "<N>" or "<days>" or "<rows>" => "1",
                _ => after ? "1" : "x",
            };
        }

        return string.Join(' ', tokens);
    }

    [Theory]
    [MemberData(nameof(GrammarSummaryCommands))]
    public void Grammar_summary_parses_with_every_optional_present(string sourceFile, string summary)
    {
        var instance = Instantiate(summary);
        var args = Tokenize(instance).Skip(1).ToArray();
        var result = Program.BuildRootCommand().Parse(args);

        Assert.True(result.Errors.Count == 0,
            $"{sourceFile} grammar summary \"{summary}\" (as \"{instance}\") failed to parse: " +
            string.Join("; ", result.Errors.Select(e => e.Message)));
    }

    [Fact]
    public void The_grammar_check_catches_a_positional_the_command_does_not_take()
    {
        // The summary that stood in README for months: `predict [model] [data]` — predict takes the data file, and the
        // model is the --name option. Written in full, it has a second positional, which the parser refuses.
        var args = Tokenize(Instantiate("mloop predict [model] [data]")).Skip(1).ToArray();

        Assert.NotEmpty(Program.BuildRootCommand().Parse(args).Errors);
    }

    [Fact]
    public void At_least_one_grammar_summary_was_found_in_each_doc()
    {
        foreach (var relativePath in CoreDocFiles)
            Assert.True(ExtractGrammarSummaries(File.ReadAllLines(Path.Combine(RepoRoot, relativePath))).Count > 0,
                $"{relativePath}: no grammar-summary lines found.");
    }

    /// <summary>
    /// A <c>{1}</c> outside quotes is a template slot of a tool that runs the line (GNU parallel, a CI matrix), not an argument a user types.
    /// A brace inside quotes is a JSON value and stays.
    /// </summary>
    private static bool HasTemplateBrace(string line)
    {
        char? quote = null;
        foreach (var c in line)
        {
            if (quote is char q) { if (c == q) quote = null; continue; }
            if (c is '\'' or '"') { quote = c; continue; }
            if (c == '{') return true;
        }
        return false;
    }

    /// <summary>A transcript shows the shell prompt before the command: <c>$ mloop train ...</c> is the same example.</summary>
    /// <summary>
    /// The <c>mloop</c> command a line shows, or <see langword="null"/>: the line itself when it starts with <c>mloop </c>, the
    /// command after a wrapper (<c>nice -n 19 mloop train …</c>), or a job handed to a script as a quoted command without the
    /// program's name (<c>"train data.csv price --time 1800" \</c>, as <c>examples/scripts/mloop-queue.sh</c> takes them) —
    /// two shapes the line-start scan never saw, and the old positional-label form of <c>train</c> lived in both.
    /// </summary>
    internal static string? AsMloopCommand(string line)
    {
        if (line.StartsWith("mloop ", StringComparison.Ordinal))
            return line;
        if (WrappedCommand.Match(line) is { Success: true } wrapped)
            return wrapped.Groups[1].Value;
        if (QuotedJob.Match(line) is { Success: true } quoted && RootVerbs.Value.Contains(quoted.Groups["job"].Value.Split(' ')[0]))
            return "mloop " + quoted.Groups["job"].Value;
        return null;
    }

    private static string StripPrompt(string line) => line.StartsWith("$ ", StringComparison.Ordinal) ? line[2..].TrimStart() : line;

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
    /// Cuts the line at the first shell redirection, pipe, background marker (<c>&amp;</c>) or command separator, and drops a trailing line-continuation backslash. Checked outside quotes so a value that
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
            if (c is '>' or '|' or '&' or ';') return line[..i].TrimEnd();
        }

        return line.EndsWith('\\') ? line[..^1].TrimEnd() : line;
    }

    /// <summary>
    /// Shell-like tokenizer: splits on whitespace, but a single- or double-quoted run (e.g. the
    /// JSON payload in <c>--vars '{"a": 1}'</c>) stays one token with its quotes stripped, matching
    /// how a real shell would hand the example to the process. A naive <c>Split(' ')</c> would
    /// break that payload into several tokens and report a false parse failure unrelated to the
    /// CLI's actual grammar.
    /// </summary>
    internal static IEnumerable<string> Tokenize(string commandLine)
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

    [Fact]
    public void The_scan_reaches_the_documents_beyond_README_and_GUIDE()
    {
        // The scan once read two files, and the fourteen broken lines of a wider look sat in the others. If discovery
        // stopped finding them, this test would go on passing over two files - so it must see the files that hold them.
        Assert.Contains("CONTRIBUTING.md", DocFiles);
        Assert.Contains("docs/ARCHITECTURE.md", DocFiles);
        Assert.Contains(DocFiles, f => f.StartsWith("examples/", StringComparison.Ordinal));
        Assert.DoesNotContain(DocFiles, f => f.Equals("CHANGELOG.md", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void A_transcript_line_with_a_shell_prompt_is_an_example_too()
    {
        Assert.Equal(["mloop list --json"], ExtractCommands(["$ mloop list --json   # View experiments"]));
    }

    [Fact]
    public void Every_exemption_still_names_a_line_that_fails_to_parse()
    {
        // An exemption is a promise that the line is wrong on purpose. When the line is fixed, deleted or accepted by the CLI,
        // the entry would only keep hiding the next real defect behind the same words.
        foreach (var (file, command, why) in DeliberatelyNotParsable)
        {
            var lines = ExtractCommands(File.ReadAllLines(Path.Combine(RepoRoot, file)));
            Assert.True(lines.Contains(command), $"{file}: the exempted line \"{command}\" ({why}) is no longer in the file - remove its entry.");
            var result = Program.BuildRootCommand().Parse(Tokenize(command).Skip(1).ToArray());
            Assert.True(result.Errors.Count > 0, $"{file}: \"{command}\" parses now - remove its entry.");
        }
    }
}
