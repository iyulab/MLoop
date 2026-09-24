using System.Text.RegularExpressions;
using MLoop.CLI;

namespace MLoop.Tests.Documentation;

/// <summary>
/// A command the CLI tells a person to run must be one it can run.
/// </summary>
/// <remarks>
/// <para>
/// Measured on a real failure (a Parquet file reaching the label check): the suggestions told the
/// reader to run <c>mloop experiments list</c> and <c>mloop experiments promote</c> — a command group
/// this CLI does not have — and <c>mloop analyze &lt;file&gt;</c>, where <c>analyze</c> requires an
/// aspect first. Each was a reasonable sentence when written; the command surface moved under them
/// and nothing compared the two. Advice that fails is worse than none: the reader has now two errors
/// and a reason to distrust the third suggestion.
/// </para>
/// <para>
/// So every <c>[cyan]mloop …[/]</c> span in the CLI's source is parsed against the real command
/// tree, with placeholders (<c>&lt;data-file&gt;</c>, interpolations) replaced by plausible values.
/// A parse error means the advice does not run. Spans with an interpolation are included — the
/// interpolated part is data, the command around it is not.
/// </para>
/// </remarks>
public class SuggestedCommandContractTests
{
    private static readonly Regex CommandSpan = new(@"\[cyan\](mloop [^\[]*?)\[/\]", RegexOptions.Compiled);

    // Placeholders a suggestion uses, and a value that parses where they stand.
    private static readonly (Regex Pattern, string Value)[] Placeholders =
    [
        (new Regex(@"\{[^}]*\}"), "value"),
        (new Regex(@"<experiment-id>|<exp-id>"), "exp-001"),
        (new Regex(@"<task-type>"), "regression"),
        (new Regex(@"<seconds>|<n>|<port>"), "10"),
        (new Regex(@"<[^>]+>"), "data.csv"),
    ];

    internal static string Substitute(string command)
    {
        foreach (var (pattern, value) in Placeholders)
            command = pattern.Replace(command, value);
        // Markup escapes a literal bracket as a doubled one.
        return command.Replace("[[", "[").Replace("]]", "]").Trim().TrimEnd('.', ',');
    }

    /// <summary>The parse errors for one suggested command line, or none when it runs.</summary>
    internal static IReadOnlyList<string> ParseErrors(string suggested)
    {
        var args = Substitute(suggested).Split(' ', StringSplitOptions.RemoveEmptyEntries).Skip(1).ToArray();
        return Program.BuildRootCommand().Parse(args).Errors.Select(e => e.Message).ToList();
    }

    private static IEnumerable<(string File, string Command)> SuggestedCommands() =>
        from file in RepoSourceTree.ProductionSourceFiles(Path.Combine("tools", "MLoop.CLI"))
        from Match m in CommandSpan.Matches(File.ReadAllText(file))
        select (Path.GetRelativePath(RepoSourceTree.RepoRoot, file), m.Groups[1].Value);

    [Fact]
    public void Every_suggested_command_parses()
    {
        var offenders =
            (from s in SuggestedCommands()
             let errors = ParseErrors(s.Command)
             where errors.Count > 0
             select $"{s.File}: `{s.Command}` — {string.Join("; ", errors)}")
            .Distinct()
            .ToList();

        Assert.True(offenders.Count == 0, "Suggested commands that do not run:\n" + string.Join("\n", offenders));
    }

    [Fact]
    public void The_scan_sees_the_suggestions_it_polices()
    {
        var commands = SuggestedCommands().Select(s => s.Command).ToList();

        // Known sites: a matcher that stopped matching would report "no offenders" with confidence.
        Assert.Contains(commands, c => c.StartsWith("mloop promote", StringComparison.Ordinal));
        Assert.Contains(commands, c => c.StartsWith("mloop info", StringComparison.Ordinal));
        Assert.True(commands.Count >= 20, $"only {commands.Count} suggested commands found");
    }

    [Theory]
    [InlineData("mloop experiments list")]              // no such command group
    [InlineData("mloop analyze <data-file>")]           // analyze needs an aspect first
    [InlineData("mloop nosuchcommand")]
    public void The_checker_refuses_advice_that_does_not_run(string command) =>
        Assert.NotEmpty(ParseErrors(command));

    [Theory]
    [InlineData("mloop promote <experiment-id>")]
    [InlineData("mloop analyze profile <data-file>")]
    [InlineData("mloop train --time <seconds>")]
    [InlineData("mloop train --label {label}")]
    public void The_checker_accepts_advice_that_runs(string command) =>
        Assert.Empty(ParseErrors(command));
}
