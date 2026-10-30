using System.CommandLine;
using System.Text.RegularExpressions;
using MLoop.CLI;

namespace MLoop.Tests.Documentation;

/// <summary>
/// Which options the guide mentions, measured against the options that exist.
/// </summary>
/// <remarks>
/// <para>Every command has a guide section (<see cref="GuideCoverageContractTests"/>) and every example parses
/// (<c>ReadmeCommandContractTests</c>), yet an option added to a command after its section was written was
/// mentioned nowhere: a check of the guide's option blocks against <c>--help</c> found five of ninety-four long
/// options absent from the whole guide (<c>predict --include-features</c>, <c>promote --no-backup</c> and
/// <c>--decide-only</c>, <c>info --sample-size</c>, <c>prep run --input</c>) — the options of the commands that
/// grew most. A reader of the guide cannot tell a missing option from one that does not exist.</para>
/// <para>The guard asks the weakest useful question — each visible option is named somewhere in the guide, by its
/// long name or one of its long aliases — and not that it sits in its own command's section: a shared option
/// (<c>--json</c>, <c>--records</c>) is explained once. A new option fails here until the guide names it, or until
/// it is hidden on purpose.</para>
/// </remarks>
public class GuideOptionCoverageContractTests
{
    /// <summary>The visible options of every visible command, subcommands included, each with its long names (the name and the long aliases).</summary>
    internal static IEnumerable<(string Command, IReadOnlyList<string> Names)> OptionsOf(Command command, string path = "")
    {
        foreach (var option in command.Options)
        {
            if (option.Hidden || option.Name is "--help" or "--version")
                continue;
            var names = new[] { option.Name }.Concat(option.Aliases).Where(n => n.StartsWith("--", StringComparison.Ordinal)).Distinct(StringComparer.Ordinal).ToList();
            if (names.Count > 0)
                yield return (path.Length == 0 ? command.Name : path, names);
        }
        foreach (var sub in command.Subcommands.Where(c => !c.Hidden))
            foreach (var found in OptionsOf(sub, path.Length == 0 ? sub.Name : $"{path} {sub.Name}"))
                yield return found;
    }

    /// <summary>The options none of whose long names the text mentions as a whole word (<c>--foo</c> is not found inside <c>--foo-bar</c>).</summary>
    internal static List<string> NamedNowhere(IEnumerable<(string Command, IReadOnlyList<string> Names)> options, string text)
        => [.. options
            .Where(o => !o.Names.Any(n => Regex.IsMatch(text, @"(?<![\w-])" + Regex.Escape(n) + @"(?![\w-])")))
            .Select(o => $"{o.Command} {o.Names[0]}")
            .Order(StringComparer.Ordinal)];

    [Fact]
    public void EveryVisibleOptionIsNamedSomewhereInTheGuide()
    {
        var guide = File.ReadAllText(Path.Combine(RepoSourceTree.RepoRoot, "docs", "GUIDE.md"));
        var options = OptionsOf(Program.BuildRootCommand()).ToList();

        // The scan reaches a real command tree: a long option of `train` and one of a nested subcommand are among what it checks.
        Assert.Contains(options, o => o.Command == "train" && o.Names.Contains("--label"));
        Assert.Contains(options, o => o.Command.StartsWith("prep ", StringComparison.Ordinal));

        var missing = NamedNowhere(options, guide);

        Assert.True(missing.Count == 0,
            "These options exist and the guide never names them. Name each in its command's section, or hide it:\n"
            + string.Join('\n', missing));
    }

    /// <summary>The matcher sees what it must: a named option passes, an unnamed one is reported, and a longer name does not stand for a shorter one.</summary>
    [Fact]
    public void TheMatcherReportsAnOptionTheTextDoesNotName()
    {
        var options = new List<(string Command, IReadOnlyList<string> Names)>
        {
            ("demo", ["--named"]),
            ("demo", ["--unnamed"]),
            ("demo", ["--foo"]),
            ("demo sub", ["--rows", "--size"]),
        };
        var text = "Use `--named` here. Also `--foo-bar` and the alias --size.";

        Assert.Equal(["demo --foo", "demo --unnamed"], NamedNowhere(options, text));
    }
}
