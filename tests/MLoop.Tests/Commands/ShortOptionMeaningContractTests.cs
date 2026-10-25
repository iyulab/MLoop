using System.CommandLine;
using MLoop.CLI;

namespace MLoop.Tests.Commands;

/// <summary>
/// A short option letter means one thing across the whole CLI. Before this guard <c>-f</c> was
/// <c>--force</c> on two commands, <c>--from</c> on a third and <c>--feedback</c> on a fourth; <c>-l</c>
/// was a label, a limit and a log flag — so a letter learned on one command did something else, or
/// nothing, on the next. The guard reads the live command tree, not the source, so an alias added by
/// any route (constructor, <c>Aliases.Add</c>) is seen.
/// </summary>
public class ShortOptionMeaningContractTests
{
    private static IEnumerable<Command> Walk(Command command)
    {
        yield return command;
        foreach (var child in command.Subcommands)
            foreach (var descendant in Walk(child))
                yield return descendant;
    }

    /// <summary>Short letter → the distinct long option names it stands for.</summary>
    internal static IReadOnlyDictionary<string, IReadOnlySet<string>> ShortLetterMeanings(Command root)
    {
        var meanings = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (var command in Walk(root))
            foreach (var option in command.Options)
                foreach (var alias in option.Aliases.Append(option.Name))
                {
                    if (alias.Length != 2 || alias[0] != '-' || alias[1] == '-')
                        continue;
                    if (!meanings.TryGetValue(alias, out var names))
                        meanings[alias] = names = new HashSet<string>(StringComparer.Ordinal);
                    names.Add(option.Name);
                }
        return meanings.ToDictionary(pair => pair.Key, pair => (IReadOnlySet<string>)pair.Value);
    }

    [Fact]
    public void EveryShortLetterStandsForOneLongOption()
    {
        var conflicts = ShortLetterMeanings(Program.BuildRootCommand())
            .Where(pair => pair.Value.Count > 1)
            .Select(pair => $"{pair.Key}: {string.Join(", ", pair.Value.Order())}")
            .ToList();

        Assert.True(conflicts.Count == 0,
            "A short option letter must mean the same long option on every command (drop the alias where it means something else):\n  "
            + string.Join("\n  ", conflicts));
    }

    [Fact]
    public void TheGuardSeesTheLettersTheCliUses()
    {
        // Blind-guard companion: a scan that stopped seeing aliases would report "no conflicts".
        var meanings = ShortLetterMeanings(Program.BuildRootCommand());

        Assert.Contains("--name", meanings["-n"]);
        Assert.Contains("--output", meanings["-o"]);
        Assert.True(meanings.Count >= 10, $"only {meanings.Count} short letters seen");
    }

    [Fact]
    public void TheGuardRefusesTheShapeItForbids()
    {
        var root = new RootCommand();
        var first = new Command("a");
        first.Options.Add(new Option<string>("--force", "-f"));
        var second = new Command("b");
        second.Options.Add(new Option<string>("--from", "-f"));
        var third = new Command("c");
        third.Options.Add(new Option<string>("--force", "-f"));
        root.Subcommands.Add(first);
        root.Subcommands.Add(second);
        root.Subcommands.Add(third);

        var meanings = ShortLetterMeanings(root);

        Assert.Equal(["--force", "--from"], meanings["-f"].Order());

        var legitimate = new RootCommand();
        legitimate.Subcommands.Add(first);
        legitimate.Subcommands.Add(third);
        Assert.Single(ShortLetterMeanings(legitimate)["-f"]);
    }
}
