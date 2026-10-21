using System.CommandLine;
using MLoop.CLI;

namespace MLoop.Tests.Documentation;

/// <summary>
/// <c>-h</c> asks every command for its help. An option that takes <c>-h</c> as its own alias
/// turns <c>mloop serve -h</c> into "Required argument missing for option: '-h'" — the user
/// asked how to use the command and was told they used it wrong.
/// </summary>
public class HelpAliasContractTests
{
    private static IEnumerable<string[]> CommandPaths(Command command, string[] path)
    {
        yield return path;
        foreach (var sub in command.Subcommands)
            foreach (var p in CommandPaths(sub, [.. path, sub.Name]))
                yield return p;
    }

    private static List<string[]> AllCommandPaths() =>
        CommandPaths(Program.BuildRootCommand(), []).ToList();

    [Theory]
    [InlineData("-h")]
    [InlineData("-?")]
    [InlineData("--help")]
    public void Every_command_answers_help(string help)
    {
        var refused = AllCommandPaths()
            .Select(path => (Path: string.Join(' ', path), Errors: Program.BuildRootCommand().Parse([.. path, help]).Errors))
            .Where(r => r.Errors.Count > 0)
            .Select(r => $"mloop {r.Path} {help}: {string.Join("; ", r.Errors.Select(e => e.Message))}")
            .ToList();

        Assert.Empty(refused);
    }

    [Fact]
    public void The_walk_reaches_nested_commands()
    {
        var paths = AllCommandPaths().Select(p => string.Join(' ', p)).ToList();

        Assert.Contains("serve", paths);
        Assert.Contains("feedback add", paths);
        Assert.Contains("runtime install", paths);
    }

    [Fact]
    public void An_option_claiming_minus_h_is_refused()
    {
        // The shape this guard exists for: an option alias that shadows help.
        var root = new RootCommand();
        var command = new Command("probe");
        command.Options.Add(new Option<string>("--host", "-h"));
        root.Subcommands.Add(command);

        Assert.NotEmpty(root.Parse(["probe", "-h"]).Errors);
    }
}
