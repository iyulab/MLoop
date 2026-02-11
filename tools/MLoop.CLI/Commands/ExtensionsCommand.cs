using System.CommandLine;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop extensions - Lists all discovered extensibility scripts (hooks and metrics).
/// Phase 1 feature — currently shows placeholder message.
/// </summary>
public static class ExtensionsCommand
{
    public static Command Create()
    {
        var command = new Command("extensions", "List all discovered extensibility scripts");

        var listCommand = new Command("list", "List all hooks and metrics");
        listCommand.SetAction((parseResult) => ExecuteListAsync());

        command.Subcommands.Add(listCommand);
        command.SetAction((parseResult) => ExecuteListAsync()); // Default to list when no subcommand

        return command;
    }

    private static Task<int> ExecuteListAsync()
    {
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[yellow]This command is not yet available[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]The 'extensions' command lists hooks and metrics (Phase 1 features).[/]");
        AnsiConsole.MarkupLine("[grey]This command will be enabled in a future release.[/]");
        AnsiConsole.WriteLine();
        return Task.FromResult(0);
    }
}
