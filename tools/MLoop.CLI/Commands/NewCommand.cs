using System.CommandLine;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop new - Generate new project components
/// </summary>
public static class NewCommand
{
    public static Command Create()
    {
        var command = new Command("new", "Generate new project components (hooks)");

        // Add subcommands using Subcommands property
        command.Subcommands.Add(NewHookCommand.Create());

        return command;
    }
}
