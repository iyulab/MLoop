using System.CommandLine;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop new - Generate new project components
/// </summary>
public static class NewCommand
{
    public static Command Create()
    {
        var command = new Command("new", "Generate new project components (hooks, metrics, scripts)");

        // Add subcommands using Subcommands property
        command.Subcommands.Add(NewHookCommand.Create());
        // Future: command.Subcommands.Add(NewMetricCommand.Create());
        // Future: command.Subcommands.Add(NewScriptCommand.Create());

        return command;
    }
}
