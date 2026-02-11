using System.CommandLine;

using Microsoft.Extensions.Logging;

using FilePrepper.CLI.Commands;

namespace MLoop.CLI.Commands;

/// <summary>
/// Data preprocessing command - integrates FilePrepper CLI commands.
/// Usage: mloop prep [command] [options]
/// </summary>
public class PrepCommand : Command
{
    public PrepCommand() : base("prep", "Data preprocessing tools (powered by FilePrepper)")
    {
        // Add 'run' subcommand for YAML pipeline execution
        this.Add(PrepRunCommand.Create());

        // Create a minimal logger factory for FilePrepper commands
        var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.SetMinimumLevel(LogLevel.Warning);
        });

        // Add all FilePrepper commands via CommandFactory (single source of truth)
        foreach (var command in CommandFactory.CreateAllCommands(loggerFactory))
        {
            this.Add(command);
        }
    }
}
