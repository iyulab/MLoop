using System.CommandLine;

namespace MLoop.CLI.Commands;

/// <summary>
/// Data preprocessing command.
/// Usage: mloop prep run [options]
///
/// For individual preprocessing operations (add-columns, remove-columns, etc.),
/// install the FilePrepper CLI tool: dotnet tool install -g fileprepper-cli
/// </summary>
public class PrepCommand : Command
{
    public PrepCommand() : base("prep", "Data preprocessing tools (powered by FilePrepper)")
    {
        // YAML pipeline execution (uses MLoop.Core + FilePrepper library)
        this.Add(PrepRunCommand.Create());
    }
}
