using System.CommandLine;
using Microsoft.Extensions.Logging;
using MLoop.AIAgent.Infrastructure;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop agents - Manage AI agent installations
/// </summary>
public static class AgentsCommand
{
    public static Command Create()
    {
        var command = new Command("agents", "Manage AI agent installations");

        // Subcommands
        command.Add(CreateListCommand());
        command.Add(CreateInstallCommand());
        command.Add(CreateInfoCommand());
        command.Add(CreateValidateCommand());

        return command;
    }

    /// <summary>
    /// mloop agents list - List installed agents
    /// </summary>
    private static Command CreateListCommand()
    {
        var command = new Command("list", "List all installed AI agents");

        command.SetAction(async (parseResult) =>
        {
            try
            {
                using var loggerFactory = LoggerFactory.Create(builder =>
                {
                    builder.AddConsole();
                    builder.SetMinimumLevel(LogLevel.Warning);
                });

                var manager = new AgentManager(loggerFactory.CreateLogger<AgentManager>());
                var statuses = await manager.CheckAgentStatusAsync();

                if (statuses.Count == 0)
                {
                    AnsiConsole.MarkupLine("[yellow]No agents installed.[/]");
                    AnsiConsole.MarkupLine("[grey]Run 'mloop agents install' to install built-in agents.[/]");
                    return 0;
                }

                // Display agents table
                var table = new Table()
                    .Border(TableBorder.Rounded)
                    .AddColumn("[blue]Agent Name[/]")
                    .AddColumn("[blue]Type[/]")
                    .AddColumn("[blue]Version[/]")
                    .AddColumn("[blue]Status[/]");

                foreach (var status in statuses)
                {
                    var type = status.IsBuiltIn ? "Built-in" : "Custom";
                    var statusText = BuildStatusText(status);

                    table.AddRow(
                        status.Name,
                        type,
                        status.Version,
                        statusText);
                }

                AnsiConsole.Write(table);
                AnsiConsole.WriteLine();

                // Display warnings for user-modified agents with updates
                var modifiedWithUpdates = statuses
                    .Where(s => s.UserModified && s.HasUpdate)
                    .ToList();

                if (modifiedWithUpdates.Any())
                {
                    AnsiConsole.MarkupLine("[yellow]⚠️ Updates available for modified agents:[/]");
                    foreach (var agent in modifiedWithUpdates)
                    {
                        AnsiConsole.MarkupLine($"  [grey]•[/] [blue]{agent.Name}[/] - Use 'mloop agents install --force' to update");
                    }
                    AnsiConsole.WriteLine();
                }

                return 0;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
                return 1;
            }
        });

        return command;
    }

    /// <summary>
    /// mloop agents install - Install or update agents
    /// </summary>
    private static Command CreateInstallCommand()
    {
        var forceOption = new Option<bool>("--force")
        {
            Description = "Force update even if user has modified the agent",
            DefaultValueFactory = _ => false
        };

        var agentOption = new Option<string?>("--agent", "-a")
        {
            Description = "Specific agent to install (installs all if not specified)"
        };

        var command = new Command("install", "Install or update AI agents")
        {
            forceOption,
            agentOption
        };

        command.SetAction(async (parseResult) =>
        {
            var force = parseResult.GetValue(forceOption);
            var agentName = parseResult.GetValue(agentOption);

            try
            {
                using var loggerFactory = LoggerFactory.Create(builder =>
                {
                    builder.AddConsole();
                    builder.SetMinimumLevel(LogLevel.Information);
                });

                var manager = new AgentManager(loggerFactory.CreateLogger<AgentManager>());

                InstallationResult result;

                if (!string.IsNullOrEmpty(agentName))
                {
                    // Install specific agent
                    AnsiConsole.MarkupLine($"[blue]Installing agent '{agentName}'...[/]");
                    result = await manager.InstallAgentAsync(agentName, force);
                }
                else
                {
                    // Install all built-in agents
                    AnsiConsole.MarkupLine("[blue]Installing all built-in agents...[/]");
                    result = await manager.InstallBuiltInAgentsAsync(force);
                }

                // Display results
                if (result.Success)
                {
                    AnsiConsole.MarkupLine($"[green]✓[/] {result.Message}");

                    if (result.InstalledAgents.Any())
                    {
                        AnsiConsole.MarkupLine("[green]Installed:[/]");
                        foreach (var agent in result.InstalledAgents)
                        {
                            AnsiConsole.MarkupLine($"  [grey]•[/] {agent}");
                        }
                    }

                    if (result.UpdatedAgents.Any())
                    {
                        AnsiConsole.MarkupLine("[green]Updated:[/]");
                        foreach (var agent in result.UpdatedAgents)
                        {
                            AnsiConsole.MarkupLine($"  [grey]•[/] {agent}");
                        }
                    }

                    if (result.SkippedAgents.Any())
                    {
                        AnsiConsole.MarkupLine("[yellow]Skipped:[/]");
                        foreach (var agent in result.SkippedAgents)
                        {
                            AnsiConsole.MarkupLine($"  [grey]•[/] {agent}");
                        }
                    }

                    if (result.Warnings.Any())
                    {
                        AnsiConsole.WriteLine();
                        AnsiConsole.MarkupLine("[yellow]⚠️ Warnings:[/]");
                        foreach (var warning in result.Warnings)
                        {
                            AnsiConsole.MarkupLine($"  [grey]•[/] {warning}");
                        }
                    }
                }
                else
                {
                    AnsiConsole.MarkupLine($"[red]✗[/] {result.Message}");
                    return 1;
                }

                AnsiConsole.WriteLine();
                return 0;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
                AnsiConsole.WriteException(ex);
                return 1;
            }
        });

        return command;
    }

    /// <summary>
    /// mloop agents info - Show agent information
    /// </summary>
    private static Command CreateInfoCommand()
    {
        var agentArg = new Argument<string>("agent-name")
        {
            Description = "Name of the agent to show information for"
        };

        var command = new Command("info", "Show detailed agent information")
        {
            agentArg
        };

        command.SetAction(async (parseResult) =>
        {
            var agentName = parseResult.GetValue(agentArg);

            if (string.IsNullOrEmpty(agentName))
            {
                AnsiConsole.MarkupLine("[red]Agent name is required.[/]");
                return 1;
            }

            try
            {
                using var loggerFactory = LoggerFactory.Create(builder =>
                {
                    builder.AddConsole();
                    builder.SetMinimumLevel(LogLevel.Warning);
                });

                var manager = new AgentManager(loggerFactory.CreateLogger<AgentManager>());
                var status = await manager.CheckSingleAgentStatusAsync(agentName);

                if (!status.IsInstalled)
                {
                    AnsiConsole.MarkupLine($"[yellow]Agent '{agentName}' is not installed.[/]");
                    AnsiConsole.MarkupLine("[grey]Run 'mloop agents install' to install it.[/]");
                    return 1;
                }

                // Display agent information
                var panel = new Panel(new Markup(
                    $"[blue]Name:[/] {status.Name}\n" +
                    $"[blue]Type:[/] {(status.IsBuiltIn ? "Built-in" : "Custom")}\n" +
                    $"[blue]Version:[/] {status.Version}\n" +
                    $"[blue]Status:[/] {BuildStatusText(status)}\n"))
                {
                    Header = new PanelHeader($"[blue]Agent: {agentName}[/]"),
                    Border = BoxBorder.Rounded
                };

                AnsiConsole.Write(panel);
                AnsiConsole.WriteLine();

                // Show agent directory
                var userAgentsDir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                    ".mloop", "agents", agentName);

                AnsiConsole.MarkupLine($"[grey]Location:[/] {userAgentsDir}");
                AnsiConsole.WriteLine();

                // Show warnings if applicable
                if (status.UserModified && status.HasUpdate)
                {
                    AnsiConsole.MarkupLine("[yellow]⚠️ You have modified this agent and updates are available.[/]");
                    AnsiConsole.MarkupLine("[grey]Run 'mloop agents install --agent {0} --force' to update and lose modifications.[/]", agentName);
                    AnsiConsole.WriteLine();
                }

                return 0;
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
                return 1;
            }
        });

        return command;
    }

    /// <summary>
    /// mloop agents validate - Validate agent files
    /// </summary>
    private static Command CreateValidateCommand()
    {
        var agentOption = new Option<string?>("--agent", "-a")
        {
            Description = "Specific agent to validate (validates all if not specified)"
        };

        var command = new Command("validate", "Validate agent directory structure and files")
        {
            agentOption
        };

        command.SetAction(async (parseResult) =>
        {
            var agentName = parseResult.GetValue(agentOption);

            try
            {
                using var loggerFactory = LoggerFactory.Create(builder =>
                {
                    builder.AddConsole();
                    builder.SetMinimumLevel(LogLevel.Warning);
                });

                var manager = new AgentManager(loggerFactory.CreateLogger<AgentManager>());

                if (!string.IsNullOrEmpty(agentName))
                {
                    // Validate specific agent
                    var result = await manager.ValidateAgentAsync(agentName);
                    DisplayValidationResult(result);
                    return result.IsValid ? 0 : 1;
                }
                else
                {
                    // Validate all agents
                    var statuses = await manager.CheckAgentStatusAsync();
                    var allValid = true;

                    foreach (var status in statuses)
                    {
                        var result = await manager.ValidateAgentAsync(status.Name);
                        DisplayValidationResult(result);
                        AnsiConsole.WriteLine();

                        if (!result.IsValid)
                            allValid = false;
                    }

                    return allValid ? 0 : 1;
                }
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
                return 1;
            }
        });

        return command;
    }

    private static string BuildStatusText(AgentStatus status)
    {
        if (status.HasError)
        {
            return $"[red]Error:[/] {status.ErrorMessage}";
        }

        var parts = new List<string>();

        if (status.UserModified)
        {
            parts.Add("[yellow]Modified[/]");
        }
        else
        {
            parts.Add("[green]OK[/]");
        }

        if (status.HasUpdate)
        {
            parts.Add("[cyan]Update available[/]");
        }

        return string.Join(" ", parts);
    }

    private static void DisplayValidationResult(MLoop.AIAgent.Infrastructure.ValidationResult result)
    {
        var icon = result.IsValid ? "[green]✓[/]" : "[red]✗[/]";
        AnsiConsole.MarkupLine($"{icon} [blue]{result.AgentName}[/]");

        if (result.Errors.Any())
        {
            AnsiConsole.MarkupLine("[red]  Errors:[/]");
            foreach (var error in result.Errors)
            {
                AnsiConsole.MarkupLine($"    [grey]•[/] {error}");
            }
        }

        if (result.Warnings.Any())
        {
            AnsiConsole.MarkupLine("[yellow]  Warnings:[/]");
            foreach (var warning in result.Warnings)
            {
                AnsiConsole.MarkupLine($"    [grey]•[/] {warning}");
            }
        }

        if (result.IsValid && !result.Warnings.Any())
        {
            AnsiConsole.MarkupLine("[green]  All checks passed[/]");
        }
    }
}
