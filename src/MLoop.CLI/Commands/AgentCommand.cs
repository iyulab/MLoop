using System.CommandLine;
using MLoop.AIAgent.Core;
using MLoop.AIAgent.Infrastructure;
using Ironbees.Core;
using Ironbees.AgentFramework;
using Microsoft.Extensions.Logging;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop agent - Conversational AI agent for ML project management
/// </summary>
public static class AgentCommand
{
    public static Command Create()
    {
        var queryArg = new Argument<string?>("query")
        {
            Description = "Your question or request to the AI agent (optional for interactive mode)",
            Arity = ArgumentArity.ZeroOrOne
        };

        var agentOption = new Option<string?>(
            "--agent",
            description: "Specific agent to use: data-analyst, preprocessing-expert, model-architect, mlops-manager");

        var interactiveOption = new Option<bool>(
            "--interactive",
            getDefaultValue: () => false,
            description: "Start interactive conversation mode");

        var projectPathOption = new Option<string?>(
            "--project",
            getDefaultValue: () => Directory.GetCurrentDirectory(),
            description: "Path to MLoop project directory");

        var command = new Command("agent", "Conversational AI agent for ML project management")
        {
            queryArg,
            agentOption,
            interactiveOption,
            projectPathOption
        };

        command.SetHandler(ExecuteAsync, queryArg, agentOption, interactiveOption, projectPathOption);

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? query,
        string? agentName,
        bool interactive,
        string? projectPath)
    {
        try
        {
            // Display welcome banner
            DisplayWelcomeBanner();

            // Initialize Ironbees orchestrator
            var orchestrator = await InitializeOrchestratorAsync(projectPath);

            // Determine mode: interactive or single query
            if (interactive || string.IsNullOrWhiteSpace(query))
            {
                return await RunInteractiveModeAsync(orchestrator, projectPath);
            }
            else
            {
                return await RunSingleQueryModeAsync(orchestrator, query, agentName);
            }
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return 1;
        }
    }

    private static void DisplayWelcomeBanner()
    {
        AnsiConsole.Write(
            new FigletText("MLoop Agent")
                .LeftJustified()
                .Color(Color.Blue));

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[blue]🤖 AI-Powered ML Project Assistant[/]");
        AnsiConsole.MarkupLine("[grey]Powered by Ironbees Agent Framework[/]");
        AnsiConsole.WriteLine();
    }

    private static async Task<IronbeesOrchestrator> InitializeOrchestratorAsync(string? projectPath)
    {
        IronbeesOrchestrator? orchestrator = null;

        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .SpinnerStyle(Style.Parse("blue"))
            .StartAsync("[blue]Initializing AI agents...[/]", async ctx =>
            {
                try
                {
                    // Determine agents directory
                    var agentsDirectory = !string.IsNullOrEmpty(projectPath)
                        ? Path.Combine(projectPath, ".mloop", "agents")
                        : Path.Combine(Directory.GetCurrentDirectory(), ".mloop", "agents");

                    // Create orchestrator using factory method
                    orchestrator = IronbeesOrchestrator.CreateFromEnvironment(
                        agentsDirectory: agentsDirectory);

                    // Load agents
                    await orchestrator.InitializeAsync();

                    ctx.Status("[green]AI agents loaded successfully[/]");
                }
                catch (Exception ex)
                {
                    ctx.Status($"[red]Failed to initialize agents: {ex.Message}[/]");
                    throw;
                }
            });

        if (orchestrator == null)
        {
            throw new InvalidOperationException("Failed to initialize orchestrator");
        }

        AnsiConsole.MarkupLine("[green]✓[/] AI agents loaded successfully");
        AnsiConsole.WriteLine();

        return orchestrator;
    }

    private static async Task<int> RunSingleQueryModeAsync(
        IronbeesOrchestrator orchestrator,
        string query,
        string? agentName)
    {
        try
        {
            AnsiConsole.MarkupLine($"[blue]❓ Query:[/] {query}");
            AnsiConsole.WriteLine();

            // Stream response
            await AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("blue"))
                .StartAsync("[blue]Agent thinking...[/]", async ctx =>
                {
                    var responseBuilder = new System.Text.StringBuilder();

                    if (!string.IsNullOrEmpty(agentName))
                    {
                        AnsiConsole.MarkupLine($"[grey]Using agent: {agentName}[/]");
                        await foreach (var chunk in orchestrator.StreamAsync(query, agentName))
                        {
                            responseBuilder.Append(chunk);
                        }
                    }
                    else
                    {
                        await foreach (var chunk in orchestrator.StreamWithAutoSelectionAsync(query))
                        {
                            responseBuilder.Append(chunk);
                        }
                    }

                    ctx.Status("[green]Response received[/]");

                    // Display response
                    AnsiConsole.WriteLine();
                    AnsiConsole.MarkupLine("[blue]🤖 Agent:[/]");
                    AnsiConsole.MarkupLine(responseBuilder.ToString());
                });

            AnsiConsole.WriteLine();
            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error processing query:[/] {ex.Message}");
            return 1;
        }
    }

    private static async Task<int> RunInteractiveModeAsync(
        IronbeesOrchestrator orchestrator,
        string? projectPath)
    {
        AnsiConsole.MarkupLine("[yellow]💬 Interactive mode[/]");
        AnsiConsole.MarkupLine("[grey]Type 'exit' or 'quit' to end conversation[/]");
        AnsiConsole.MarkupLine("[grey]Type '/agents' to list available agents[/]");
        AnsiConsole.MarkupLine("[grey]Type '/switch <agent-name>' to switch agent[/]");
        AnsiConsole.WriteLine();

        using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
        var conversationManager = new ConversationManager(
            loggerFactory.CreateLogger<ConversationManager>());

        if (!string.IsNullOrEmpty(projectPath))
        {
            conversationManager.SetProjectContext(projectPath);
        }

        string? currentAgent = null;

        while (true)
        {
            // Prompt for user input
            var agentIndicator = string.IsNullOrEmpty(currentAgent)
                ? "[grey](auto)[/]"
                : $"[blue]({currentAgent})[/]";

            AnsiConsole.Markup($"{agentIndicator} [green]You:[/] ");
            var userInput = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userInput))
            {
                continue;
            }

            // Check for exit commands
            if (userInput.Trim().ToLower() is "exit" or "quit")
            {
                AnsiConsole.MarkupLine("[yellow]👋 Goodbye![/]");
                break;
            }

            // Check for special commands
            if (userInput.StartsWith('/'))
            {
                HandleSpecialCommand(userInput, orchestrator, ref currentAgent);
                continue;
            }

            // Add to conversation history
            conversationManager.AddUserMessage(userInput);

            try
            {
                // Stream agent response
                AnsiConsole.Markup("[blue]🤖 Agent:[/] ");

                var responseBuilder = new System.Text.StringBuilder();

                if (!string.IsNullOrEmpty(currentAgent))
                {
                    await foreach (var chunk in orchestrator.StreamAsync(userInput, currentAgent))
                    {
                        AnsiConsole.Markup(chunk);
                        responseBuilder.Append(chunk);
                    }
                }
                else
                {
                    await foreach (var chunk in orchestrator.StreamWithAutoSelectionAsync(userInput))
                    {
                        AnsiConsole.Markup(chunk);
                        responseBuilder.Append(chunk);
                    }
                }

                // Add agent response to history
                conversationManager.AddAgentResponse(
                    currentAgent ?? "auto",
                    responseBuilder.ToString());

                AnsiConsole.WriteLine();
                AnsiConsole.WriteLine();
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
                AnsiConsole.WriteLine();
            }
        }

        return 0;
    }

    private static void HandleSpecialCommand(
        string command,
        IronbeesOrchestrator orchestrator,
        ref string? currentAgent)
    {
        var parts = command.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var cmd = parts[0].ToLower();

        switch (cmd)
        {
            case "/agents":
                DisplayAvailableAgents(orchestrator);
                break;

            case "/switch":
                if (parts.Length > 1)
                {
                    currentAgent = parts[1];
                    AnsiConsole.MarkupLine($"[green]✓[/] Switched to agent: [blue]{currentAgent}[/]");
                }
                else
                {
                    AnsiConsole.MarkupLine("[yellow]Usage:[/] /switch <agent-name>");
                }
                break;

            case "/auto":
                currentAgent = null;
                AnsiConsole.MarkupLine("[green]✓[/] Switched to auto agent selection");
                break;

            case "/help":
                DisplayInteractiveHelp();
                break;

            default:
                AnsiConsole.MarkupLine($"[yellow]Unknown command:[/] {cmd}");
                AnsiConsole.MarkupLine("[grey]Type /help for available commands[/]");
                break;
        }

        AnsiConsole.WriteLine();
    }

    private static void DisplayAvailableAgents(IronbeesOrchestrator orchestrator)
    {
        AnsiConsole.MarkupLine("[blue]📋 Available Agents:[/]");
        AnsiConsole.WriteLine();

        var agents = orchestrator.GetAvailableAgents();

        var table = new Table()
            .Border(TableBorder.Rounded)
            .AddColumn("[blue]Agent Name[/]")
            .AddColumn("[blue]Description[/]");

        foreach (var agent in agents)
        {
            var description = agent switch
            {
                "data-analyst" => "Analyzes dataset characteristics and provides insights",
                "preprocessing-expert" => "Generates preprocessing scripts for data cleaning",
                "model-architect" => "Recommends ML models based on task and data",
                "mlops-manager" => "Manages ML project lifecycle and operations",
                _ => "AI agent for MLoop tasks"
            };

            table.AddRow(agent, description);
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    private static void DisplayInteractiveHelp()
    {
        AnsiConsole.MarkupLine("[blue]💡 Interactive Mode Commands:[/]");
        AnsiConsole.WriteLine();

        var panel = new Panel(new Markup(
            "[grey]/agents[/]          - List available AI agents\n" +
            "[grey]/switch <name>[/]   - Switch to specific agent\n" +
            "[grey]/auto[/]            - Enable auto agent selection\n" +
            "[grey]/help[/]            - Show this help message\n" +
            "[grey]exit / quit[/]      - Exit interactive mode"))
        {
            Header = new PanelHeader("[blue]Commands[/]"),
            Border = BoxBorder.Rounded
        };

        AnsiConsole.Write(panel);
        AnsiConsole.WriteLine();
    }
}
