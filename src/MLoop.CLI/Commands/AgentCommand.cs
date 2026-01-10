using System.CommandLine;
using System.Text;
using Microsoft.Extensions.Logging;
using MLoop.AIAgent.Core;
using MLoop.AIAgent.Infrastructure;
using MLoop.AIAgent.Tools;
using MLoop.CLI.Infrastructure.Configuration;
using Spectre.Console;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

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

        var agentOption = new Option<string?>("--agent", "-a")
        {
            Description = "Specific agent to use: data-analyst, preprocessing-expert, model-architect, mlops-manager"
        };

        var interactiveOption = new Option<bool>("--interactive", "-i")
        {
            Description = "Start interactive conversation mode",
            DefaultValueFactory = _ => false
        };

        var listAgentsOption = new Option<bool>("--list-agents", "-l")
        {
            Description = "List available AI agents and exit",
            DefaultValueFactory = _ => false
        };

        var projectPathOption = new Option<string?>("--project", "-p")
        {
            Description = "Path to MLoop project directory",
            DefaultValueFactory = _ => Directory.GetCurrentDirectory()
        };

        var command = new Command("agent", "Conversational AI agent for ML project management");
        command.Arguments.Add(queryArg);
        command.Options.Add(agentOption);
        command.Options.Add(interactiveOption);
        command.Options.Add(listAgentsOption);
        command.Options.Add(projectPathOption);

        // Add workflow subcommand
        command.Subcommands.Add(AgentWorkflowCommand.Create());

        command.SetAction((parseResult) =>
        {
            var query = parseResult.GetValue(queryArg);
            var agentName = parseResult.GetValue(agentOption);
            var interactive = parseResult.GetValue(interactiveOption);
            var listAgents = parseResult.GetValue(listAgentsOption);
            var projectPath = parseResult.GetValue(projectPathOption);
            return ExecuteAsync(query, agentName, interactive, listAgents, projectPath);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? query,
        string? agentName,
        bool interactive,
        bool listAgents,
        string? projectPath)
    {
        try
        {
            // Handle --list-agents first (before full initialization)
            if (listAgents)
            {
                return await ListAvailableAgentsAsync(projectPath);
            }

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
                return await RunSingleQueryModeAsync(orchestrator, query, agentName, projectPath);
            }
        }
        catch (Exception ex)
        {
            // Escape markup characters in error message
            var errorMessage = ex.Message.Replace("[", "[[").Replace("]", "]]");
            AnsiConsole.MarkupLine($"[red]Error:[/] {errorMessage}");
            AnsiConsole.WriteException(ex);
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
        var effectiveProjectPath = projectPath ?? Directory.GetCurrentDirectory();

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

                    // Register MLOps tools for function calling (fixes hallucination issue)
                    ctx.Status("[blue]Registering MLOps tools...[/]");
                    var projectManager = new MLoopProjectManager();
                    var mlopsTools = new MLOpsTools(projectManager, effectiveProjectPath);
                    orchestrator.RegisterTools(mlopsTools.CreateTools());

                    ctx.Status("[green]AI agents loaded successfully[/]");
                }
                catch (InvalidOperationException ex) when (ex.Message.Contains("LLM provider"))
                {
                    ctx.Status("[red]LLM provider not configured[/]");
                    throw new InvalidOperationException(
                        "No LLM provider configured. Please set environment variables.\n" +
                        "Options:\n" +
                        "  GPUSTACK_ENDPOINT + GPUSTACK_API_KEY + GPUSTACK_MODEL\n" +
                        "  AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY + AZURE_OPENAI_MODEL\n" +
                        "  OPENAI_API_KEY + OPENAI_MODEL\n\n" +
                        "Or create a .env file in the project directory.", ex);
                }
                catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.NotFound)
                {
                    ctx.Status("[red]LLM endpoint not reachable[/]");
                    throw new InvalidOperationException(
                        $"LLM endpoint returned 404 (Not Found).\n" +
                        "Possible causes:\n" +
                        "  - Incorrect endpoint URL\n" +
                        "  - Model name not found on server\n" +
                        "  - API version mismatch\n\n" +
                        "Please verify your LLM configuration in .env file.", ex);
                }
                catch (HttpRequestException ex)
                {
                    ctx.Status("[red]LLM connection failed[/]");
                    throw new InvalidOperationException(
                        $"Failed to connect to LLM provider: {ex.Message}\n" +
                        "Please check:\n" +
                        "  - Network connectivity\n" +
                        "  - Endpoint URL is correct\n" +
                        "  - API key is valid", ex);
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
        AnsiConsole.MarkupLine("[green]✓[/] MLOps tools registered for function calling");
        AnsiConsole.WriteLine();

        return orchestrator;
    }

    private static async Task<int> RunSingleQueryModeAsync(
        IronbeesOrchestrator orchestrator,
        string query,
        string? agentName,
        string? projectPath = null)
    {
        try
        {
            // Load project context for auto-injection
            var effectivePath = projectPath ?? Directory.GetCurrentDirectory();
            var projectContext = LoadProjectContext(effectivePath);

            // Display context info if available
            if (projectContext.HasMLoopConfig || projectContext.DataFiles.Count > 0)
            {
                AnsiConsole.MarkupLine("[grey]📁 Project context auto-detected[/]");
                if (projectContext.HasMLoopConfig)
                {
                    AnsiConsole.MarkupLine($"[grey]   Project: {projectContext.ProjectName ?? "Unnamed"}[/]");
                    if (!string.IsNullOrEmpty(projectContext.LabelColumn))
                        AnsiConsole.MarkupLine($"[grey]   Label: {projectContext.LabelColumn}[/]");
                }
                if (projectContext.DataFiles.Count > 0)
                {
                    AnsiConsole.MarkupLine($"[grey]   Data files: {projectContext.DataFiles.Count} found[/]");
                }
                AnsiConsole.WriteLine();
            }

            AnsiConsole.MarkupLine($"[blue]❓ Query:[/] {query}");
            AnsiConsole.WriteLine();

            // Enhance query with project context
            var enhancedQuery = EnhanceQueryWithContext(query, projectContext);

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
                        await foreach (var chunk in orchestrator.StreamAsync(enhancedQuery, agentName))
                        {
                            responseBuilder.Append(chunk);
                        }
                    }
                    else
                    {
                        await foreach (var chunk in orchestrator.StreamWithAutoSelectionAsync(enhancedQuery))
                        {
                            responseBuilder.Append(chunk);
                        }
                    }

                    ctx.Status("[green]Response received[/]");

                    // Display response
                    AnsiConsole.WriteLine();
                    AnsiConsole.MarkupLine("[blue]🤖 Agent:[/]");

                    // Escape markup characters in agent response to prevent parsing errors
                    var response = responseBuilder.ToString()
                        .Replace("[", "[[")
                        .Replace("]", "]]");
                    AnsiConsole.MarkupLine(response);
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
        // Load project context for auto-injection
        var effectivePath = projectPath ?? Directory.GetCurrentDirectory();
        var projectContext = LoadProjectContext(effectivePath);

        AnsiConsole.MarkupLine("[yellow]💬 Interactive mode[/]");

        // Display auto-detected context
        if (projectContext.HasMLoopConfig || projectContext.DataFiles.Count > 0)
        {
            AnsiConsole.MarkupLine("[grey]📁 Project context auto-detected[/]");
            if (projectContext.HasMLoopConfig)
            {
                AnsiConsole.MarkupLine($"[grey]   Project: {projectContext.ProjectName ?? "Unnamed"}[/]");
                if (!string.IsNullOrEmpty(projectContext.LabelColumn))
                    AnsiConsole.MarkupLine($"[grey]   Label: {projectContext.LabelColumn}[/]");
                if (!string.IsNullOrEmpty(projectContext.TaskType))
                    AnsiConsole.MarkupLine($"[grey]   Task: {projectContext.TaskType}[/]");
            }
            if (projectContext.DataFiles.Count > 0)
            {
                AnsiConsole.MarkupLine($"[grey]   Data files: {projectContext.DataFiles.Count} found[/]");
            }
            if (projectContext.RecentExperiments.Count > 0)
            {
                AnsiConsole.MarkupLine($"[grey]   Experiments: {string.Join(", ", projectContext.RecentExperiments.Take(3))}[/]");
            }
        }

        AnsiConsole.MarkupLine("[grey]Type 'exit' or 'quit' to end conversation[/]");
        AnsiConsole.MarkupLine("[grey]Type '/agents' to list available agents[/]");
        AnsiConsole.MarkupLine("[grey]Type '/switch <agent-name>' to switch agent[/]");
        AnsiConsole.MarkupLine("[grey]Type '/context' to show project context[/]");
        AnsiConsole.WriteLine();

        // Initialize conversation service with file-based persistence
        var conversationsDir = !string.IsNullOrEmpty(projectPath)
            ? Path.Combine(projectPath, ".mloop", "conversations")
            : Path.Combine(Directory.GetCurrentDirectory(), ".mloop", "conversations");

        using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
        using var conversationService = new ConversationService(
            conversationsDir,
            loggerFactory.CreateLogger<ConversationService>());

        // Start a new conversation with unique ID
        var conversationId = $"session_{DateTime.UtcNow:yyyyMMdd_HHmmss}";
        await conversationService.StartOrResumeAsync(conversationId);

        if (!string.IsNullOrEmpty(projectPath))
        {
            conversationService.SetProjectContext(projectPath);
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
                // Save conversation before exit
                await conversationService.SaveCurrentAsync();
                AnsiConsole.MarkupLine("[yellow]👋 Goodbye! Conversation saved.[/]");
                break;
            }

            // Check for special commands
            if (userInput.StartsWith('/'))
            {
                HandleSpecialCommand(userInput, orchestrator, ref currentAgent, projectContext);
                continue;
            }

            // Add to conversation history
            conversationService.AddUserMessage(userInput);

            // Enhance query with project context
            var enhancedQuery = EnhanceQueryWithContext(userInput, projectContext);

            try
            {
                // Stream agent response
                AnsiConsole.Markup("[blue]🤖 Agent:[/] ");

                var responseBuilder = new System.Text.StringBuilder();

                if (!string.IsNullOrEmpty(currentAgent))
                {
                    await foreach (var chunk in orchestrator.StreamAsync(enhancedQuery, currentAgent))
                    {
                        // Escape markup characters in streamed chunks
                        var escapedChunk = chunk.Replace("[", "[[").Replace("]", "]]");
                        AnsiConsole.Markup(escapedChunk);
                        responseBuilder.Append(chunk);
                    }
                }
                else
                {
                    await foreach (var chunk in orchestrator.StreamWithAutoSelectionAsync(enhancedQuery))
                    {
                        // Escape markup characters in streamed chunks
                        var escapedChunk = chunk.Replace("[", "[[").Replace("]", "]]");
                        AnsiConsole.Markup(escapedChunk);
                        responseBuilder.Append(chunk);
                    }
                }

                // Add agent response to history
                conversationService.AddAgentResponse(responseBuilder.ToString());

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
        ref string? currentAgent,
        ProjectContext? projectContext = null)
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

            case "/context":
                DisplayProjectContext(projectContext);
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

    private static void DisplayProjectContext(ProjectContext? context)
    {
        if (context == null)
        {
            AnsiConsole.MarkupLine("[yellow]No project context available[/]");
            return;
        }

        AnsiConsole.MarkupLine("[blue]📁 Project Context[/]");
        AnsiConsole.WriteLine();

        var table = new Table()
            .Border(TableBorder.Rounded)
            .AddColumn("[blue]Property[/]")
            .AddColumn("[blue]Value[/]");

        table.AddRow("Project Path", context.ProjectPath);
        table.AddRow("Project Name", context.ProjectName ?? "[grey]Not set[/]");
        table.AddRow("MLoop Config", context.HasMLoopConfig ? "[green]Found[/]" : "[grey]Not found[/]");
        table.AddRow("Label Column", context.LabelColumn ?? "[grey]Not set[/]");
        table.AddRow("Task Type", context.TaskType ?? "[grey]Not set[/]");

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();

        if (context.DataFiles.Count > 0)
        {
            AnsiConsole.MarkupLine($"[blue]📊 Data Files ({context.DataFiles.Count}):[/]");
            var dataTable = new Table()
                .Border(TableBorder.Simple)
                .AddColumn("File")
                .AddColumn("Size");

            foreach (var file in context.DataFiles)
            {
                var sizeKb = file.SizeBytes / 1024.0;
                dataTable.AddRow(file.RelativePath, $"{sizeKb:F1} KB");
            }

            AnsiConsole.Write(dataTable);
            AnsiConsole.WriteLine();
        }

        if (context.RecentExperiments.Count > 0)
        {
            AnsiConsole.MarkupLine($"[blue]🧪 Recent Experiments:[/]");
            foreach (var exp in context.RecentExperiments)
            {
                AnsiConsole.MarkupLine($"  - {exp}");
            }
        }
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
            "[grey]/context[/]         - Show auto-detected project context\n" +
            "[grey]/help[/]            - Show this help message\n" +
            "[grey]exit / quit[/]      - Exit interactive mode"))
        {
            Header = new PanelHeader("[blue]Commands[/]"),
            Border = BoxBorder.Rounded
        };

        AnsiConsole.Write(panel);
        AnsiConsole.WriteLine();
    }

    private static async Task<int> ListAvailableAgentsAsync(string? projectPath)
    {
        try
        {
            AnsiConsole.MarkupLine("[blue]🤖 MLoop AI Agents[/]");
            AnsiConsole.WriteLine();

            // Initialize orchestrator to get agent list
            IronbeesOrchestrator? orchestrator = null;

            await AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("blue"))
                .StartAsync("[blue]Loading agents...[/]", async ctx =>
                {
                    var agentsDirectory = !string.IsNullOrEmpty(projectPath)
                        ? Path.Combine(projectPath, ".mloop", "agents")
                        : Path.Combine(Directory.GetCurrentDirectory(), ".mloop", "agents");

                    orchestrator = IronbeesOrchestrator.CreateFromEnvironment(agentsDirectory: agentsDirectory);
                    await orchestrator.InitializeAsync();
                });

            if (orchestrator == null)
            {
                AnsiConsole.MarkupLine("[red]Failed to initialize agents[/]");
                return 1;
            }

            var agents = orchestrator.GetAvailableAgents();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .AddColumn("[blue]Agent Name[/]")
                .AddColumn("[blue]Description[/]")
                .AddColumn("[blue]Use Case[/]");

            foreach (var agent in agents)
            {
                var (description, useCase) = agent switch
                {
                    "data-analyst" => ("Analyzes dataset characteristics and provides insights",
                                       "Dataset analysis, ML readiness assessment"),
                    "preprocessing-expert" => ("Generates preprocessing scripts for data cleaning",
                                               "C# script generation, data transformation"),
                    "model-architect" => ("Recommends ML models based on task and data",
                                          "Problem classification, AutoML configuration"),
                    "mlops-manager" => ("Manages ML project lifecycle and operations",
                                        "Workflow orchestration, training execution"),
                    _ => ("AI agent for MLoop tasks", "General assistance")
                };

                table.AddRow(agent, description, useCase);
            }

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[grey]Total: {agents.Count()} agent(s) available[/]");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[grey]Usage: mloop agent \"your question\" --agent <agent-name>[/]");
            AnsiConsole.MarkupLine("[grey]       mloop agent --interactive[/]");

            return 0;
        }
        catch (Exception ex)
        {
            var errorMessage = ex.Message.Replace("[", "[[").Replace("]", "]]");
            AnsiConsole.MarkupLine($"[red]Error:[/] {errorMessage}");
            return 1;
        }
    }

    /// <summary>
    /// Loads project context from mloop.yaml and scans for data files.
    /// This context is injected into agent queries for automatic project awareness.
    /// </summary>
    private static ProjectContext LoadProjectContext(string projectPath)
    {
        var context = new ProjectContext { ProjectPath = projectPath };

        // 1. Load mloop.yaml configuration
        var yamlPath = Path.Combine(projectPath, "mloop.yaml");
        if (File.Exists(yamlPath))
        {
            try
            {
                var yamlContent = File.ReadAllText(yamlPath);
                var deserializer = new DeserializerBuilder()
                    .WithNamingConvention(UnderscoredNamingConvention.Instance)
                    .IgnoreUnmatchedProperties()
                    .Build();

                var config = deserializer.Deserialize<MLoopConfig>(yamlContent);
                if (config?.Models != null && config.Models.TryGetValue(ConfigDefaults.DefaultModelName, out var defaultModel))
                {
                    context.LabelColumn = defaultModel.Label;
                    context.TaskType = defaultModel.Task;
                    context.ProjectName = config.Project;
                }
                else
                {
                    // Try old format
                    var oldConfig = deserializer.Deserialize<Dictionary<string, object?>>(yamlContent);
                    if (oldConfig != null)
                    {
                        context.LabelColumn = oldConfig.TryGetValue("label_column", out var label) ? label?.ToString() : null;
                        context.TaskType = oldConfig.TryGetValue("task", out var task) ? task?.ToString() : null;
                        context.ProjectName = oldConfig.TryGetValue("name", out var name) ? name?.ToString() : null;
                    }
                }

                context.HasMLoopConfig = true;
            }
            catch
            {
                // Ignore YAML parse errors
            }
        }

        // 2. Scan for data files
        var dataDirectories = new[] { "data", "datasets", "data/raw", "data/processed" };
        foreach (var dir in dataDirectories)
        {
            var fullPath = Path.Combine(projectPath, dir);
            if (Directory.Exists(fullPath))
            {
                var csvFiles = Directory.GetFiles(fullPath, "*.csv", SearchOption.TopDirectoryOnly);
                foreach (var file in csvFiles.Take(10)) // Limit to first 10 files
                {
                    var relativePath = Path.GetRelativePath(projectPath, file);
                    var fileInfo = new FileInfo(file);
                    context.DataFiles.Add(new DataFileInfo
                    {
                        RelativePath = relativePath,
                        FileName = fileInfo.Name,
                        SizeBytes = fileInfo.Length
                    });
                }
            }
        }

        // 3. Check for experiments
        var experimentsDir = Path.Combine(projectPath, ".mloop", "experiments");
        if (Directory.Exists(experimentsDir))
        {
            var experiments = Directory.GetDirectories(experimentsDir)
                .Select(Path.GetFileName)
                .Where(n => n != null)
                .Cast<string>()
                .OrderByDescending(n => n)
                .Take(5)
                .ToList();

            context.RecentExperiments = experiments;
        }

        return context;
    }

    /// <summary>
    /// Formats project context as a prefix for user queries.
    /// </summary>
    private static string FormatContextPrefix(ProjectContext context)
    {
        if (!context.HasMLoopConfig && context.DataFiles.Count == 0 && context.RecentExperiments.Count == 0)
        {
            return string.Empty;
        }

        var sb = new StringBuilder();
        sb.AppendLine("[Project Context - Auto-detected]");

        if (context.HasMLoopConfig)
        {
            sb.AppendLine($"Project: {context.ProjectName ?? "Unnamed"}");
            if (!string.IsNullOrEmpty(context.LabelColumn))
                sb.AppendLine($"Label Column: {context.LabelColumn}");
            if (!string.IsNullOrEmpty(context.TaskType))
                sb.AppendLine($"Task Type: {context.TaskType}");
        }

        if (context.DataFiles.Count > 0)
        {
            sb.AppendLine($"Data Files ({context.DataFiles.Count} found):");
            foreach (var file in context.DataFiles.Take(5))
            {
                var sizeKb = file.SizeBytes / 1024.0;
                sb.AppendLine($"  - {file.RelativePath} ({sizeKb:F1} KB)");
            }
            if (context.DataFiles.Count > 5)
                sb.AppendLine($"  ... and {context.DataFiles.Count - 5} more files");
        }

        if (context.RecentExperiments.Count > 0)
        {
            sb.AppendLine($"Recent Experiments: {string.Join(", ", context.RecentExperiments)}");
        }

        sb.AppendLine("[End Context]");
        sb.AppendLine();
        sb.AppendLine("User Query:");

        return sb.ToString();
    }

    /// <summary>
    /// Enhances a user query with project context.
    /// </summary>
    private static string EnhanceQueryWithContext(string query, ProjectContext context)
    {
        var prefix = FormatContextPrefix(context);
        if (string.IsNullOrEmpty(prefix))
            return query;

        return prefix + query;
    }

    private class ProjectContext
    {
        public string ProjectPath { get; set; } = string.Empty;
        public string? ProjectName { get; set; }
        public string? LabelColumn { get; set; }
        public string? TaskType { get; set; }
        public bool HasMLoopConfig { get; set; }
        public List<DataFileInfo> DataFiles { get; set; } = [];
        public List<string> RecentExperiments { get; set; } = [];
    }

    private class DataFileInfo
    {
        public string RelativePath { get; set; } = string.Empty;
        public string FileName { get; set; } = string.Empty;
        public long SizeBytes { get; set; }
    }
}
