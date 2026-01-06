// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using System.CommandLine;
using Microsoft.Extensions.Logging;
using MLoop.AIAgent.Core.Orchestration;
using MLoop.AIAgent.Infrastructure;
using MLoop.CLI.Infrastructure.Configuration;
using Spectre.Console;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MLoop.CLI.Commands;

/// <summary>
/// CLI command for automated MLOps orchestration.
/// Provides end-to-end ML pipeline execution with HITL checkpoints.
/// </summary>
public static class OrchestrateCommand
{
    public static Command Create()
    {
        var dataFileArg = new Argument<string?>("data-file")
        {
            Description = "Path to the data file (CSV, Parquet, etc.)",
            Arity = ArgumentArity.ZeroOrOne
        };

        var targetOption = new Option<string?>("--target", "-t")
        {
            Description = "Target column name for prediction"
        };

        var taskTypeOption = new Option<string?>("--task-type")
        {
            Description = "ML task type (BinaryClassification, MulticlassClassification, Regression)"
        };

        var maxTimeOption = new Option<int?>("--max-training-time", "-m")
        {
            Description = "Maximum training time in seconds"
        };

        var skipHitlOption = new Option<bool>("--skip-hitl", "-y")
        {
            Description = "Skip all HITL checkpoints (auto-approve everything)"
        };

        var autoApproveOption = new Option<bool>("--auto-approve")
        {
            Description = "Auto-approve high confidence decisions"
        };

        var thresholdOption = new Option<double?>("--threshold")
        {
            Description = "Auto-approval confidence threshold (0.0-1.0)"
        };

        var resumeOption = new Option<string?>("--resume", "-r")
        {
            Description = "Resume a paused or active session by ID"
        };

        var listSessionsOption = new Option<bool>("--list-sessions", "-l")
        {
            Description = "List all orchestration sessions"
        };

        var verboseOption = new Option<bool>("--verbose", "-v")
        {
            Description = "Enable verbose output"
        };

        var command = new Command("orchestrate", "End-to-end MLOps automation with AI agents");
        command.Arguments.Add(dataFileArg);
        command.Options.Add(targetOption);
        command.Options.Add(taskTypeOption);
        command.Options.Add(maxTimeOption);
        command.Options.Add(skipHitlOption);
        command.Options.Add(autoApproveOption);
        command.Options.Add(thresholdOption);
        command.Options.Add(resumeOption);
        command.Options.Add(listSessionsOption);
        command.Options.Add(verboseOption);

        command.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg);
            var target = parseResult.GetValue(targetOption);
            var taskType = parseResult.GetValue(taskTypeOption);
            var maxTime = parseResult.GetValue(maxTimeOption);
            var skipHitl = parseResult.GetValue(skipHitlOption);
            var autoApprove = parseResult.GetValue(autoApproveOption);
            var threshold = parseResult.GetValue(thresholdOption);
            var resumeSessionId = parseResult.GetValue(resumeOption);
            var listSessions = parseResult.GetValue(listSessionsOption);
            var verbose = parseResult.GetValue(verboseOption);

            return ExecuteAsync(
                dataFile, target, taskType, maxTime,
                skipHitl, autoApprove, threshold,
                resumeSessionId, listSessions, verbose);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? dataFile,
        string? target,
        string? taskType,
        int? maxTime,
        bool skipHitl,
        bool autoApprove,
        double? threshold,
        string? resumeSessionId,
        bool listSessions,
        bool verbose)
    {
        try
        {
            // Initialize orchestrator
            var chatClient = CreateChatClient();
            if (chatClient == null)
            {
                return 1;
            }

            var store = OrchestrationSessionStore.ForProject(Directory.GetCurrentDirectory());
            var orchestrator = new MLOpsOrchestratorService(chatClient, store);

            // Handle --list-sessions
            if (listSessions)
            {
                await ListSessionsAsync(orchestrator);
                return 0;
            }

            // Handle --resume
            if (!string.IsNullOrEmpty(resumeSessionId))
            {
                return await ResumeSessionAsync(orchestrator, resumeSessionId, verbose);
            }

            // Validate data file for new orchestration
            if (string.IsNullOrEmpty(dataFile) || !File.Exists(dataFile))
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Data file is required. Use [yellow]mloop orchestrate <data-file>[/]");
                AnsiConsole.MarkupLine("       Or use [yellow]--list-sessions[/] to see existing sessions");
                AnsiConsole.MarkupLine("       Or use [yellow]--resume <session-id>[/] to resume a session");
                return 1;
            }

            // Load project config (mloop.yaml) for fallback values
            var (configLabelColumn, configTaskType) = LoadProjectConfig();

            // Use CLI options first, then fall back to mloop.yaml config
            var effectiveTarget = target ?? configLabelColumn;
            var effectiveTaskType = taskType ?? configTaskType;

            if (!string.IsNullOrEmpty(configLabelColumn) && string.IsNullOrEmpty(target))
            {
                AnsiConsole.MarkupLine($"[grey]Using label column from mloop.yaml:[/] [blue]{configLabelColumn}[/]");
            }

            if (!string.IsNullOrEmpty(configTaskType) && string.IsNullOrEmpty(taskType))
            {
                AnsiConsole.MarkupLine($"[grey]Using task type from mloop.yaml:[/] [blue]{configTaskType}[/]");
            }

            // Build options
            var options = new OrchestrationOptions
            {
                TargetColumn = effectiveTarget,
                TaskType = effectiveTaskType,
                MaxTrainingTimeSeconds = maxTime,
                SkipHitl = skipHitl,
                AutoApproveHighConfidence = autoApprove,
                AutoApprovalThreshold = threshold ?? 0.85
            };

            // Execute orchestration
            return await ExecuteOrchestrationAsync(
                orchestrator,
                dataFile,
                options,
                verbose);
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            if (verbose)
            {
                AnsiConsole.WriteException(ex);
            }
            return 1;
        }
    }

    private static async Task ListSessionsAsync(MLOpsOrchestratorService orchestrator)
    {
        AnsiConsole.MarkupLine("[blue]MLOps Orchestration Sessions[/]");
        AnsiConsole.WriteLine();

        var sessions = await orchestrator.ListSessionsAsync();
        if (sessions.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]No orchestration sessions found.[/]");
            return;
        }

        var table = new Table();
        table.AddColumn("Session ID");
        table.AddColumn("Status");
        table.AddColumn("State");
        table.AddColumn("Data File");
        table.AddColumn("Updated");

        foreach (var session in sessions)
        {
            var statusColor = session.Status switch
            {
                SessionStatus.Active => "green",
                SessionStatus.Paused => "yellow",
                SessionStatus.Completed => "blue",
                SessionStatus.Failed => "red",
                SessionStatus.Cancelled => "grey",
                _ => "white"
            };

            table.AddRow(
                session.SessionId,
                $"[{statusColor}]{session.Status}[/]",
                session.State.ToString(),
                TruncatePath(session.DataFilePath, 30),
                session.UpdatedAt.ToString("g"));
        }

        AnsiConsole.Write(table);

        // Show resumable sessions
        var resumable = await orchestrator.GetResumableSessionsAsync();
        if (resumable.Count > 0)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[yellow]Resumable sessions:[/] {resumable.Count}");
            AnsiConsole.MarkupLine("Use [blue]mloop orchestrate --resume <session-id>[/] to continue");
        }
    }

    private static async Task<int> ResumeSessionAsync(
        MLOpsOrchestratorService orchestrator,
        string sessionId,
        bool verbose)
    {
        AnsiConsole.MarkupLine($"[blue]Resuming session:[/] {sessionId}");
        AnsiConsole.WriteLine();

        await foreach (var evt in orchestrator.ResumeAsync(sessionId))
        {
            HandleOrchestrationEvent(evt, verbose);

            if (evt is OrchestrationCompletedEvent or OrchestrationFailedEvent)
            {
                return evt is OrchestrationCompletedEvent ? 0 : 1;
            }

            // Handle HITL events
            if (evt is HitlRequestedEvent hitlRequest)
            {
                await PromptHitlDecisionAsync(hitlRequest);
            }
        }

        return 0;
    }

    private static async Task<int> ExecuteOrchestrationAsync(
        MLOpsOrchestratorService orchestrator,
        string dataFilePath,
        OrchestrationOptions options,
        bool verbose)
    {
        // Display header
        var panel = new Panel(new Markup($"[bold blue]MLOps Orchestrator[/]\n\nData: [yellow]{Path.GetFileName(dataFilePath)}[/]"))
        {
            Border = BoxBorder.Rounded,
            Padding = new Padding(2, 1)
        };
        AnsiConsole.Write(panel);
        AnsiConsole.WriteLine();

        var exitCode = 0;

        await foreach (var evt in orchestrator.ExecuteAsync(dataFilePath, options))
        {
            HandleOrchestrationEvent(evt, verbose);

            if (evt is OrchestrationCompletedEvent completed)
            {
                DisplayCompletionSummary(completed);
                exitCode = 0;
            }
            else if (evt is OrchestrationFailedEvent failed)
            {
                AnsiConsole.MarkupLine($"[red]Orchestration failed:[/] {failed.Error}");
                exitCode = 1;
            }
            else if (evt is HitlRequestedEvent hitlRequest)
            {
                // Auto-approve if --auto-approve flag is set (fixes non-interactive terminal issue)
                if (options.AutoApproveHighConfidence)
                {
                    AnsiConsole.MarkupLine($"  [yellow]Auto-approved:[/] {hitlRequest.CheckpointName}");
                }
                else
                {
                    await PromptHitlDecisionAsync(hitlRequest);
                }
            }
        }

        return exitCode;
    }

    private static void HandleOrchestrationEvent(OrchestrationEvent evt, bool verbose)
    {
        switch (evt)
        {
            case OrchestrationStartedEvent started:
                AnsiConsole.MarkupLine($"[green]Started[/] Session: [yellow]{started.SessionId}[/]");
                break;

            case StateChangedEvent stateChanged:
                AnsiConsole.MarkupLine($"  -> [blue]{stateChanged.FromState}[/] -> [green]{stateChanged.ToState}[/]");
                break;

            case PhaseStartedEvent phaseStarted:
                AnsiConsole.MarkupLine($"\n[bold cyan]Phase {phaseStarted.PhaseNumber}:[/] {phaseStarted.PhaseName}");
                break;

            case PhaseCompletedEvent phaseCompleted:
                AnsiConsole.MarkupLine($"  [green]OK[/] {phaseCompleted.PhaseName} completed ({phaseCompleted.Duration:mm\\:ss})");
                break;

            case AgentStartedEvent agentStarted:
                AnsiConsole.MarkupLine($"  [dim]Agent:[/] {agentStarted.AgentType}");
                break;

            case AgentCompletedEvent agentCompleted:
                if (verbose)
                {
                    AnsiConsole.MarkupLine($"    [grey]Agent {agentCompleted.AgentType} finished ({agentCompleted.Duration:ss\\.fff}s)[/]");
                }
                break;

            case HitlRequestedEvent hitlRequest:
                DisplayHitlRequest(hitlRequest);
                break;

            case HitlResponseReceivedEvent hitlResponse:
                AnsiConsole.MarkupLine($"  [yellow]Decision:[/] {hitlResponse.SelectedOptionId}");
                break;

            case ProgressUpdateEvent progress:
                if (verbose)
                {
                    AnsiConsole.MarkupLine($"  [grey]{progress.CurrentOperation} ({progress.Percentage}%)[/]");
                }
                break;

            case OrchestrationFailedEvent:
                // Handled separately
                break;

            case OrchestrationCompletedEvent:
                // Handled separately
                break;

            default:
                if (verbose)
                {
                    AnsiConsole.MarkupLine($"[grey]Event: {evt.GetType().Name}[/]");
                }
                break;
        }
    }

    private static void DisplayHitlRequest(HitlRequestedEvent request)
    {
        AnsiConsole.WriteLine();

        var panel = new Panel(new Markup($"[bold yellow]HITL Checkpoint:[/] {request.CheckpointName}\n\n{request.Question}"))
        {
            Border = BoxBorder.Double,
            Padding = new Padding(2, 1),
            Header = new PanelHeader(" Human Review Required ")
        };
        AnsiConsole.Write(panel);

        // Display context if available
        if (request.Context != null && request.Context.Count > 0)
        {
            AnsiConsole.MarkupLine("\n[dim]Context:[/]");
            foreach (var (key, value) in request.Context.Take(5))
            {
                AnsiConsole.MarkupLine($"  [grey]{key}:[/] {value}");
            }
        }
    }

    private static async Task<string> PromptHitlDecisionAsync(HitlRequestedEvent request)
    {
        AnsiConsole.WriteLine();

        var choices = request.Options.Select(o => o.Label).ToList();
        var selection = AnsiConsole.Prompt(
            new SelectionPrompt<string>()
                .Title("[yellow]Select your decision:[/]")
                .AddChoices(choices));

        var selectedOption = request.Options.FirstOrDefault(o => o.Label == selection);

        await Task.CompletedTask; // Placeholder for future async callback
        return selectedOption?.Id ?? "approve";
    }

    private static void DisplayCompletionSummary(OrchestrationCompletedEvent completed)
    {
        AnsiConsole.WriteLine();

        var summaryPanel = new Panel(new Markup(
            $"""
            [bold green]Orchestration Completed[/]

            Session: [yellow]{completed.SessionId}[/]
            Duration: [cyan]{completed.TotalDuration:hh\:mm\:ss}[/]

            [dim]All phases completed successfully.[/]
            """))
        {
            Border = BoxBorder.Rounded,
            Padding = new Padding(2, 1),
            Header = new PanelHeader(" Summary ")
        };

        AnsiConsole.Write(summaryPanel);
    }

    private static Microsoft.Extensions.AI.IChatClient? CreateChatClient()
    {
        try
        {
            // Use IronbeesOrchestrator factory to create chat client with full middleware
            using var loggerFactory = LoggerFactory.Create(builder =>
                builder.SetMinimumLevel(LogLevel.Warning));

            var orchestrator = IronbeesOrchestrator.CreateFromEnvironment(loggerFactory);
            return orchestrator.ChatClient;
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("LLM provider"))
        {
            AnsiConsole.MarkupLine("[yellow]Warning:[/] No AI provider configured.");
            AnsiConsole.MarkupLine("Set one of the following environment variable combinations:");
            AnsiConsole.MarkupLine("  [blue]AZURE_OPENAI_ENDPOINT[/] + [blue]AZURE_OPENAI_KEY[/] + [blue]AZURE_OPENAI_MODEL[/]");
            AnsiConsole.MarkupLine("  [blue]OPENAI_API_KEY[/] + [blue]OPENAI_MODEL[/]");
            return null;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error creating chat client:[/] {ex.Message}");
            return null;
        }
    }

    private static string TruncatePath(string path, int maxLength)
    {
        if (string.IsNullOrEmpty(path)) return string.Empty;
        if (path.Length <= maxLength) return path;

        var fileName = Path.GetFileName(path);
        if (fileName.Length >= maxLength - 3)
        {
            return "..." + fileName[^(maxLength - 3)..];
        }

        return "..." + path[^(maxLength - 3)..];
    }

    /// <summary>
    /// Loads mloop.yaml configuration from the current directory.
    /// Returns the default model's label and task type if available.
    /// </summary>
    private static (string? LabelColumn, string? TaskType) LoadProjectConfig()
    {
        var yamlPath = Path.Combine(Directory.GetCurrentDirectory(), "mloop.yaml");

        if (!File.Exists(yamlPath))
        {
            return (null, null);
        }

        try
        {
            var yamlContent = File.ReadAllText(yamlPath);

            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .IgnoreUnmatchedProperties()
                .Build();

            // Try loading as new multi-model format
            var config = deserializer.Deserialize<MLoopConfig>(yamlContent);

            if (config?.Models != null && config.Models.TryGetValue(ConfigDefaults.DefaultModelName, out var defaultModel))
            {
                return (defaultModel.Label, defaultModel.Task);
            }

            // Try loading as old single-model format
            var oldConfig = deserializer.Deserialize<Dictionary<string, object?>>(yamlContent);
            if (oldConfig != null)
            {
                var labelColumn = oldConfig.TryGetValue("label_column", out var label) ? label?.ToString() : null;
                var taskType = oldConfig.TryGetValue("task", out var task) ? task?.ToString() : null;
                return (labelColumn, taskType);
            }

            return (null, null);
        }
        catch (Exception)
        {
            // Silently ignore config loading errors
            return (null, null);
        }
    }
}
