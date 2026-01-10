using System.CommandLine;
using Ironbees.AgentMode.Workflow;
using Ironbees.AgentMode.Models;
using Microsoft.Extensions.Logging;
using MLoop.AIAgent.Infrastructure;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop agent workflow - Execute and manage AI agent workflows
/// </summary>
public static class AgentWorkflowCommand
{
    public static Command Create()
    {
        var command = new Command("workflow", "Execute and manage AI agent workflows");

        // Subcommands
        command.Subcommands.Add(CreateRunCommand());
        command.Subcommands.Add(CreateListCommand());
        command.Subcommands.Add(CreateValidateCommand());
        command.Subcommands.Add(CreateStatusCommand());

        return command;
    }

    #region Run Subcommand

    private static Command CreateRunCommand()
    {
        var workflowArg = new Argument<string>("workflow-file")
        {
            Description = "Path to workflow YAML file"
        };

        var inputOption = new Option<string?>("--input", "-i")
        {
            Description = "Input message for the workflow"
        };

        var projectOption = new Option<string?>("--project", "-p")
        {
            Description = "Path to MLoop project directory",
            DefaultValueFactory = _ => Directory.GetCurrentDirectory()
        };

        var command = new Command("run", "Execute a workflow from YAML file");
        command.Arguments.Add(workflowArg);
        command.Options.Add(inputOption);
        command.Options.Add(projectOption);

        command.SetAction(async (parseResult) =>
        {
            var workflowFile = parseResult.GetValue(workflowArg)!;
            var input = parseResult.GetValue(inputOption) ?? "Start workflow execution";
            var projectPath = parseResult.GetValue(projectOption);
            return await ExecuteRunAsync(workflowFile, input, projectPath);
        });

        return command;
    }

    private static async Task<int> ExecuteRunAsync(string workflowFile, string input, string? projectPath)
    {
        try
        {
            // Validate workflow file exists
            if (!File.Exists(workflowFile))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Workflow file not found: {workflowFile}");
                return 1;
            }

            AnsiConsole.Write(
                new FigletText("Workflow")
                    .LeftJustified()
                    .Color(Color.Blue));

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[blue]Workflow:[/] {Path.GetFileName(workflowFile)}");
            AnsiConsole.MarkupLine($"[blue]Input:[/] {(input.Length > 50 ? input[..50] + "..." : input)}");
            AnsiConsole.WriteLine();

            // Initialize orchestrator
            IronbeesOrchestrator? orchestrator = null;
            WorkflowService? workflowService = null;

            await AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("blue"))
                .StartAsync("[blue]Initializing workflow engine...[/]", async ctx =>
                {
                    var agentsDirectory = !string.IsNullOrEmpty(projectPath)
                        ? Path.Combine(projectPath, ".mloop", "agents")
                        : Path.Combine(Directory.GetCurrentDirectory(), ".mloop", "agents");

                    orchestrator = IronbeesOrchestrator.CreateFromEnvironment(agentsDirectory: agentsDirectory);
                    await orchestrator.InitializeAsync();

                    workflowService = new WorkflowService(agentsDirectory, orchestrator);
                });

            if (orchestrator == null || workflowService == null)
            {
                throw new InvalidOperationException("Failed to initialize workflow service");
            }

            AnsiConsole.MarkupLine("[green]\\u2713[/] Workflow engine initialized");
            AnsiConsole.WriteLine();

            // Load and validate workflow
            var workflow = await workflowService.LoadWorkflowAsync(workflowFile);
            var validation = workflowService.ValidateWorkflow(workflow);

            if (!validation.IsValid)
            {
                AnsiConsole.MarkupLine("[red]Workflow validation failed:[/]");
                foreach (var error in validation.Errors)
                {
                    AnsiConsole.MarkupLine($"  [red]\\u2716[/] {error}");
                }
                return 1;
            }

            AnsiConsole.MarkupLine($"[green]\\u2713[/] Workflow validated: [cyan]{workflow.Name}[/] v{workflow.Version}");
            AnsiConsole.MarkupLine($"[grey]Description: {workflow.Description ?? "No description"}[/]");
            AnsiConsole.WriteLine();

            // Display workflow states
            DisplayWorkflowStates(workflow);

            // Confirm execution
            if (!AnsiConsole.Confirm("Start workflow execution?"))
            {
                AnsiConsole.MarkupLine("[yellow]Workflow execution cancelled.[/]");
                return 0;
            }

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[blue]Starting workflow execution...[/]");
            AnsiConsole.WriteLine();

            // Execute workflow and display state updates
            var workingDirectory = projectPath ?? Directory.GetCurrentDirectory();
            string? currentExecutionId = null;

            await foreach (var state in workflowService.ExecuteWorkflowAsync(workflow, input, workingDirectory))
            {
                currentExecutionId = state.ExecutionId;
                DisplayStateUpdate(state);

                // Handle human gates
                if (state.Status == WorkflowExecutionStatus.WaitingForApproval)
                {
                    AnsiConsole.WriteLine();
                    AnsiConsole.MarkupLine("[yellow]Workflow waiting for approval...[/]");

                    var approve = AnsiConsole.Confirm("Approve this step?");
                    if (approve)
                    {
                        await workflowService.ApproveAsync(state.ExecutionId, true);
                        AnsiConsole.MarkupLine("[green]\\u2713 Approved[/]");
                    }
                    else
                    {
                        var feedback = AnsiConsole.Ask<string>("Rejection reason (optional):", "");
                        await workflowService.ApproveAsync(state.ExecutionId, false, feedback);
                        AnsiConsole.MarkupLine("[yellow]\\u2716 Rejected[/]");
                    }
                    AnsiConsole.WriteLine();
                }
            }

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[green]\\u2713 Workflow completed successfully![/]");

            return 0;
        }
        catch (Exception ex)
        {
            var errorMessage = ex.Message.Replace("[", "[[").Replace("]", "]]");
            AnsiConsole.MarkupLine($"[red]Error:[/] {errorMessage}");
            return 1;
        }
    }

    #endregion

    #region List Subcommand

    private static Command CreateListCommand()
    {
        var pathOption = new Option<string?>("--path", "-p")
        {
            Description = "Directory containing workflow files",
            DefaultValueFactory = _ => null
        };

        var command = new Command("list", "List available workflows");
        command.Options.Add(pathOption);

        command.SetAction((parseResult) =>
        {
            var path = parseResult.GetValue(pathOption);
            return ExecuteListAsync(path);
        });

        return command;
    }

    private static Task<int> ExecuteListAsync(string? path)
    {
        try
        {
            var searchPath = path ?? Path.Combine(Directory.GetCurrentDirectory(), ".mloop", "workflows");

            if (!Directory.Exists(searchPath))
            {
                // Also check for workflows directory at project root
                var altPath = Path.Combine(Directory.GetCurrentDirectory(), "workflows");
                if (Directory.Exists(altPath))
                {
                    searchPath = altPath;
                }
                else
                {
                    AnsiConsole.MarkupLine($"[yellow]No workflows directory found at:[/]");
                    AnsiConsole.MarkupLine($"  [grey]{searchPath}[/]");
                    AnsiConsole.MarkupLine($"[grey]Create workflow YAML files in this directory to get started.[/]");
                    return Task.FromResult(0);
                }
            }

            var workflowFiles = Directory.GetFiles(searchPath, "*.yaml", SearchOption.TopDirectoryOnly)
                .Concat(Directory.GetFiles(searchPath, "*.yml", SearchOption.TopDirectoryOnly))
                .ToList();

            if (workflowFiles.Count == 0)
            {
                AnsiConsole.MarkupLine($"[yellow]No workflow files found in:[/] {searchPath}");
                return Task.FromResult(0);
            }

            AnsiConsole.MarkupLine("[blue]Available Workflows:[/]");
            AnsiConsole.WriteLine();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .AddColumn("[blue]Name[/]")
                .AddColumn("[blue]File[/]")
                .AddColumn("[blue]Size[/]");

            foreach (var file in workflowFiles)
            {
                var fileName = Path.GetFileName(file);
                var fileInfo = new FileInfo(file);
                table.AddRow(
                    Path.GetFileNameWithoutExtension(file),
                    fileName,
                    $"{fileInfo.Length / 1024.0:F1} KB");
            }

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[grey]Found {workflowFiles.Count} workflow(s) in {searchPath}[/]");

            return Task.FromResult(0);
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return Task.FromResult(1);
        }
    }

    #endregion

    #region Validate Subcommand

    private static Command CreateValidateCommand()
    {
        var workflowArg = new Argument<string>("workflow-file")
        {
            Description = "Path to workflow YAML file to validate"
        };

        var projectOption = new Option<string?>("--project", "-p")
        {
            Description = "Path to MLoop project directory",
            DefaultValueFactory = _ => Directory.GetCurrentDirectory()
        };

        var command = new Command("validate", "Validate a workflow YAML file");
        command.Arguments.Add(workflowArg);
        command.Options.Add(projectOption);

        command.SetAction(async (parseResult) =>
        {
            var workflowFile = parseResult.GetValue(workflowArg)!;
            var projectPath = parseResult.GetValue(projectOption);
            return await ExecuteValidateAsync(workflowFile, projectPath);
        });

        return command;
    }

    private static async Task<int> ExecuteValidateAsync(string workflowFile, string? projectPath)
    {
        try
        {
            if (!File.Exists(workflowFile))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Workflow file not found: {workflowFile}");
                return 1;
            }

            AnsiConsole.MarkupLine($"[blue]Validating:[/] {Path.GetFileName(workflowFile)}");
            AnsiConsole.WriteLine();

            // Initialize minimal orchestrator for validation
            var agentsDirectory = !string.IsNullOrEmpty(projectPath)
                ? Path.Combine(projectPath, ".mloop", "agents")
                : Path.Combine(Directory.GetCurrentDirectory(), ".mloop", "agents");

            var orchestrator = IronbeesOrchestrator.CreateFromEnvironment(agentsDirectory: agentsDirectory);
            using var workflowService = new WorkflowService(agentsDirectory, orchestrator);

            // Load and validate
            var workflow = await workflowService.LoadWorkflowAsync(workflowFile);
            var validation = workflowService.ValidateWorkflow(workflow);

            // Display workflow info
            AnsiConsole.MarkupLine($"[blue]Name:[/] {workflow.Name}");
            AnsiConsole.MarkupLine($"[blue]Version:[/] {workflow.Version}");
            AnsiConsole.MarkupLine($"[blue]Description:[/] {workflow.Description ?? "N/A"}");
            AnsiConsole.WriteLine();

            // Display states
            DisplayWorkflowStates(workflow);

            // Display validation result
            if (validation.IsValid)
            {
                AnsiConsole.MarkupLine("[green]\\u2713 Workflow is valid[/]");
                return 0;
            }
            else
            {
                AnsiConsole.MarkupLine("[red]\\u2716 Validation errors:[/]");
                foreach (var error in validation.Errors)
                {
                    AnsiConsole.MarkupLine($"  [red]-[/] {error}");
                }
                return 1;
            }
        }
        catch (Exception ex)
        {
            var errorMessage = ex.Message.Replace("[", "[[").Replace("]", "]]");
            AnsiConsole.MarkupLine($"[red]Validation error:[/] {errorMessage}");
            return 1;
        }
    }

    #endregion

    #region Status Subcommand

    private static Command CreateStatusCommand()
    {
        var projectOption = new Option<string?>("--project", "-p")
        {
            Description = "Path to MLoop project directory",
            DefaultValueFactory = _ => Directory.GetCurrentDirectory()
        };

        var command = new Command("status", "Show active workflow executions");
        command.Options.Add(projectOption);

        command.SetAction(async (parseResult) =>
        {
            var projectPath = parseResult.GetValue(projectOption);
            return await ExecuteStatusAsync(projectPath);
        });

        return command;
    }

    private static async Task<int> ExecuteStatusAsync(string? projectPath)
    {
        try
        {
            var agentsDirectory = !string.IsNullOrEmpty(projectPath)
                ? Path.Combine(projectPath, ".mloop", "agents")
                : Path.Combine(Directory.GetCurrentDirectory(), ".mloop", "agents");

            var orchestrator = IronbeesOrchestrator.CreateFromEnvironment(agentsDirectory: agentsDirectory);
            using var workflowService = new WorkflowService(agentsDirectory, orchestrator);

            var executions = await workflowService.ListActiveExecutionsAsync();

            if (executions.Count == 0)
            {
                AnsiConsole.MarkupLine("[grey]No active workflow executions.[/]");
                return 0;
            }

            AnsiConsole.MarkupLine("[blue]Active Workflow Executions:[/]");
            AnsiConsole.WriteLine();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .AddColumn("[blue]Execution ID[/]")
                .AddColumn("[blue]Workflow[/]")
                .AddColumn("[blue]Current State[/]")
                .AddColumn("[blue]Status[/]")
                .AddColumn("[blue]Started[/]");

            foreach (var exec in executions)
            {
                var statusColor = exec.Status switch
                {
                    WorkflowExecutionStatus.Running => "green",
                    WorkflowExecutionStatus.WaitingForApproval => "yellow",
                    WorkflowExecutionStatus.Failed => "red",
                    _ => "grey"
                };

                table.AddRow(
                    exec.ExecutionId,
                    exec.WorkflowName,
                    exec.CurrentState,
                    $"[{statusColor}]{exec.Status}[/]",
                    exec.StartedAt.ToString("yyyy-MM-dd HH:mm:ss"));
            }

            AnsiConsole.Write(table);

            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return 1;
        }
    }

    #endregion

    #region Helper Methods

    private static void DisplayWorkflowStates(WorkflowDefinition workflow)
    {
        if (workflow.States == null || workflow.States.Count == 0)
        {
            return;
        }

        AnsiConsole.MarkupLine("[blue]Workflow States:[/]");

        var tree = new Tree("[blue]Workflow Flow[/]");

        foreach (var state in workflow.States)
        {
            var stateIcon = state.Type switch
            {
                WorkflowStateType.Start => "[green]\\u25B6[/]",
                WorkflowStateType.Terminal => "[red]\\u25A0[/]",
                WorkflowStateType.Agent => "[blue]\\u25CF[/]",
                WorkflowStateType.HumanGate => "[yellow]\\u2616[/]",
                _ => "[grey]\\u25CB[/]"
            };

            var nextInfo = !string.IsNullOrEmpty(state.Next)
                ? $"[grey]-> {state.Next}[/]"
                : "";

            tree.AddNode($"{stateIcon} {state.Id} [grey]({state.Type})[/] {nextInfo}");
        }

        AnsiConsole.Write(tree);
        AnsiConsole.WriteLine();
    }

    private static void DisplayStateUpdate(WorkflowRuntimeState state)
    {
        var statusColor = state.Status switch
        {
            WorkflowExecutionStatus.Running => "blue",
            WorkflowExecutionStatus.WaitingForApproval => "yellow",
            WorkflowExecutionStatus.Completed => "green",
            WorkflowExecutionStatus.Failed => "red",
            WorkflowExecutionStatus.Cancelled => "grey",
            _ => "grey"
        };

        var stateIcon = state.Status switch
        {
            WorkflowExecutionStatus.Running => "\\u25B6",
            WorkflowExecutionStatus.WaitingForApproval => "\\u2616",
            WorkflowExecutionStatus.Completed => "\\u2713",
            WorkflowExecutionStatus.Failed => "\\u2716",
            WorkflowExecutionStatus.Cancelled => "\\u25A0",
            _ => "\\u25CB"
        };

        AnsiConsole.MarkupLine(
            $"[{statusColor}]{stateIcon}[/] " +
            $"[blue]{state.CurrentStateId}[/] - " +
            $"[{statusColor}]{state.Status}[/]");

        // Display any output data
        if (state.OutputData.Count > 0)
        {
            foreach (var kvp in state.OutputData)
            {
                var value = kvp.Value?.ToString() ?? "null";
                if (value.Length > 100)
                {
                    value = value[..100] + "...";
                }
                AnsiConsole.MarkupLine($"  [grey]{kvp.Key}:[/] {value.Replace("[", "[[").Replace("]", "]]")}");
            }
        }
    }

    #endregion
}
