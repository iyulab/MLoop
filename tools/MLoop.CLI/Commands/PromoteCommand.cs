using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop promote - Manually promotes an experiment to production
/// </summary>
public static class PromoteCommand
{
    public static Command Create()
    {
        var experimentIdArg = new Argument<string>("experiment-id")
        {
            Description = "Experiment ID to promote (e.g., exp-003)"
        };

        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')"
        };

        var forceOption = new Option<bool>("--force", "-f")
        {
            Description = "Skip confirmation when replacing existing production model",
            DefaultValueFactory = _ => false
        };

        var command = new Command("promote", "Promote an experiment to production");
        command.Arguments.Add(experimentIdArg);
        command.Options.Add(nameOption);
        command.Options.Add(forceOption);

        command.SetAction((parseResult) =>
        {
            var experimentId = parseResult.GetValue(experimentIdArg)!;
            var name = parseResult.GetValue(nameOption);
            var force = parseResult.GetValue(forceOption);
            return ExecuteAsync(experimentId, name, force);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string experimentId,
        string? modelName,
        bool force)
    {
        try
        {
            // Initialize components
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);

            // Find project root
            string projectRoot;
            try
            {
                projectRoot = projectDiscovery.FindRoot();
            }
            catch (InvalidOperationException)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Not inside a MLoop project.");
                AnsiConsole.MarkupLine("Run [blue]mloop init[/] to create a new project.");
                return 1;
            }

            // Resolve model name (defaults to "default")
            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? ConfigDefaults.DefaultModelName
                : modelName.Trim().ToLowerInvariant();

            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

            // Verify experiment exists for this model
            if (!experimentStore.ExperimentExists(resolvedModelName, experimentId))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Experiment '{experimentId}' not found for model '[cyan]{resolvedModelName}[/]'.");
                AnsiConsole.MarkupLine($"[yellow]Tip:[/] Run [blue]mloop list --name {resolvedModelName}[/] to see all experiments.");
                return 1;
            }

            // Load experiment to show details
            var experiment = await experimentStore.LoadAsync(resolvedModelName, experimentId, CancellationToken.None);

            // Check if experiment is completed
            if (!experiment.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Cannot promote experiment with status '{experiment.Status}'.");
                AnsiConsole.MarkupLine("[yellow]Only completed experiments can be promoted.[/]");
                return 1;
            }

            // Show promotion details
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule($"[bold]Promoting to Production[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey);

            table.AddColumn("[bold]Property[/]");
            table.AddColumn("[bold]Value[/]");

            table.AddRow("Model", $"[cyan]{resolvedModelName}[/]");
            table.AddRow("Experiment ID", $"[cyan]{experiment.ExperimentId}[/]");
            table.AddRow("Task", $"[yellow]{experiment.Task}[/]");
            table.AddRow("Timestamp", experiment.Timestamp.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss"));

            if (experiment.Metrics != null && experiment.Metrics.Any())
            {
                var metricsStr = string.Join(", ", experiment.Metrics.Select(m => $"{m.Key}={m.Value:F4}"));
                table.AddRow("Metrics", $"[yellow]{metricsStr}[/]");
            }

            if (experiment.Result != null)
            {
                table.AddRow("Best Trainer", $"[green]{experiment.Result.BestTrainer}[/]");
                table.AddRow("Training Time", $"{experiment.Result.TrainingTimeSeconds:F2}s");
            }

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();

            // Check if replacing existing production model
            var currentProduction = await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None);
            if (currentProduction != null)
            {
                AnsiConsole.MarkupLine($"[yellow]Warning:[/] This will replace the current production model: [cyan]{currentProduction.ExperimentId}[/]");
                AnsiConsole.WriteLine();

                if (!force)
                {
                    if (!AnsiConsole.Confirm("Continue with promotion?"))
                    {
                        AnsiConsole.MarkupLine("[grey]Promotion cancelled.[/]");
                        return 0;
                    }
                }
            }

            // Promote with progress indicator
            await AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .StartAsync("[yellow]Promoting to production...[/]", async ctx =>
                {
                    await modelRegistry.PromoteAsync(resolvedModelName, experimentId, CancellationToken.None);
                    ctx.Status("[green]Promotion complete![/]");
                });

            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[green]Promotion Complete![/]").LeftJustified());
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
            AnsiConsole.MarkupLine($"[green]>[/] Experiment: [cyan]{experimentId}[/]");
            AnsiConsole.MarkupLine($"[green]>[/] Status: [green bold]production[/]");
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
            AnsiConsole.MarkupLine($"  mloop predict --name {resolvedModelName}");
            AnsiConsole.MarkupLine($"  mloop info --name {resolvedModelName}");
            AnsiConsole.WriteLine();

            return 0;
        }
        catch (FileNotFoundException ex)
        {
            AnsiConsole.Markup("[red]Error:[/] ");
            AnsiConsole.WriteLine(ex.Message);
            return 1;
        }
        catch (Exception ex)
        {
            AnsiConsole.Markup("[red]Error:[/] ");
            AnsiConsole.WriteLine(ex.Message);

            if (ex.InnerException != null)
            {
                AnsiConsole.Markup("[grey]Details:[/] ");
                AnsiConsole.WriteLine(ex.InnerException.Message);
            }

            return 1;
        }
    }
}
