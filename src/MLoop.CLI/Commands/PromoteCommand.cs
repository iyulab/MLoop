using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop promote - Manually promotes an experiment to production or staging
/// </summary>
public static class PromoteCommand
{
    public static Command Create()
    {
        var experimentIdArg = new Argument<string>("experiment-id")
        {
            Description = "Experiment ID to promote (e.g., exp-003)"
        };

        var stageOption = new Option<string>("--stage", "-s")
        {
            Description = "Target stage (production or staging)",
            DefaultValueFactory = _ => "production"
        };

        var command = new Command("promote", "Promote an experiment to production or staging");
        command.Arguments.Add(experimentIdArg);
        command.Options.Add(stageOption);

        command.SetAction((parseResult) =>
        {
            var experimentId = parseResult.GetValue(experimentIdArg)!;
            var stageInput = parseResult.GetValue(stageOption)!;
            return ExecuteAsync(experimentId, stageInput);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string experimentId,
        string stageInput)
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

            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

            // Parse stage
            if (!Enum.TryParse<ModelStage>(stageInput, ignoreCase: true, out var targetStage))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Invalid stage '{stageInput}'. Valid values: production, staging");
                return 1;
            }

            // Verify experiment exists
            if (!experimentStore.ExperimentExists(experimentId))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Experiment '{experimentId}' not found.");
                AnsiConsole.MarkupLine("[yellow]Tip:[/] Run [blue]mloop list[/] to see all experiments.");
                return 1;
            }

            // Load experiment to show details
            var experiment = await experimentStore.LoadAsync(experimentId, CancellationToken.None);

            // Check if experiment is completed
            if (!experiment.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Cannot promote experiment with status '{experiment.Status}'.");
                AnsiConsole.MarkupLine("[yellow]Only completed experiments can be promoted.[/]");
                return 1;
            }

            // Show promotion details
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[bold]Promoting {experimentId} to {targetStage.ToString().ToLowerInvariant()}[/]");
            AnsiConsole.WriteLine();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey);

            table.AddColumn("[bold]Property[/]");
            table.AddColumn("[bold]Value[/]");

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
            if (targetStage == ModelStage.Production)
            {
                var currentProduction = await modelRegistry.GetAsync(ModelStage.Production, CancellationToken.None);
                if (currentProduction != null)
                {
                    AnsiConsole.MarkupLine($"[yellow]⚠ This will replace the current production model:[/] [cyan]{currentProduction.ExperimentId}[/]");
                }
            }

            // Promote with progress indicator
            await AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .StartAsync($"[yellow]Promoting to {targetStage.ToString().ToLowerInvariant()}...[/]", async ctx =>
                {
                    await modelRegistry.PromoteAsync(experimentId, targetStage, CancellationToken.None);
                    ctx.Status($"[green]Promotion complete![/]");
                });

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[green]✓[/] Successfully promoted [cyan]{experimentId}[/] to [green bold]{targetStage.ToString().ToLowerInvariant()}[/]!");
            AnsiConsole.WriteLine();

            if (targetStage == ModelStage.Production)
            {
                AnsiConsole.MarkupLine("[grey]You can now use [blue]mloop predict[/] to make predictions with this model.[/]");
            }

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
