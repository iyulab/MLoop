using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop list - Lists all experiments with their status and metrics
/// </summary>
public static class ListCommand
{
    public static Command Create()
    {
        var allOption = new Option<bool>("--all", "-a")
        {
            Description = "Show all experiments including failed ones (default: completed only)",
            DefaultValueFactory = _ => false
        };

        var command = new Command("list", "List all experiments");
        command.Options.Add(allOption);

        command.SetAction((parseResult) =>
        {
            var showAll = parseResult.GetValue(allOption);
            return ExecuteAsync(showAll);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(bool showAll)
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

            // Get all experiments
            var experiments = await experimentStore.ListAsync(CancellationToken.None);
            var experimentsList = experiments.ToList();

            // Get production model
            var productionModel = await modelRegistry.GetAsync(ModelStage.Production, CancellationToken.None);

            // Filter if needed
            if (!showAll)
            {
                experimentsList = experimentsList
                    .Where(e => e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
                    .ToList();
            }

            if (!experimentsList.Any())
            {
                AnsiConsole.MarkupLine("[yellow]No experiments found.[/]");
                AnsiConsole.MarkupLine("[grey]Run [blue]mloop train[/] to create your first experiment.[/]");
                return 0;
            }

            // Sort by timestamp (newest first)
            experimentsList = experimentsList.OrderByDescending(e => e.Timestamp).ToList();

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[bold]Experiments in {Path.GetFileName(projectRoot)}[/]");
            AnsiConsole.WriteLine();

            // Create table
            var table = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey);

            table.AddColumn(new TableColumn("[bold]ID[/]").Centered());
            table.AddColumn(new TableColumn("[bold]Timestamp[/]"));
            table.AddColumn(new TableColumn("[bold]Status[/]").Centered());
            table.AddColumn(new TableColumn("[bold]Best Metric[/]").RightAligned());
            table.AddColumn(new TableColumn("[bold]Stage[/]").Centered());

            foreach (var exp in experimentsList)
            {
                var isProduction = productionModel != null && exp.ExperimentId == productionModel.ExperimentId;

                var id = isProduction
                    ? $"[green bold]{exp.ExperimentId}[/]"
                    : $"[cyan]{exp.ExperimentId}[/]";

                var timestamp = exp.Timestamp.ToLocalTime().ToString("yyyy-MM-dd HH:mm");

                var status = exp.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase)
                    ? "[green]✓ Completed[/]"
                    : exp.Status.Equals("Failed", StringComparison.OrdinalIgnoreCase)
                        ? "[red]✗ Failed[/]"
                        : $"[yellow]{exp.Status}[/]";

                var metric = exp.BestMetric.HasValue
                    ? $"[yellow]{exp.BestMetric.Value:F4}[/]"
                    : "[grey]-[/]";

                var stage = isProduction
                    ? "[green bold]★ Production[/]"
                    : "[grey]Staging[/]";

                table.AddRow(id, timestamp, status, metric, stage);
            }

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();

            // Show summary
            var totalCount = experimentsList.Count;
            var completedCount = experimentsList.Count(e => e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase));
            var failedCount = experimentsList.Count(e => e.Status.Equals("Failed", StringComparison.OrdinalIgnoreCase));

            AnsiConsole.MarkupLine($"[grey]Total: {totalCount} | Completed: [green]{completedCount}[/] | Failed: [red]{failedCount}[/][/]");

            if (productionModel != null)
            {
                AnsiConsole.MarkupLine($"[grey]Production model: [green]{productionModel.ExperimentId}[/] (promoted {productionModel.PromotedAt.ToLocalTime():yyyy-MM-dd HH:mm})[/]");
            }
            else
            {
                AnsiConsole.MarkupLine("[grey]No production model. Train a model to auto-promote to production.[/]");
            }

            AnsiConsole.WriteLine();

            return 0;
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
