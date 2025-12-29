using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop list - Lists all experiments with their status and metrics
/// </summary>
public static class ListCommand
{
    public static Command Create()
    {
        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = "Filter by model name (shows all models if omitted)"
        };

        var allOption = new Option<bool>("--all", "-a")
        {
            Description = "Show all experiments including failed ones (default: completed only)",
            DefaultValueFactory = _ => false
        };

        var command = new Command("list", "List all experiments");
        command.Options.Add(nameOption);
        command.Options.Add(allOption);

        command.SetAction((parseResult) =>
        {
            var name = parseResult.GetValue(nameOption);
            var showAll = parseResult.GetValue(allOption);
            return ExecuteAsync(name, showAll);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(string? modelName, bool showAll)
    {
        try
        {
            var ctx = CommandContext.TryCreate();
            if (ctx == null) return 1;

            var resolvedModelName = CommandContext.ResolveOptionalModelName(modelName);

            // Get experiments (filtered by model if specified)
            var experiments = await ctx.ExperimentStore.ListAsync(resolvedModelName, CancellationToken.None);
            var experimentsList = experiments.ToList();

            // Get production models for display
            var productionModels = await ctx.ModelRegistry.ListAsync(null, CancellationToken.None);
            var productionDict = productionModels.ToDictionary(m => m.ModelName, m => m.ExperimentId);

            // Filter if needed
            if (!showAll)
            {
                experimentsList = experimentsList
                    .Where(e => e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
                    .ToList();
            }

            if (!experimentsList.Any())
            {
                if (resolvedModelName != null)
                {
                    AnsiConsole.MarkupLine($"[yellow]No experiments found for model '[cyan]{resolvedModelName}[/]'.[/]");
                    AnsiConsole.MarkupLine($"[grey]Run [blue]mloop train --name {resolvedModelName}[/] to create an experiment.[/]");
                }
                else
                {
                    AnsiConsole.MarkupLine("[yellow]No experiments found.[/]");
                    AnsiConsole.MarkupLine("[grey]Run [blue]mloop train[/] to create your first experiment.[/]");
                }
                return 0;
            }

            // Sort by timestamp (newest first)
            experimentsList = experimentsList.OrderByDescending(e => e.Timestamp).ToList();

            AnsiConsole.WriteLine();

            var title = resolvedModelName != null
                ? $"[bold]Experiments for model '[cyan]{resolvedModelName}[/]'[/]"
                : $"[bold]All Experiments in {Path.GetFileName(ctx.ProjectRoot)}[/]";

            AnsiConsole.Write(new Rule(title).LeftJustified());
            AnsiConsole.WriteLine();

            // Create table
            var table = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey);

            // Add model column only when showing all models
            if (resolvedModelName == null)
            {
                table.AddColumn(new TableColumn("[bold]Model[/]"));
            }

            table.AddColumn(new TableColumn("[bold]ID[/]").Centered());
            table.AddColumn(new TableColumn("[bold]Timestamp[/]"));
            table.AddColumn(new TableColumn("[bold]Status[/]").Centered());
            table.AddColumn(new TableColumn("[bold]Best Metric[/]").RightAligned());
            table.AddColumn(new TableColumn("[bold]Stage[/]").Centered());

            foreach (var exp in experimentsList)
            {
                var expModelName = exp.ModelName ?? ConfigDefaults.DefaultModelName;
                var isProduction = productionDict.TryGetValue(expModelName, out var prodExpId) && exp.ExperimentId == prodExpId;

                var id = isProduction
                    ? $"[green bold]{exp.ExperimentId}[/]"
                    : $"[cyan]{exp.ExperimentId}[/]";

                var timestamp = exp.Timestamp.ToLocalTime().ToString("yyyy-MM-dd HH:mm");

                var status = exp.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase)
                    ? "[green]Completed[/]"
                    : exp.Status.Equals("Failed", StringComparison.OrdinalIgnoreCase)
                        ? "[red]Failed[/]"
                        : $"[yellow]{exp.Status}[/]";

                var metric = exp.BestMetric.HasValue
                    ? $"[yellow]{exp.BestMetric.Value:F4}[/]"
                    : "[grey]-[/]";

                var stage = isProduction
                    ? "[green bold]Production[/]"
                    : "[grey]-[/]";

                if (resolvedModelName == null)
                {
                    table.AddRow($"[blue]{expModelName}[/]", id, timestamp, status, metric, stage);
                }
                else
                {
                    table.AddRow(id, timestamp, status, metric, stage);
                }
            }

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();

            // Show summary
            var totalCount = experimentsList.Count;
            var completedCount = experimentsList.Count(e => e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase));
            var failedCount = experimentsList.Count(e => e.Status.Equals("Failed", StringComparison.OrdinalIgnoreCase));

            AnsiConsole.MarkupLine($"[grey]Total: {totalCount} | Completed: [green]{completedCount}[/] | Failed: [red]{failedCount}[/][/]");

            // Show production info
            if (resolvedModelName != null)
            {
                // Single model - show its production status
                if (productionDict.TryGetValue(resolvedModelName, out var prodId))
                {
                    AnsiConsole.MarkupLine($"[grey]Production model: [green]{prodId}[/][/]");
                }
                else
                {
                    AnsiConsole.MarkupLine($"[grey]No production model for '{resolvedModelName}'. Use [blue]mloop promote <exp-id> --name {resolvedModelName}[/][/]");
                }
            }
            else
            {
                // All models - show all production models
                var distinctModels = experimentsList.Select(e => e.ModelName ?? ConfigDefaults.DefaultModelName).Distinct().ToList();
                var prodCount = distinctModels.Count(m => productionDict.ContainsKey(m));
                AnsiConsole.MarkupLine($"[grey]Models: {distinctModels.Count} | Production: {prodCount}[/]");
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
