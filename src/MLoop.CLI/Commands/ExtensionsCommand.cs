using System.CommandLine;
using MLoop.Core.Scripting;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop extensions - Lists all discovered extensibility scripts (hooks and metrics)
/// </summary>
public static class ExtensionsCommand
{
    public static Command Create()
    {
        var command = new Command("extensions", "List all discovered extensibility scripts");

        var listCommand = new Command("list", "List all hooks and metrics");
        listCommand.SetHandler(ExecuteListAsync);

        command.AddCommand(listCommand);
        command.SetHandler(ExecuteListAsync); // Default to list when no subcommand

        return command;
    }

    private static async Task<int> ExecuteListAsync()
    {
        // NOTE: Phase 1 (Hooks & Metrics) - Disabled for Phase 0 (Preprocessing)
        // TODO: Re-enable when implementing Phase 1

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[yellow]⚠️  This command is not yet available[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]The 'extensions' command lists hooks and metrics (Phase 1 features).[/]");
        AnsiConsole.MarkupLine("[grey]Currently implementing Phase 0 (preprocessing scripts).[/]");
        AnsiConsole.MarkupLine("[grey]This command will be enabled in a future release.[/]");
        AnsiConsole.WriteLine();
        return 0;

#if false
        try
        {
            var projectRoot = Directory.GetCurrentDirectory();
            var discovery = new ScriptDiscovery(projectRoot);

            AnsiConsole.MarkupLine("[blue]MLoop Extensibility Scripts[/]");
            AnsiConsole.WriteLine();

            // Check if extensibility is available
            if (!discovery.IsExtensibilityAvailable())
            {
                AnsiConsole.MarkupLine("[yellow]ℹ️  No extensibility scripts found[/]");
                AnsiConsole.MarkupLine("[grey]   Scripts directory: .mloop/scripts/[/]");
                AnsiConsole.MarkupLine("[grey]   To add extensions, place .cs files in:[/]");
                AnsiConsole.MarkupLine("[grey]   - .mloop/scripts/hooks/[/]");
                AnsiConsole.MarkupLine("[grey]   - .mloop/scripts/metrics/[/]");
                return 0;
            }

            // Discover hooks
            AnsiConsole.MarkupLine("[cyan]📋 Hooks[/]");
            var hooks = await discovery.DiscoverHooksAsync();

            if (hooks.Count == 0)
            {
                AnsiConsole.MarkupLine("  [grey]No hooks found[/]");
            }
            else
            {
                var hookTable = new Table();
                hookTable.Border(TableBorder.Rounded);
                hookTable.AddColumn("[green]Name[/]");
                hookTable.AddColumn("[cyan]Type[/]");

                foreach (var hook in hooks)
                {
                    hookTable.AddRow(
                        $"[green]{hook.Name}[/]",
                        $"[cyan]{hook.GetType().Name}[/]"
                    );
                }

                AnsiConsole.Write(hookTable);
                AnsiConsole.MarkupLine($"  [grey]Total: {hooks.Count} hook(s)[/]");
            }

            AnsiConsole.WriteLine();

            // Discover metrics
            AnsiConsole.MarkupLine("[cyan]📊 Metrics[/]");
            var metrics = await discovery.DiscoverMetricsAsync();

            if (metrics.Count == 0)
            {
                AnsiConsole.MarkupLine("  [grey]No metrics found[/]");
            }
            else
            {
                var metricTable = new Table();
                metricTable.Border(TableBorder.Rounded);
                metricTable.AddColumn("[green]Name[/]");
                metricTable.AddColumn("[cyan]Type[/]");
                metricTable.AddColumn("[yellow]Direction[/]");

                foreach (var metric in metrics)
                {
                    var direction = metric.HigherIsBetter ? "↑ Higher is better" : "↓ Lower is better";
                    metricTable.AddRow(
                        $"[green]{metric.Name}[/]",
                        $"[cyan]{metric.GetType().Name}[/]",
                        $"[yellow]{direction}[/]"
                    );
                }

                AnsiConsole.Write(metricTable);
                AnsiConsole.MarkupLine($"  [grey]Total: {metrics.Count} metric(s)[/]");
            }

            AnsiConsole.WriteLine();

            // Summary
            var totalExtensions = hooks.Count + metrics.Count;
            if (totalExtensions > 0)
            {
                AnsiConsole.MarkupLine($"[green]✓[/] Found {totalExtensions} extension(s) total");
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine("[grey]Use 'mloop validate' to check for compilation errors[/]");
            }

            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return 1;
        }
#endif
    }
}
