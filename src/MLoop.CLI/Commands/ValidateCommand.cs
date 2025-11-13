using System.CommandLine;
using MLoop.Core.Scripting;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop validate - Validates extensibility scripts (hooks and metrics)
/// </summary>
public static class ValidateCommand
{
    public static Command Create()
    {
        var command = new Command("validate", "Validate extensibility scripts (hooks and metrics)");

        command.SetHandler(ExecuteAsync);

        return command;
    }

    private static async Task<int> ExecuteAsync()
    {
        // NOTE: Phase 1 (Hooks & Metrics) - Disabled for Phase 0 (Preprocessing)
        // TODO: Re-enable when implementing Phase 1

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[yellow]⚠️  This command is not yet available[/]");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]The 'validate' command validates hooks and metrics (Phase 1 features).[/]");
        AnsiConsole.MarkupLine("[grey]Currently implementing Phase 0 (preprocessing scripts).[/]");
        AnsiConsole.MarkupLine("[grey]This command will be enabled in a future release.[/]");
        AnsiConsole.WriteLine();
        return 0;

#if false
        try
        {
            var projectRoot = Directory.GetCurrentDirectory();
            var discovery = new ScriptDiscovery(projectRoot);

            AnsiConsole.MarkupLine("[blue]Validating MLoop extensibility scripts...[/]");
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

            var hasErrors = false;
            var totalScripts = 0;
            var validScripts = 0;

            // Validate hooks
            AnsiConsole.MarkupLine("[cyan]📋 Validating hooks...[/]");
            var hooksPath = discovery.GetHooksDirectory();
            if (Directory.Exists(hooksPath))
            {
                var hookFiles = Directory.GetFiles(hooksPath, "*.cs");
                totalScripts += hookFiles.Length;

                foreach (var hookFile in hookFiles)
                {
                    var fileName = Path.GetFileName(hookFile);
                    try
                    {
                        var loader = new ScriptLoader();
                        var hooks = await loader.LoadScriptAsync<MLoop.Extensibility.IMLoopHook>(hookFile);

                        if (hooks.Count == 0)
                        {
                            AnsiConsole.MarkupLine($"  [yellow]⚠️  {fileName}[/] - No hooks found (missing IMLoopHook implementation?)");
                            hasErrors = true;
                        }
                        else
                        {
                            foreach (var hook in hooks)
                            {
                                AnsiConsole.MarkupLine($"  [green]✓[/] {fileName} → {hook.Name}");
                            }
                            validScripts++;
                        }
                    }
                    catch (Exception ex)
                    {
                        AnsiConsole.MarkupLine($"  [red]✗[/] {fileName} - Compilation error:");
                        AnsiConsole.MarkupLine($"    [red]{ex.Message}[/]");
                        hasErrors = true;
                    }
                }
            }

            AnsiConsole.WriteLine();

            // Validate metrics
            AnsiConsole.MarkupLine("[cyan]📊 Validating metrics...[/]");
            var metricsPath = discovery.GetMetricsDirectory();
            if (Directory.Exists(metricsPath))
            {
                var metricFiles = Directory.GetFiles(metricsPath, "*.cs");
                totalScripts += metricFiles.Length;

                foreach (var metricFile in metricFiles)
                {
                    var fileName = Path.GetFileName(metricFile);
                    try
                    {
                        var loader = new ScriptLoader();
                        var metrics = await loader.LoadScriptAsync<MLoop.Extensibility.IMLoopMetric>(metricFile);

                        if (metrics.Count == 0)
                        {
                            AnsiConsole.MarkupLine($"  [yellow]⚠️  {fileName}[/] - No metrics found (missing IMLoopMetric implementation?)");
                            hasErrors = true;
                        }
                        else
                        {
                            foreach (var metric in metrics)
                            {
                                var direction = metric.HigherIsBetter ? "↑ Higher is better" : "↓ Lower is better";
                                AnsiConsole.MarkupLine($"  [green]✓[/] {fileName} → {metric.Name} ({direction})");
                            }
                            validScripts++;
                        }
                    }
                    catch (Exception ex)
                    {
                        AnsiConsole.MarkupLine($"  [red]✗[/] {fileName} - Compilation error:");
                        AnsiConsole.MarkupLine($"    [red]{ex.Message}[/]");
                        hasErrors = true;
                    }
                }
            }

            AnsiConsole.WriteLine();

            // Summary
            if (totalScripts == 0)
            {
                AnsiConsole.MarkupLine("[yellow]ℹ️  No scripts to validate[/]");
                return 0;
            }

            if (hasErrors)
            {
                AnsiConsole.MarkupLine($"[red]✗ Validation failed:[/] {validScripts}/{totalScripts} scripts valid");
                return 1;
            }
            else
            {
                AnsiConsole.MarkupLine($"[green]✓ Validation successful:[/] All {totalScripts} scripts compiled successfully");
                return 0;
            }
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return 1;
        }
#endif
    }
}
