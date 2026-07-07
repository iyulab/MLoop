using System.CommandLine;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Evaluation;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop promote - Manually promotes an experiment to production
/// </summary>
public static class PromoteCommand
{
    public static Command Create()
    {
        var experimentIdArg = new Argument<string?>("experiment-id")
        {
            Description = "Experiment ID to promote (e.g., exp-003); omit when using --latest/--best",
            Arity = ArgumentArity.ZeroOrOne
        };

        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')"
        };

        var latestOption = new Option<bool>("--latest")
        {
            Description = "Promote the most recent completed experiment",
            DefaultValueFactory = _ => false
        };

        var bestOption = new Option<bool>("--best")
        {
            Description = "Promote the completed experiment with the best metric (direction-aware)",
            DefaultValueFactory = _ => false
        };

        var forceOption = new Option<bool>("--force", "-f")
        {
            Description = "Skip confirmation when replacing existing production model",
            DefaultValueFactory = _ => false
        };

        var noBackupOption = new Option<bool>("--no-backup")
        {
            Description = "Skip backup of current production model before promotion",
            DefaultValueFactory = _ => false
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output the result as JSON (machine-readable, non-interactive)"
        };

        var command = new Command("promote", "Promote an experiment to production");
        command.Arguments.Add(experimentIdArg);
        command.Options.Add(nameOption);
        command.Options.Add(latestOption);
        command.Options.Add(bestOption);
        command.Options.Add(forceOption);
        command.Options.Add(noBackupOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var experimentId = parseResult.GetValue(experimentIdArg);
            var name = parseResult.GetValue(nameOption);
            var latest = parseResult.GetValue(latestOption);
            var best = parseResult.GetValue(bestOption);
            var force = parseResult.GetValue(forceOption);
            var noBackup = parseResult.GetValue(noBackupOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteAsync(experimentId, name, latest, best, force, noBackup, json);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? experimentId,
        string? modelName,
        bool latest,
        bool best,
        bool force,
        bool noBackup,
        bool jsonOutput)
    {
        try
        {
            // Exactly one selector: an explicit ID, --latest, or --best.
            var selectorCount = (experimentId != null ? 1 : 0) + (latest ? 1 : 0) + (best ? 1 : 0);
            if (selectorCount != 1)
            {
                WriteError(selectorCount == 0
                    ? "Specify an experiment: an explicit ID, --latest, or --best."
                    : "Use only one of: an explicit experiment ID, --latest, --best.", jsonOutput);
                return 1;
            }

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
                WriteError("Not inside a MLoop project. Run 'mloop init' to create a new project.", jsonOutput);
                return 1;
            }

            // Resolve model name (defaults to "default")
            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? ConfigDefaults.DefaultModelName
                : modelName.Trim().ToLowerInvariant();

            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

            // Resolve --latest/--best to a concrete experiment ID.
            string? autoSelectionReason = null;
            if (experimentId == null)
            {
                var candidates = (await experimentStore.ListAsync(resolvedModelName, CancellationToken.None)).ToList();
                var (selected, reason, error) = SelectExperiment(candidates, resolvedModelName, best);
                if (selected == null)
                {
                    WriteError(error!, jsonOutput);
                    return 1;
                }
                experimentId = selected;
                autoSelectionReason = reason;
            }

            // Verify experiment exists for this model
            if (!experimentStore.ExperimentExists(resolvedModelName, experimentId))
            {
                WriteError($"Experiment '{experimentId}' not found for model '{resolvedModelName}'. " +
                           $"Run 'mloop list --name {resolvedModelName}' to see all experiments.", jsonOutput);
                return 1;
            }

            // Load experiment to show details
            var experiment = await experimentStore.LoadAsync(resolvedModelName, experimentId, CancellationToken.None);

            // Check if experiment is completed
            if (!experiment.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
            {
                WriteError($"Cannot promote experiment with status '{experiment.Status}'. " +
                           "Only completed experiments can be promoted.", jsonOutput);
                return 1;
            }

            if (!jsonOutput)
                ShowPromotionDetails(resolvedModelName, experiment, autoSelectionReason);

            // Check if replacing existing production model
            var currentProduction = await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None);
            if (currentProduction != null && !jsonOutput)
            {
                AnsiConsole.MarkupLine($"[yellow]Warning:[/] This will replace the current production model: [cyan]{currentProduction.ExperimentId}[/]");
                AnsiConsole.WriteLine();

                // --json is a non-interactive automation surface: never prompt there.
                if (!force && !Console.IsInputRedirected)
                {
                    if (!AnsiConsole.Confirm("Continue with promotion?"))
                    {
                        AnsiConsole.MarkupLine("[grey]Promotion cancelled.[/]");
                        return 0;
                    }
                }
            }

            // Create promotion manager for backup + history
            var promotionManager = new FilePromotionManager(projectRoot);

            // Backup current production before promotion
            string? backupPath = null;
            if (!noBackup && currentProduction != null)
            {
                backupPath = await promotionManager.BackupProductionAsync(resolvedModelName);
            }

            if (jsonOutput)
            {
                await modelRegistry.PromoteAsync(resolvedModelName, experimentId, CancellationToken.None);
            }
            else
            {
                // Promote with progress indicator
                await AnsiConsole.Status()
                    .Spinner(Spinner.Known.Dots)
                    .StartAsync("[yellow]Promoting to production...[/]", async ctx =>
                    {
                        await modelRegistry.PromoteAsync(resolvedModelName, experimentId, CancellationToken.None);
                        ctx.Status("[green]Promotion complete![/]");
                    });
            }

            // Record promotion history
            await promotionManager.RecordPromotionAsync(
                resolvedModelName, experimentId,
                currentProduction?.ExperimentId, "promote");

            if (jsonOutput)
            {
                OutputAsJson(resolvedModelName, experimentId, autoSelectionReason,
                    currentProduction?.ExperimentId, backupPath);
            }
            else
            {
                ShowPromotionResult(resolvedModelName, experimentId, backupPath);
            }

            return 0;
        }
        catch (Exception ex)
        {
            if (jsonOutput)
            {
                WriteError(ex.Message, jsonOutput: true);
                return 1;
            }
            ErrorSuggestions.DisplayError(ex, "promote");
            return 1;
        }
    }

    /// <summary>
    /// Picks the newest (--latest) or best-metric (--best) completed experiment. Metric comparison
    /// is direction-aware via <see cref="MetricDirection"/> (rmse-style = lower wins).
    /// </summary>
    internal static (string? ExperimentId, string? Reason, string? Error) SelectExperiment(
        IReadOnlyList<ExperimentSummary> candidates, string modelName, bool best)
    {
        var experiments = candidates
            .Where(e => e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (experiments.Count == 0)
            return (null, null,
                $"No completed experiments found for model '{modelName}'. Run 'mloop train --name {modelName}' first.");

        if (!best)
        {
            var newest = experiments.OrderByDescending(e => e.Timestamp).First();
            return (newest.ExperimentId, "latest", null);
        }

        var scored = experiments.Where(e => e.BestMetric.HasValue && !string.IsNullOrEmpty(e.MetricName)).ToList();
        if (scored.Count == 0)
            return (null, null,
                $"No completed experiment for model '{modelName}' has a recorded metric — cannot select --best. " +
                "Use --latest or an explicit experiment ID.");

        // Comparing across different metric names is meaningless; require a single metric.
        var metricNames = scored.Select(e => e.MetricName!.ToLowerInvariant()).Distinct().ToList();
        if (metricNames.Count > 1)
            return (null, null,
                $"Experiments for model '{modelName}' were optimized for different metrics ({string.Join(", ", metricNames)}) — " +
                "--best cannot compare across them. Use --latest or an explicit experiment ID.");

        var lowerIsBetter = MetricDirection.IsLowerBetter(metricNames[0]);
        var winner = (lowerIsBetter
                ? scored.OrderBy(e => e.BestMetric!.Value)
                : scored.OrderByDescending(e => e.BestMetric!.Value))
            .ThenByDescending(e => e.Timestamp) // deterministic tie-break: newer wins
            .First();

        return (winner.ExperimentId, $"best {metricNames[0]}={winner.BestMetric!.Value:F4}", null);
    }

    private static void ShowPromotionDetails(string modelName, ExperimentData experiment, string? autoSelectionReason)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule($"[bold]Promoting to Production[/]").LeftJustified());
        AnsiConsole.WriteLine();

        if (autoSelectionReason != null)
            AnsiConsole.MarkupLine($"[grey]Auto-selected ({autoSelectionReason})[/]");

        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey);

        table.AddColumn("[bold]Property[/]");
        table.AddColumn("[bold]Value[/]");

        table.AddRow("Model", $"[cyan]{modelName}[/]");
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
            table.AddRow("Best Trainer", $"[green]{Markup.Escape(experiment.Result.BestTrainer)}[/]");
            table.AddRow("Training Time", $"{experiment.Result.TrainingTimeSeconds:F2}s");
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    private static void ShowPromotionResult(string modelName, string experimentId, string? backupPath)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[green]Promotion Complete![/]").LeftJustified());
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{modelName}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Experiment: [cyan]{experimentId}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Status: [green bold]production[/]");
        if (backupPath != null)
        {
            AnsiConsole.MarkupLine($"[green]>[/] Backup: [grey]{backupPath}[/]");
        }
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
        AnsiConsole.MarkupLine($"  mloop predict --name {modelName}");
        AnsiConsole.MarkupLine($"  mloop info --name {modelName}");
        AnsiConsole.WriteLine();
    }

    private static void OutputAsJson(
        string modelName, string experimentId, string? autoSelectionReason,
        string? previousProduction, string? backupPath)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        Console.WriteLine(JsonSerializer.Serialize(new
        {
            Model = modelName,
            ExperimentId = experimentId,
            Status = "promoted",
            SelectedBy = autoSelectionReason ?? "explicit",
            PreviousProduction = previousProduction,
            BackupPath = backupPath,
        }, options));
    }

    private static void WriteError(string message, bool jsonOutput)
    {
        if (jsonOutput)
        {
            Console.WriteLine(JsonSerializer.Serialize(new { error = message }));
        }
        else
        {
            AnsiConsole.MarkupLineInterpolated($"[red]Error:[/] {message}");
        }
    }
}
