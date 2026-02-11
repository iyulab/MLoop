using System.CommandLine;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
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

        var noBackupOption = new Option<bool>("--no-backup")
        {
            Description = "Skip backup of current production model before promotion",
            DefaultValueFactory = _ => false
        };

        var command = new Command("promote", "Promote an experiment to production");
        command.Arguments.Add(experimentIdArg);
        command.Options.Add(nameOption);
        command.Options.Add(forceOption);
        command.Options.Add(noBackupOption);

        command.SetAction((parseResult) =>
        {
            var experimentId = parseResult.GetValue(experimentIdArg)!;
            var name = parseResult.GetValue(nameOption);
            var force = parseResult.GetValue(forceOption);
            var noBackup = parseResult.GetValue(noBackupOption);
            return ExecuteAsync(experimentId, name, force, noBackup);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string experimentId,
        string? modelName,
        bool force,
        bool noBackup)
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

            // Backup current production before promotion
            string? backupPath = null;
            if (!noBackup && currentProduction != null)
            {
                var productionPath = modelRegistry.GetProductionPath(resolvedModelName);
                if (Directory.Exists(productionPath))
                {
                    var backupsDir = Path.Combine(projectRoot, "models", resolvedModelName, "backups");
                    backupPath = Path.Combine(backupsDir, $"{currentProduction.ExperimentId}-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}");
                    Directory.CreateDirectory(backupsDir);
                    CopyDirectory(productionPath, backupPath);
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

            // Record promotion history
            await RecordPromotionHistoryAsync(
                projectRoot, resolvedModelName, experimentId,
                currentProduction?.ExperimentId, "promote");

            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[green]Promotion Complete![/]").LeftJustified());
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
            AnsiConsole.MarkupLine($"[green]>[/] Experiment: [cyan]{experimentId}[/]");
            AnsiConsole.MarkupLine($"[green]>[/] Status: [green bold]production[/]");
            if (backupPath != null)
            {
                AnsiConsole.MarkupLine($"[green]>[/] Backup: [grey]{backupPath}[/]");
            }
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
            AnsiConsole.MarkupLine($"  mloop predict --name {resolvedModelName}");
            AnsiConsole.MarkupLine($"  mloop info --name {resolvedModelName}");
            AnsiConsole.WriteLine();

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "promote");
            return 1;
        }
    }

    private static async Task RecordPromotionHistoryAsync(
        string projectRoot,
        string modelName,
        string experimentId,
        string? previousExpId,
        string action)
    {
        var historyPath = Path.Combine(projectRoot, "models", modelName, "promotion-history.json");
        Directory.CreateDirectory(Path.GetDirectoryName(historyPath)!);

        var records = new List<Dictionary<string, object?>>();
        if (File.Exists(historyPath))
        {
            var json = await File.ReadAllTextAsync(historyPath);
            records = JsonSerializer.Deserialize<List<Dictionary<string, object?>>>(json) ?? [];
        }

        records.Add(new Dictionary<string, object?>
        {
            ["modelName"] = modelName,
            ["experimentId"] = experimentId,
            ["previousExperimentId"] = previousExpId,
            ["action"] = action,
            ["timestamp"] = DateTimeOffset.UtcNow
        });

        var options = new JsonSerializerOptions { WriteIndented = true };
        await File.WriteAllTextAsync(historyPath, JsonSerializer.Serialize(records, options));
    }

    private static void CopyDirectory(string source, string destination)
    {
        Directory.CreateDirectory(destination);

        foreach (var file in Directory.GetFiles(source))
        {
            File.Copy(file, Path.Combine(destination, Path.GetFileName(file)), overwrite: true);
        }

        foreach (var dir in Directory.GetDirectories(source))
        {
            CopyDirectory(dir, Path.Combine(destination, Path.GetFileName(dir)));
        }
    }
}
