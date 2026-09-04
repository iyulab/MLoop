using System.CommandLine;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Models;
using MLoop.Core.Storage;
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

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output in JSON format (machine-readable)"
        };

        var trialsOption = new Option<string?>("--trials")
        {
            Description = "Show the trial leaderboard for one experiment (e.g. clustering's K=2..10 " +
                          "search) instead of the experiment overview. The data already exists in " +
                          "leaderboard.json — this is just a CLI surface for it."
        };

        var command = new Command("list", "List all experiments");
        command.Options.Add(nameOption);
        command.Options.Add(allOption);
        command.Options.Add(jsonOption);
        command.Options.Add(trialsOption);

        command.SetAction((parseResult) =>
        {
            var name = parseResult.GetValue(nameOption);
            var showAll = parseResult.GetValue(allOption);
            var json = parseResult.GetValue(jsonOption);
            var trialsExperimentId = parseResult.GetValue(trialsOption);
            return trialsExperimentId != null
                ? ExecuteTrialsAsync(name, trialsExperimentId, json)
                : ExecuteAsync(name, showAll, json);
        });

        return command;
    }

    /// <summary>
    /// Renders one experiment's <c>leaderboard.json</c> — the trial history a search produced
    /// (e.g. clustering's K=2..10 sweep) that until now had no CLI surface, only the file itself.
    /// </summary>
    private static async Task<int> ExecuteTrialsAsync(string? modelName, string experimentId, bool jsonOutput)
    {
        try
        {
            var ctx = CommandContext.TryCreate();
            if (ctx == null) return 1;

            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? ConfigDefaults.DefaultModelName
                : modelName.Trim().ToLowerInvariant();

            if (!ctx.ExperimentStore.ExperimentExists(resolvedModelName, experimentId))
            {
                ErrorConsole.Error(
                    $"Experiment not found: {resolvedModelName}/{experimentId}",
                    "Run mloop list --name " + resolvedModelName + " to see available experiment IDs.");
                return 1;
            }

            var experimentPath = ctx.ExperimentStore.GetExperimentPath(resolvedModelName, experimentId);
            var leaderboardPath = ctx.FileSystem.CombinePath(experimentPath, ExperimentLayout.LeaderboardFileName);

            if (!ctx.FileSystem.FileExists(leaderboardPath))
            {
                ErrorConsole.Error(
                    $"No trial leaderboard for experiment '{experimentId}'.",
                    "Only tasks AutoML searches over (e.g. binary/multiclass/regression, or a hand-rolled " +
                    "sweep like clustering's K) produce one; a single fixed-pipeline run has nothing to rank.");
                return 1;
            }

            var leaderboard = await ctx.FileSystem.ReadJsonAsync<LeaderboardFile>(leaderboardPath);

            if (jsonOutput)
            {
                Console.WriteLine(JsonSerializer.Serialize(leaderboard, new JsonSerializerOptions
                {
                    WriteIndented = true,
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                }));
                return 0;
            }

            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule($"[bold]Trial leaderboard — {experimentId}[/]").LeftJustified());
            AnsiConsole.WriteLine();

            if (leaderboard.Metric == null)
            {
                AnsiConsole.MarkupLine(
                    "[yellow]Ranked by an unrecognized metric — trials are listed in completion order, not ranked.[/]");
            }
            else
            {
                var arrow = leaderboard.Direction == "lower_is_better" ? "↓ lower is better" : "↑ higher is better";
                AnsiConsole.MarkupLine($"[grey]Ranked by [cyan]{Markup.Escape(leaderboard.Metric)}[/] ({arrow})[/]");
            }
            AnsiConsole.WriteLine();

            var table = new Table().Border(TableBorder.Rounded).BorderColor(Color.Grey);
            table.AddColumn(new TableColumn("[bold]Rank[/]").Centered());
            table.AddColumn(new TableColumn("[bold]Trainer[/]"));
            table.AddColumn(new TableColumn("[bold]" + (leaderboard.Metric ?? "Metric") + "[/]").RightAligned());
            table.AddColumn(new TableColumn("[bold]Runtime[/]").RightAligned());

            var rank = 0;
            foreach (var trial in leaderboard.Trials)
            {
                rank++;
                var metricDisplay = leaderboard.Metric != null && trial.Metrics.TryGetValue(leaderboard.Metric, out var v)
                    ? v.ToString("F4")
                    : "[grey]-[/]";
                var rankDisplay = rank == 1 ? "[green bold]1[/]" : rank.ToString();

                table.AddRow(
                    rankDisplay,
                    $"[cyan]{Markup.Escape(trial.Trainer.Display)}[/]",
                    metricDisplay,
                    $"{trial.RuntimeSeconds:F1}s");
            }

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[grey]{leaderboard.TrialCount} trial(s)[/]");
            AnsiConsole.WriteLine();

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "list");
            return 1;
        }
    }

    /// <summary>Mirrors the anonymous shape <c>ExperimentStore.SaveTrialsAsync</c> writes.</summary>
    private sealed class LeaderboardFile
    {
        public string? Metric { get; init; }
        public string? Direction { get; init; }
        public int TrialCount { get; init; }
        public List<TrialRecord> Trials { get; init; } = [];
    }

    private static async Task<int> ExecuteAsync(string? modelName, bool showAll, bool jsonOutput = false)
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

            // Sort by timestamp (newest first) — same order for both output modes.
            experimentsList = experimentsList.OrderByDescending(e => e.Timestamp).ToList();

            if (jsonOutput)
            {
                OutputAsJson(resolvedModelName, experimentsList, productionDict);
                return 0;
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
            table.AddColumn(new TableColumn("[bold]When[/]"));
            table.AddColumn(new TableColumn("[bold]Status[/]").Centered());
            table.AddColumn(new TableColumn("[bold]Trainer[/]"));
            table.AddColumn(new TableColumn("[bold]Metric[/]").RightAligned());
            table.AddColumn(new TableColumn("[bold]Stage[/]").Centered());

            foreach (var exp in experimentsList)
            {
                var expModelName = exp.ModelName ?? ConfigDefaults.DefaultModelName;
                var isProduction = productionDict.TryGetValue(expModelName, out var prodExpId) && exp.ExperimentId == prodExpId;

                var id = isProduction
                    ? $"[green bold]{exp.ExperimentId}[/]"
                    : $"[cyan]{exp.ExperimentId}[/]";

                var when = FormatRelativeTime(exp.Timestamp);

                var status = exp.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase)
                    ? "[green]Completed[/]"
                    : exp.Status.Equals("Failed", StringComparison.OrdinalIgnoreCase)
                        ? "[red]Failed[/]"
                        : $"[yellow]{exp.Status}[/]";

                var trainer = !string.IsNullOrEmpty(exp.BestTrainer)
                    ? $"[cyan]{Markup.Escape(exp.BestTrainer)}[/]"
                    : "[grey]-[/]";

                var metricDisplay = FormatMetric(exp);

                var stage = isProduction
                    ? "[green bold]Production[/]"
                    : "[grey]-[/]";

                if (resolvedModelName == null)
                {
                    table.AddRow($"[blue]{expModelName}[/]", id, when, status, trainer, metricDisplay, stage);
                }
                else
                {
                    table.AddRow(id, when, status, trainer, metricDisplay, stage);
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
            ErrorSuggestions.DisplayError(ex, "list");
            return 1;
        }
    }

    private static void OutputAsJson(
        string? modelFilter,
        IReadOnlyList<Infrastructure.FileSystem.ExperimentSummary> experiments,
        IReadOnlyDictionary<string, string> productionByModel)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var payload = new
        {
            Model = modelFilter,
            Experiments = experiments.Select(e =>
            {
                var expModelName = e.ModelName ?? ConfigDefaults.DefaultModelName;
                return new
                {
                    ModelName = expModelName,
                    e.ExperimentId,
                    Timestamp = e.Timestamp.ToString("o"),
                    e.Status,
                    e.BestTrainer,
                    e.MetricName,
                    e.BestMetric,
                    IsProduction = productionByModel.TryGetValue(expModelName, out var prodId)
                                   && e.ExperimentId == prodId,
                };
            }),
            Production = productionByModel,
        };

        Console.WriteLine(JsonSerializer.Serialize(payload, options));
    }

    internal static string FormatRelativeTime(DateTime timestamp)
    {
        var local = timestamp.Kind == DateTimeKind.Utc ? timestamp.ToLocalTime() : timestamp;
        var elapsed = DateTime.Now - local;

        var relative = elapsed.TotalMinutes switch
        {
            < 1 => "just now",
            < 60 => $"{(int)elapsed.TotalMinutes}m ago",
            < 1440 => $"{(int)elapsed.TotalHours}h ago",
            < 10080 => $"{(int)elapsed.TotalDays}d ago",
            _ => local.ToString("yyyy-MM-dd")
        };

        return $"[grey]{relative}[/]";
    }

    internal static string FormatMetric(Infrastructure.FileSystem.ExperimentSummary exp)
    {
        if (!exp.BestMetric.HasValue)
            return "[grey]-[/]";

        var value = $"{exp.BestMetric.Value:F4}";
        var name = !string.IsNullOrEmpty(exp.MetricName) ? $" [grey]({exp.MetricName})[/]" : "";
        return $"[yellow]{value}[/]{name}";
    }
}
