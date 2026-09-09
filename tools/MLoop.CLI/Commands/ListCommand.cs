using System.CommandLine;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Models;
using MLoop.Core.Storage;
using Spectre.Console;
using MLoop.CLI.Infrastructure.Display;

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
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        try
        {
            var ctx = CommandContext.TryCreate();
            if (ctx == null) return 1;

            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? ConfigDefaults.DefaultModelName
                : modelName.Trim().ToLowerInvariant();

            if (!ctx.ExperimentStore.ExperimentExists(resolvedModelName, experimentId))
            {
                var tip = "Run mloop list --name " + resolvedModelName + " to see available experiment IDs.";
                ErrorConsole.Error($"Experiment not found: {resolvedModelName}/{experimentId}", tip);
                // An error exit is still an exit point: under --json the consumer gets a payload to
                // parse rather than an empty stdout, which fails JSON.parse exactly as prose would.
                if (jsonOutput)
                    JsonError.EmitMarkup($"Experiment not found: {resolvedModelName}/{experimentId}", tip);
                return 1;
            }

            var experimentPath = ctx.ExperimentStore.GetExperimentPath(resolvedModelName, experimentId);
            var leaderboardPath = ctx.FileSystem.CombinePath(experimentPath, ExperimentLayout.LeaderboardFileName);

            if (!ctx.FileSystem.FileExists(leaderboardPath))
            {
                var tip = "Only tasks AutoML searches over (e.g. binary/multiclass/regression, or a hand-rolled " +
                          "sweep like clustering's K) produce one; a single fixed-pipeline run has nothing to rank.";
                ErrorConsole.Error($"No trial leaderboard for experiment '{experimentId}'.", tip);
                if (jsonOutput)
                    JsonError.EmitMarkup($"No trial leaderboard for experiment '{experimentId}'.", tip);
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
            var trialNotes = new List<string>();
            foreach (var trial in leaderboard.Trials)
            {
                rank++;
                var metricDisplay = leaderboard.Metric != null && trial.Metrics.TryGetValue(leaderboard.Metric, out var v)
                    ? v.ToString("F4")
                    : "[grey]-[/]";
                var rankDisplay = rank == 1 ? "[green bold]1[/]" : rank.ToString();

                var note = TrainerDisplay.Footnote($"[{rank}]", trial.Trainer);
                if (note != null)
                    trialNotes.Add(note);

                table.AddRow(
                    rankDisplay,
                    $"[cyan]{Markup.Escape(TrainerDisplay.Short(trial.Trainer))}[/]",
                    metricDisplay,
                    $"{trial.RuntimeSeconds:F1}s");
            }

            AnsiConsole.Write(table);
            WriteTrainerNotes(trialNotes);
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
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

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
            // When every listed experiment optimized the same metric, its name belongs in the
            // heading rather than repeated down the column. Saying it once per table instead of once
            // per row is what leaves the Trainer column enough width to hold a trainer.
            var sharedMetric = experimentsList
                .Select(e => e.MetricName)
                .Where(m => !string.IsNullOrEmpty(m))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
            var metricHeading = sharedMetric.Count == 1 ? $"Metric ({sharedMetric[0]})" : "Metric";
            table.AddColumn(new TableColumn($"[bold]{Markup.Escape(metricHeading)}[/]").RightAligned());
            table.AddColumn(new TableColumn("[bold]Stage[/]").Centered());

            var experimentNotes = new List<string>();
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

                var note = TrainerDisplay.Footnote($"[{exp.ExperimentId}]", exp.Trainer);
                if (note != null)
                    experimentNotes.Add(note);

                var trainer = !string.IsNullOrEmpty(exp.BestTrainer)
                    ? $"[cyan]{Markup.Escape(exp.Trainer is not null
                        ? TrainerDisplay.Short(exp.Trainer)
                        : TrainerDisplay.Short(exp.BestTrainer))}[/]"
                    : "[grey]-[/]";

                var metricDisplay = FormatMetric(exp, nameInHeading: sharedMetric.Count == 1);

                // Abbreviated rather than dropped. The ID column marks the production row in colour
                // only, and colour is gone under NO_COLOR, in a pipe, and for a reader with a colour
                // vision deficiency (WCAG 1.4.1) — so this word is that fact's only accessible
                // channel, not a duplicate of the colour. Shortening it gives the width back to the
                // columns that carry more than one value — a long trainer name can still wrap a row
                // on its own — while removing it would trade a wrapped row for an unreadable one.
                var stage = isProduction
                    ? "[green bold]prod[/]"
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
            WriteTrainerNotes(experimentNotes);
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

    /// <summary>
    /// Prints what the Trainer column could not hold: the whole pipeline, and the reason a fallback
    /// stood in. Below the table rather than inside it, because a row is one line and these are not.
    /// </summary>
    private static void WriteTrainerNotes(IReadOnlyList<string> notes)
    {
        if (notes.Count == 0)
            return;

        foreach (var note in notes)
            AnsiConsole.MarkupLine($"[grey]{Markup.Escape(note)}[/]");
    }

    internal static string FormatRelativeTime(DateTime timestamp)
    {
        var local = TimestampDisplay.AsLocal(timestamp);
        var elapsed = DateTimeOffset.Now - local;

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

    /// <summary>
    /// The metric value, and its name when the caller has not already put that in the heading.
    /// </summary>
    internal static string FormatMetric(Infrastructure.FileSystem.ExperimentSummary exp, bool nameInHeading = false)
    {
        if (!exp.BestMetric.HasValue)
            return "[grey]-[/]";

        var value = $"{exp.BestMetric.Value:F4}";
        var name = !nameInHeading && !string.IsNullOrEmpty(exp.MetricName)
            ? $" [grey]({exp.MetricName})[/]"
            : "";
        return $"[yellow]{value}[/]{name}";
    }
}
