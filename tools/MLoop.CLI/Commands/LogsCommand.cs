using System.CommandLine;
using System.Text.Json;
using MLoop.CLI.Infrastructure;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.Display;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop logs - View prediction logs for monitoring and analysis
/// </summary>
public static class LogsCommand
{
    public static Command Create()
    {
        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')"
        };

        var limitOption = new Option<int>("--limit", "-l")
        {
            Description = "Maximum number of entries to show",
            DefaultValueFactory = _ => 20
        };

        var fromOption = new Option<DateTime?>("--from")
        {
            Description = "Start date filter (yyyy-MM-dd)"
        };

        var toOption = new Option<DateTime?>("--to")
        {
            Description = "End date filter (yyyy-MM-dd)"
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output in JSON format"
        };

        var command = new Command("logs", "View prediction logs");
        command.Options.Add(nameOption);
        command.Options.Add(limitOption);
        command.Options.Add(fromOption);
        command.Options.Add(toOption);
        command.Options.Add(jsonOption);
        command.Subcommands.Add(CreatePrune());

        command.SetAction((parseResult) =>
        {
            var name = parseResult.GetValue(nameOption);
            var limit = parseResult.GetValue(limitOption);
            var from = parseResult.GetValue(fromOption);
            var to = parseResult.GetValue(toOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteAsync(name, limit, from, to, json);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? modelName,
        int limit,
        DateTime? from,
        DateTime? to,
        bool jsonOutput)
    {
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

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
                ErrorConsole.Error(
                    ProjectDiscovery.NotInsideProjectCause,
                    ProjectDiscovery.NotInsideProjectGuidance);
                return 1;
            }

            // Resolve model name
            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? null
                : modelName.Trim().ToLowerInvariant();

            // Create logger and fetch logs
            var logger = new FilePredictionLogger(projectRoot);

            var (fromOffset, toOffset) = DateRangeFilter.ToFilterRange(from, to);

            var logs = await logger.GetLogsAsync(
                resolvedModelName,
                fromOffset,
                toOffset,
                limit);

            // "no logs yet" is an ordinary state, not a failure — but under --json it still has to
            // come back as an empty payload rather than prose, so the hint is human-only and the
            // JSON path falls through to emit an empty array.
            if (!logs.Any() && !jsonOutput)
            {
                if (resolvedModelName != null)
                {
                    AnsiConsole.MarkupLine($"[yellow]No prediction logs found for model '[cyan]{resolvedModelName}[/]'.[/]");
                }
                else
                {
                    AnsiConsole.MarkupLine("[yellow]No prediction logs found.[/]");
                }
                AnsiConsole.MarkupLine("[grey]Use [blue]mloop predict --log[/] to log predictions.[/]");
                return 0;
            }

            if (jsonOutput)
            {
                OutputAsJson(logs);
            }
            else
            {
                OutputAsTable(logs, resolvedModelName);
            }

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "logs");
            return 1;
        }
    }

    private static Command CreatePrune()
    {
        var daysOption = new Option<int?>("--days")
        {
            Description = "Days to keep besides today (UTC); default: retention.days in mloop.yaml"
        };
        var dryRunOption = new Option<bool>("--dry-run")
        {
            Description = "Report what would be removed without removing it"
        };
        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output in JSON format"
        };

        var command = new Command("prune",
            "Remove prediction logs and feedback older than the retention period (whole days, every model)");
        command.Options.Add(daysOption);
        command.Options.Add(dryRunOption);
        command.Options.Add(jsonOption);
        command.SetAction(parseResult => PruneAsync(
            parseResult.GetValue(daysOption), parseResult.GetValue(dryRunOption), parseResult.GetValue(jsonOption)));
        return command;
    }

    private static async Task<int> PruneAsync(int? days, bool dryRun, bool jsonOutput)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;
        try
        {
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);
            string projectRoot;
            try
            {
                projectRoot = projectDiscovery.FindRoot();
            }
            catch (InvalidOperationException)
            {
                ErrorConsole.Error(ProjectDiscovery.NotInsideProjectCause, ProjectDiscovery.NotInsideProjectGuidance);
                return 1;
            }

            days ??= (await new ConfigLoader(fileSystem, projectDiscovery).LoadUserConfigAsync()).Retention?.Days;
            if (days is not { } keep)
            {
                ErrorConsole.Error("No retention period: pass --days or set retention.days in mloop.yaml.",
                    "Example: mloop logs prune --days 30 --dry-run");
                return 1;
            }
            if (keep < 1)
            {
                ErrorConsole.Error($"The retention period must be at least 1 day (got {keep}).",
                    "Today's logs are always kept; --days counts the days before today.");
                return 1;
            }

            var report = RetentionPruner.Prune(projectRoot, keep, DateTimeOffset.UtcNow, dryRun);
            if (jsonOutput)
            {
                Console.WriteLine(JsonSerializer.Serialize(new
                {
                    days = keep,
                    dryRun,
                    keptFrom = report.Cutoff, // the first UTC day kept — the day-files are named in UTC
                    logFiles = report.LogFiles,
                    feedbackFiles = report.FeedbackFiles,
                    bytes = report.Bytes,
                }, new JsonSerializerOptions { WriteIndented = true }));
                return 0;
            }

            var verb = dryRun ? "Would remove" : "Removed";
            // The days are UTC (the day-files are named in UTC); the boundary is shown in the reader's zone.
            var boundary = TimestampDisplay.LocalToMinutes(report.Cutoff.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc));
            AnsiConsole.MarkupLine(report.Files == 0
                ? $"[green]Nothing logged before {boundary} to remove.[/]"
                : $"[green]{verb} {report.LogFiles} log and {report.FeedbackFiles} feedback day-file(s) — everything before " +
                  $"{boundary}, {report.Bytes / 1024.0 / 1024.0:F1} MB.[/]");
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "logs prune");
            return 1;
        }
    }

    private static void OutputAsJson(IReadOnlyList<PredictionLogEntry> logs)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var output = logs.Select(l => new
        {
            l.Id,
            l.ModelName,
            l.ExperimentId,
            Input = l.Input,
            Output = l.Output,
            l.Confidence,
            Timestamp = l.Timestamp.ToString("o")
        });

        Console.WriteLine(JsonSerializer.Serialize(output, options));
    }

    private static void OutputAsTable(IReadOnlyList<PredictionLogEntry> logs, string? modelName)
    {
        var title = modelName != null
            ? $"Prediction Logs - {modelName}"
            : "Prediction Logs";

        AnsiConsole.Write(new Rule($"[cyan]{title}[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table();
        table.Border(TableBorder.Rounded);
        // The id is what `mloop feedback add --prediction-id` takes; this is where a user finds it.
        // Never wrapped: a copied id split over two lines names a prediction that does not exist.
        table.AddColumn(new TableColumn("ID").NoWrap());
        table.AddColumn(TimestampDisplay.ZoneHeading("Timestamp"));
        table.AddColumn("Model");
        table.AddColumn("Experiment");
        table.AddColumn("Input (summary)");
        table.AddColumn("Output");

        foreach (var log in logs)
        {
            var inputSummary = SummarizeInput(log.Input);
            var outputStr = log.Output?.ToString() ?? "-";

            table.AddRow(
                log.Id ?? "-",
                TimestampDisplay.LocalWithoutZone(log.Timestamp),
                log.ModelName,
                log.ExperimentId.Length > 12 ? log.ExperimentId[..12] + "..." : log.ExperimentId,
                inputSummary,
                outputStr.Length > 30 ? outputStr[..30] + "..." : outputStr
            );
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[grey]Showing {logs.Count} entries[/]");
    }

    internal static string SummarizeInput(IDictionary<string, object> input)
    {
        if (input.Count == 0)
            return "-";

        var first = input.First();
        var summary = $"{first.Key}={first.Value}";

        if (input.Count > 1)
        {
            summary += $" (+{input.Count - 1} more)";
        }

        return summary.Length > 40 ? summary[..40] + "..." : summary;
    }
}
