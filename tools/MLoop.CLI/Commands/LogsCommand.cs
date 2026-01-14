using System.CommandLine;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
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

            // Resolve model name
            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? null
                : modelName.Trim().ToLowerInvariant();

            // Create logger and fetch logs
            var logger = new FilePredictionLogger(projectRoot);

            DateTimeOffset? fromOffset = from.HasValue
                ? new DateTimeOffset(from.Value, TimeSpan.Zero)
                : null;
            DateTimeOffset? toOffset = to.HasValue
                ? new DateTimeOffset(to.Value.AddDays(1).AddTicks(-1), TimeSpan.Zero)
                : null;

            var logs = await logger.GetLogsAsync(
                resolvedModelName,
                fromOffset,
                toOffset,
                limit);

            if (!logs.Any())
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
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
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
        table.AddColumn("Timestamp");
        table.AddColumn("Model");
        table.AddColumn("Experiment");
        table.AddColumn("Input (summary)");
        table.AddColumn("Output");

        foreach (var log in logs)
        {
            var inputSummary = SummarizeInput(log.Input);
            var outputStr = log.Output?.ToString() ?? "-";

            table.AddRow(
                log.Timestamp.LocalDateTime.ToString("yyyy-MM-dd HH:mm:ss"),
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

    private static string SummarizeInput(IDictionary<string, object> input)
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
