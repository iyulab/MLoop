using System.CommandLine;
using System.CommandLine.Parsing;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// CLI command for managing prediction feedback.
/// Supports recording feedback, listing entries, and calculating metrics.
/// </summary>
public static class FeedbackCommand
{
    public static Command Create()
    {
        var command = new Command("feedback", "Manage prediction feedback for model monitoring");

        command.Subcommands.Add(CreateAddCommand());
        command.Subcommands.Add(CreateListCommand());
        command.Subcommands.Add(CreateMetricsCommand());

        return command;
    }

    private static Command CreateAddCommand()
    {
        var predictionIdOption = new Option<string>("--prediction-id", "-p")
        {
            Description = "ID of the prediction to provide feedback for",
            Required = true
        };

        var actualValueOption = new Option<string>("--actual-value", "-v")
        {
            Description = "The actual (ground truth) value",
            Required = true
        };

        var sourceOption = new Option<string?>("--source", "-s")
        {
            Description = "Feedback source (e.g., 'user', 'system', 'manual')"
        };

        var command = new Command("add", "Record feedback for a prediction");
        command.Options.Add(predictionIdOption);
        command.Options.Add(actualValueOption);
        command.Options.Add(sourceOption);

        command.SetAction((parseResult) =>
        {
            var predictionId = parseResult.GetValue(predictionIdOption)!;
            var actualValue = parseResult.GetValue(actualValueOption)!;
            var source = parseResult.GetValue(sourceOption);
            return ExecuteAddAsync(predictionId, actualValue, source);
        });

        return command;
    }

    private static Command CreateListCommand()
    {
        var nameOption = new Option<string>("--model", "-m")
        {
            Description = "Model name",
            Required = true
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

        var command = new Command("list", "List feedback entries for a model");
        command.Options.Add(nameOption);
        command.Options.Add(limitOption);
        command.Options.Add(fromOption);
        command.Options.Add(toOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var modelName = parseResult.GetValue(nameOption)!;
            var limit = parseResult.GetValue(limitOption);
            var from = parseResult.GetValue(fromOption);
            var to = parseResult.GetValue(toOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteListAsync(modelName, limit, from, to, json);
        });

        return command;
    }

    private static Command CreateMetricsCommand()
    {
        var nameOption = new Option<string>("--model", "-m")
        {
            Description = "Model name",
            Required = true
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

        var command = new Command("metrics", "Calculate accuracy metrics from feedback");
        command.Options.Add(nameOption);
        command.Options.Add(fromOption);
        command.Options.Add(toOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var modelName = parseResult.GetValue(nameOption)!;
            var from = parseResult.GetValue(fromOption);
            var to = parseResult.GetValue(toOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteMetricsAsync(modelName, from, to, json);
        });

        return command;
    }

    private static async Task<int> ExecuteAddAsync(
        string predictionId,
        string actualValue,
        string? source)
    {
        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var collector = new FileFeedbackCollector(projectRoot);

            await collector.RecordFeedbackAsync(
                predictionId,
                actualValue,
                source);

            AnsiConsole.MarkupLine($"[green]✓[/] Feedback recorded for prediction [cyan]{predictionId}[/]");
            AnsiConsole.MarkupLine($"  Actual value: [yellow]{actualValue}[/]");
            if (source != null)
            {
                AnsiConsole.MarkupLine($"  Source: [grey]{source}[/]");
            }

            return 0;
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("not found"))
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            AnsiConsole.MarkupLine("[grey]Make sure the prediction was logged with [blue]--log[/] option.[/]");
            return 1;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return 1;
        }
    }

    private static async Task<int> ExecuteListAsync(
        string modelName,
        int limit,
        DateTime? from,
        DateTime? to,
        bool jsonOutput)
    {
        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var collector = new FileFeedbackCollector(projectRoot);

            DateTimeOffset? fromOffset = from.HasValue
                ? new DateTimeOffset(from.Value, TimeSpan.Zero)
                : null;
            DateTimeOffset? toOffset = to.HasValue
                ? new DateTimeOffset(to.Value.AddDays(1).AddTicks(-1), TimeSpan.Zero)
                : null;

            var feedback = await collector.GetFeedbackAsync(
                modelName.ToLowerInvariant(),
                fromOffset,
                toOffset,
                limit);

            if (!feedback.Any())
            {
                AnsiConsole.MarkupLine($"[yellow]No feedback found for model '[cyan]{modelName}[/]'.[/]");
                AnsiConsole.MarkupLine("[grey]Use [blue]mloop feedback add[/] to record feedback.[/]");
                return 0;
            }

            if (jsonOutput)
            {
                OutputListAsJson(feedback);
            }
            else
            {
                OutputListAsTable(feedback, modelName);
            }

            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return 1;
        }
    }

    private static async Task<int> ExecuteMetricsAsync(
        string modelName,
        DateTime? from,
        DateTime? to,
        bool jsonOutput)
    {
        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var collector = new FileFeedbackCollector(projectRoot);

            DateTimeOffset? fromOffset = from.HasValue
                ? new DateTimeOffset(from.Value, TimeSpan.Zero)
                : null;
            DateTimeOffset? toOffset = to.HasValue
                ? new DateTimeOffset(to.Value.AddDays(1).AddTicks(-1), TimeSpan.Zero)
                : null;

            var metrics = await collector.CalculateMetricsAsync(
                modelName.ToLowerInvariant(),
                fromOffset,
                toOffset);

            if (metrics.TotalFeedback == 0)
            {
                AnsiConsole.MarkupLine($"[yellow]No feedback data found for model '[cyan]{modelName}[/]'.[/]");
                AnsiConsole.MarkupLine("[grey]Record feedback using [blue]mloop feedback add[/] first.[/]");
                return 0;
            }

            if (jsonOutput)
            {
                OutputMetricsAsJson(metrics);
            }
            else
            {
                OutputMetricsAsPanel(metrics);
            }

            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return 1;
        }
    }

    private static string? FindProjectRoot()
    {
        var fileSystem = new FileSystemManager();
        var projectDiscovery = new ProjectDiscovery(fileSystem);

        try
        {
            return projectDiscovery.FindRoot();
        }
        catch (InvalidOperationException)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] Not inside a MLoop project.");
            AnsiConsole.MarkupLine("Run [blue]mloop init[/] to create a new project.");
            return null;
        }
    }

    private static void OutputListAsJson(IReadOnlyList<FeedbackEntry> feedback)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var output = feedback.Select(f => new
        {
            f.PredictionId,
            f.ModelName,
            f.PredictedValue,
            f.ActualValue,
            f.Source,
            Timestamp = f.Timestamp.ToString("o")
        });

        Console.WriteLine(JsonSerializer.Serialize(output, options));
    }

    private static void OutputListAsTable(IReadOnlyList<FeedbackEntry> feedback, string modelName)
    {
        AnsiConsole.Write(new Rule($"[cyan]Feedback - {modelName}[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table();
        table.Border(TableBorder.Rounded);
        table.AddColumn("Timestamp");
        table.AddColumn("Prediction ID");
        table.AddColumn("Predicted");
        table.AddColumn("Actual");
        table.AddColumn("Match");
        table.AddColumn("Source");

        foreach (var entry in feedback)
        {
            var predictedStr = entry.PredictedValue?.ToString() ?? "-";
            var actualStr = entry.ActualValue?.ToString() ?? "-";
            var isMatch = ValuesMatch(entry.PredictedValue, entry.ActualValue);
            var matchStr = isMatch ? "[green]✓[/]" : "[red]✗[/]";

            table.AddRow(
                entry.Timestamp.LocalDateTime.ToString("yyyy-MM-dd HH:mm:ss"),
                entry.PredictionId.Length > 12 ? entry.PredictionId[..12] + "..." : entry.PredictionId,
                predictedStr.Length > 20 ? predictedStr[..20] + "..." : predictedStr,
                actualStr.Length > 20 ? actualStr[..20] + "..." : actualStr,
                matchStr,
                entry.Source ?? "-"
            );
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[grey]Showing {feedback.Count} entries[/]");
    }

    private static void OutputMetricsAsJson(FeedbackMetrics metrics)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        Console.WriteLine(JsonSerializer.Serialize(new
        {
            metrics.ModelName,
            metrics.TotalPredictions,
            metrics.TotalFeedback,
            metrics.Accuracy,
            metrics.Precision,
            metrics.Recall,
            CalculatedAt = metrics.CalculatedAt.ToString("o")
        }, options));
    }

    private static void OutputMetricsAsPanel(FeedbackMetrics metrics)
    {
        AnsiConsole.Write(new Rule($"[cyan]Feedback Metrics - {metrics.ModelName}[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var grid = new Grid();
        grid.AddColumn();
        grid.AddColumn();

        grid.AddRow("[grey]Total Feedback Entries:[/]", $"[white]{metrics.TotalFeedback}[/]");

        if (metrics.Accuracy.HasValue)
        {
            var accuracyColor = metrics.Accuracy.Value >= 0.9 ? "green"
                : metrics.Accuracy.Value >= 0.7 ? "yellow"
                : "red";
            grid.AddRow("[grey]Accuracy:[/]", $"[{accuracyColor}]{metrics.Accuracy.Value:P2}[/]");
        }
        else
        {
            grid.AddRow("[grey]Accuracy:[/]", "[grey]N/A[/]");
        }

        if (metrics.Precision.HasValue)
        {
            grid.AddRow("[grey]Precision:[/]", $"[white]{metrics.Precision.Value:P2}[/]");
        }

        if (metrics.Recall.HasValue)
        {
            grid.AddRow("[grey]Recall:[/]", $"[white]{metrics.Recall.Value:P2}[/]");
        }

        grid.AddRow("[grey]Calculated At:[/]", $"[grey]{metrics.CalculatedAt.LocalDateTime:yyyy-MM-dd HH:mm:ss}[/]");

        AnsiConsole.Write(grid);
        AnsiConsole.WriteLine();
    }

    private static bool ValuesMatch(object? predicted, object? actual)
    {
        if (predicted == null && actual == null)
            return true;
        if (predicted == null || actual == null)
            return false;

        // String comparison (case-insensitive)
        if (predicted is string predictedStr && actual is string actualStr)
            return string.Equals(predictedStr, actualStr, StringComparison.OrdinalIgnoreCase);

        return predicted.Equals(actual);
    }
}
