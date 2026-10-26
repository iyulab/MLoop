using System.CommandLine;
using System.CommandLine.Parsing;
using System.Globalization;
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
        var predictionIdOption = new Option<string>("--prediction-id")
        {
            Description = "ID of the prediction to provide feedback for",
            Required = true
        };

        var actualValueOption = new Option<string>("--actual-value")
        {
            Description = "The actual (ground truth) value",
            Required = true
        };

        var sourceOption = new Option<string?>("--source")
        {
            Description = "Where the value came from (e.g., 'user', 'reviewer-llm', 'system')"
        };

        var weightOption = new Option<double?>("--weight", "-w")
        {
            Description = "How much this row counts in training and evaluation built from feedback "
                + "(for a row reviewed with probability p, 1/p). Default 1"
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Emit the recorded feedback as JSON to stdout (narration goes to stderr)"
        };

        var command = new Command("add", "Record feedback for a prediction");
        command.Options.Add(predictionIdOption);
        command.Options.Add(actualValueOption);
        command.Options.Add(sourceOption);
        command.Options.Add(weightOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var json = parseResult.GetValue(jsonOption);
            var predictionId = parseResult.GetValue(predictionIdOption)!;
            var actualValue = parseResult.GetValue(actualValueOption)!;
            var source = parseResult.GetValue(sourceOption);
            var weight = parseResult.GetValue(weightOption);
            return ExecuteAddAsync(predictionId, actualValue, source, weight, json);
        });

        return command;
    }

    private static Command CreateListCommand()
    {
        var nameOption = new Option<string>("--name", "-n")
        {
            Description = "Model name",
            DefaultValueFactory = _ => ConfigDefaults.DefaultModelName
        };

        var limitOption = new Option<int>("--limit")
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
        var nameOption = new Option<string>("--name", "-n")
        {
            Description = "Model name",
            DefaultValueFactory = _ => ConfigDefaults.DefaultModelName
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
        string? source,
        double? weight,
        bool jsonOutput)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;
        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var collector = new FileFeedbackCollector(projectRoot);

            await collector.RecordFeedbackAsync(
                predictionId,
                actualValue,
                source,
                weight);

            AnsiConsole.MarkupLine($"[green]✓[/] Feedback recorded for prediction [cyan]{Markup.Escape(predictionId)}[/]");
            AnsiConsole.MarkupLine($"  Actual value: [yellow]{Markup.Escape(actualValue)}[/]");
            if (source != null)
            {
                AnsiConsole.MarkupLine($"  Source: [grey]{Markup.Escape(source)}[/]");
            }
            if (weight is { } w)
            {
                AnsiConsole.MarkupLine($"  Weight: [grey]{w.ToString(CultureInfo.InvariantCulture)}[/]");
            }

            if (jsonOutput)
                Console.WriteLine(JsonSerializer.Serialize(new
                {
                    predictionId,
                    actualValue,
                    source,
                    weight,
                    warnings = JsonOutputScope.Warnings
                }, new JsonSerializerOptions
                {
                    WriteIndented = true,
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                    DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
                }));

            return 0;
        }
        catch (ArgumentOutOfRangeException ex) when (ex.ParamName == "weight")
        {
            ErrorConsole.Error(
                $"--weight {weight?.ToString(CultureInfo.InvariantCulture)} cannot weight a row.",
                "A weight is a finite number that is not negative — for a row reviewed with probability p, 1/p.");
            return 1;
        }
        catch (PredictionNotFoundException ex)
        {
            ErrorConsole.Error(
                Markup.Escape(ex.Message),
                "Prediction ids are in the ID column of `mloop logs`. Only logged predictions have one — "
                + "`mloop predict --log`, or any prediction served by `mloop serve`.");
            return 1;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "feedback");
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
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var collector = new FileFeedbackCollector(projectRoot);

            var (fromOffset, toOffset) = DateRangeFilter.ToFilterRange(from, to);

            var feedback = await collector.GetFeedbackAsync(
                modelName.ToLowerInvariant(),
                fromOffset,
                toOffset,
                limit);

            // Empty is an ordinary state; --json still owes the caller a payload, not prose.
            if (!feedback.Any() && !jsonOutput)
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
            ErrorSuggestions.DisplayError(ex, "feedback");
            return 1;
        }
    }

    private static async Task<int> ExecuteMetricsAsync(
        string modelName,
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
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var collector = new FileFeedbackCollector(projectRoot);

            var (fromOffset, toOffset) = DateRangeFilter.ToFilterRange(from, to);

            var metrics = await collector.CalculateMetricsAsync(
                modelName.ToLowerInvariant(),
                fromOffset,
                toOffset);

            // Empty is an ordinary state; --json still owes the caller a payload, not prose.
            if (metrics.TotalFeedback == 0 && !jsonOutput)
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
            ErrorSuggestions.DisplayError(ex, "feedback");
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
            ErrorConsole.Error(
                ProjectDiscovery.NotInsideProjectCause,
                ProjectDiscovery.NotInsideProjectGuidance);
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
            f.Weight,
            f.ExperimentId,
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
        table.AddColumn(TimestampDisplay.ZoneHeading("Timestamp"));
        table.AddColumn("Prediction ID");
        table.AddColumn("Predicted");
        table.AddColumn("Actual");
        table.AddColumn("Match");
        table.AddColumn("Source");
        var weighted = feedback.Any(e => e.Weight is not null);
        if (weighted)
            table.AddColumn(new TableColumn("Weight").RightAligned());

        foreach (var entry in feedback)
        {
            var predictedStr = entry.PredictedValue?.ToString() ?? "-";
            var actualStr = entry.ActualValue?.ToString() ?? "-";
            var isMatch = FeedbackValueComparer.ValuesMatch(entry.PredictedValue, entry.ActualValue);
            var matchStr = isMatch ? "[green]✓[/]" : "[red]✗[/]";

            // Values are the caller's text — escaped, or a '[' in a label is read as markup.
            var cells = new List<string>
            {
                TimestampDisplay.LocalWithoutZone(entry.Timestamp),
                Markup.Escape(entry.PredictionId.Length > 12 ? entry.PredictionId[..12] + "..." : entry.PredictionId),
                Markup.Escape(predictedStr.Length > 20 ? predictedStr[..20] + "..." : predictedStr),
                Markup.Escape(actualStr.Length > 20 ? actualStr[..20] + "..." : actualStr),
                matchStr,
                Markup.Escape(entry.Source ?? "-")
            };
            if (weighted)
                cells.Add(entry.EffectiveWeight.ToString("0.###", CultureInfo.CurrentCulture));
            table.AddRow(cells.ToArray());
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
            metrics.TotalFeedback,
            metrics.Accuracy,
            metrics.Weighted,
            metrics.EffectiveSampleSize,
            metrics.ExperimentId,
            metrics.UnattributedFeedback,
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
            grid.AddRow(metrics.Weighted ? "[grey]Accuracy (weighted):[/]" : "[grey]Accuracy:[/]",
                $"[{accuracyColor}]{metrics.Accuracy.Value:P2}[/]");
            if (metrics.Weighted && metrics.EffectiveSampleSize is { } effective)
                grid.AddRow("[grey]Effective rows:[/]", $"[white]{effective:0.#}[/] [grey](what the weighted accuracy is worth as evidence)[/]");
        }
        else
        {
            grid.AddRow("[grey]Accuracy:[/]", "[grey]N/A[/]");
        }

        grid.AddRow("[grey]Calculated At:[/]", $"[grey]{TimestampDisplay.Local(metrics.CalculatedAt)}[/]");

        AnsiConsole.Write(grid);
        AnsiConsole.WriteLine();
    }
}
