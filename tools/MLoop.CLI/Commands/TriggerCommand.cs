using System.CommandLine;
using System.CommandLine.Parsing;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.DataStore.Services;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// CLI command for evaluating retraining triggers.
/// Checks if model retraining should be triggered based on feedback metrics.
/// </summary>
public static class TriggerCommand
{
    public static Command Create()
    {
        var command = new Command("trigger", "Evaluate retraining triggers for models");

        command.Subcommands.Add(CreateCheckCommand());

        return command;
    }

    private static Command CreateCheckCommand()
    {
        var nameOption = new Option<string>("--name", "-n", "--model", "-m")
        {
            Description = "Model name to evaluate",
            Required = true
        };

        var accuracyOption = new Option<double?>("--accuracy", "-a")
        {
            Description = "Accuracy threshold (0.0-1.0). Triggers if accuracy drops below this value."
        };

        var feedbackOption = new Option<int?>("--feedback", "-f")
        {
            Description = "Feedback volume threshold. Triggers if feedback count exceeds this value."
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output in JSON format"
        };

        var command = new Command("check", "Check if retraining should be triggered");
        command.Options.Add(nameOption);
        command.Options.Add(accuracyOption);
        command.Options.Add(feedbackOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var modelName = parseResult.GetValue(nameOption)!;
            var accuracy = parseResult.GetValue(accuracyOption);
            var feedback = parseResult.GetValue(feedbackOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteCheckAsync(modelName, accuracy, feedback, json);
        });

        return command;
    }

    private static async Task<int> ExecuteCheckAsync(
        string modelName,
        double? accuracyThreshold,
        int? feedbackThreshold,
        bool jsonOutput)
    {
        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var feedbackCollector = new FileFeedbackCollector(projectRoot);
            var trigger = new FeedbackBasedTrigger(feedbackCollector);

            // Build conditions
            var conditions = BuildConditions(accuracyThreshold, feedbackThreshold);
            if (conditions.Count == 0)
            {
                // Use default conditions if none specified
                conditions = (await trigger.GetDefaultConditionsAsync(modelName.ToLowerInvariant()))
                    .ToList();
            }

            TriggerEvaluation result = null!;

            await AnsiConsole.Status()
                .StartAsync($"Evaluating triggers for {modelName}...", async ctx =>
                {
                    result = await trigger.EvaluateAsync(
                        modelName.ToLowerInvariant(),
                        conditions);
                });

            if (jsonOutput)
            {
                OutputResultAsJson(result);
            }
            else
            {
                OutputResultAsPanel(result, modelName);
            }

            // Exit code: 0 = success (check completed), 1 = error
            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
            return 1;
        }
    }

    private static List<RetrainingCondition> BuildConditions(
        double? accuracyThreshold,
        int? feedbackThreshold)
    {
        var conditions = new List<RetrainingCondition>();

        if (accuracyThreshold.HasValue)
        {
            conditions.Add(new RetrainingCondition(
                ConditionType.AccuracyDrop,
                "accuracy_threshold",
                accuracyThreshold.Value,
                $"Accuracy below {accuracyThreshold.Value:P0}"));
        }

        if (feedbackThreshold.HasValue)
        {
            conditions.Add(new RetrainingCondition(
                ConditionType.FeedbackVolume,
                "feedback_threshold",
                feedbackThreshold.Value,
                $"Feedback count >= {feedbackThreshold.Value}"));
        }

        return conditions;
    }

    private static void OutputResultAsJson(TriggerEvaluation result)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var output = new
        {
            result.ShouldRetrain,
            result.RecommendedAction,
            EvaluatedAt = result.EvaluatedAt.ToString("o"),
            Conditions = result.ConditionResults.Select(r => new
            {
                Name = r.Condition.Name,
                Type = r.Condition.Type.ToString(),
                Threshold = r.Condition.Threshold,
                r.CurrentValue,
                r.IsMet,
                r.Details
            })
        };

        Console.WriteLine(JsonSerializer.Serialize(output, options));
    }

    private static void OutputResultAsPanel(TriggerEvaluation result, string modelName)
    {
        AnsiConsole.WriteLine();

        // Header with verdict
        var verdictColor = result.ShouldRetrain ? "green" : "yellow";
        var verdictText = result.ShouldRetrain ? "RETRAIN RECOMMENDED" : "NO RETRAINING NEEDED";
        var verdictIcon = result.ShouldRetrain ? "[green]●[/]" : "[yellow]○[/]";

        AnsiConsole.Write(new Rule($"[cyan]Trigger Evaluation - {modelName}[/]").LeftJustified());
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"  {verdictIcon} [{verdictColor}]{verdictText}[/]");
        AnsiConsole.WriteLine();

        // Conditions table
        var table = new Table()
            .Border(TableBorder.Rounded)
            .AddColumn(new TableColumn("[grey]Condition[/]").LeftAligned())
            .AddColumn(new TableColumn("[grey]Type[/]").Centered())
            .AddColumn(new TableColumn("[grey]Threshold[/]").RightAligned())
            .AddColumn(new TableColumn("[grey]Current[/]").RightAligned())
            .AddColumn(new TableColumn("[grey]Status[/]").Centered());

        foreach (var cr in result.ConditionResults)
        {
            var statusIcon = cr.IsMet ? "[green]✓ MET[/]" : "[grey]✗ NOT MET[/]";
            var thresholdStr = FormatThreshold(cr.Condition);
            var currentStr = FormatCurrentValue(cr);

            table.AddRow(
                $"[white]{cr.Condition.Name}[/]",
                $"[grey]{cr.Condition.Type}[/]",
                thresholdStr,
                currentStr,
                statusIcon);
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();

        // Recommendation
        if (!string.IsNullOrEmpty(result.RecommendedAction))
        {
            AnsiConsole.MarkupLine($"[grey]Recommendation:[/] {result.RecommendedAction}");
        }

        AnsiConsole.MarkupLine($"[grey]Evaluated at:[/] {result.EvaluatedAt.LocalDateTime:yyyy-MM-dd HH:mm:ss}");
        AnsiConsole.WriteLine();

        // Usage hint
        if (result.ShouldRetrain)
        {
            AnsiConsole.MarkupLine("[grey]Run [blue]mloop sample create --model {0}[/] to create retraining dataset.[/]",
                modelName);
        }
    }

    private static string FormatThreshold(RetrainingCondition condition)
    {
        return condition.Type switch
        {
            ConditionType.AccuracyDrop => $"[yellow]< {condition.Threshold:P0}[/]",
            ConditionType.FeedbackVolume => $"[yellow]>= {condition.Threshold:N0}[/]",
            ConditionType.TimeBased => $"[yellow]> {condition.Threshold:N0} days[/]",
            _ => $"[yellow]{condition.Threshold}[/]"
        };
    }

    private static string FormatCurrentValue(ConditionResult result)
    {
        var color = result.IsMet ? "green" : "white";
        return result.Condition.Type switch
        {
            ConditionType.AccuracyDrop => $"[{color}]{result.CurrentValue:P1}[/]",
            ConditionType.FeedbackVolume => $"[{color}]{result.CurrentValue:N0}[/]",
            ConditionType.TimeBased => $"[{color}]{result.CurrentValue:N0} days[/]",
            _ => $"[{color}]{result.CurrentValue}[/]"
        };
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
}
