using System.CommandLine;
using System.CommandLine.Parsing;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// CLI command for sampling prediction data for retraining.
/// Supports Random, Recent, and FeedbackPriority strategies.
/// </summary>
public static class SampleCommand
{
    public static Command Create()
    {
        var command = new Command("sample", "Create retraining datasets from predictions and feedback");

        command.Subcommands.Add(CreateCreateCommand());
        command.Subcommands.Add(CreateStatsCommand());

        return command;
    }

    private static Command CreateCreateCommand()
    {
        var nameOption = new Option<string>("--model", "-m")
        {
            Description = "Model name to sample from",
            Required = true
        };

        var sizeOption = new Option<int>("--size", "-n")
        {
            Description = "Number of samples to include",
            DefaultValueFactory = _ => 1000
        };

        var strategyOption = new Option<string>("--strategy", "-s")
        {
            Description = "Sampling strategy: random, recent, feedback-priority",
            DefaultValueFactory = _ => "random"
        };

        var outputOption = new Option<string?>("--output", "-o")
        {
            Description = "Output CSV file path (default: .mloop/samples/<model>_<timestamp>.csv)"
        };

        var command = new Command("create", "Create a retraining dataset from logged predictions");
        command.Options.Add(nameOption);
        command.Options.Add(sizeOption);
        command.Options.Add(strategyOption);
        command.Options.Add(outputOption);

        command.SetAction((parseResult) =>
        {
            var modelName = parseResult.GetValue(nameOption)!;
            var size = parseResult.GetValue(sizeOption);
            var strategy = parseResult.GetValue(strategyOption)!;
            var output = parseResult.GetValue(outputOption);
            return ExecuteCreateAsync(modelName, size, strategy, output);
        });

        return command;
    }

    private static Command CreateStatsCommand()
    {
        var nameOption = new Option<string>("--model", "-m")
        {
            Description = "Model name",
            Required = true
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output in JSON format"
        };

        var command = new Command("stats", "Show sampling statistics for a model");
        command.Options.Add(nameOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var modelName = parseResult.GetValue(nameOption)!;
            var json = parseResult.GetValue(jsonOption);
            return ExecuteStatsAsync(modelName, json);
        });

        return command;
    }

    private static async Task<int> ExecuteCreateAsync(
        string modelName,
        int sampleSize,
        string strategyName,
        string? outputPath)
    {
        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var strategy = ParseStrategy(strategyName);
            if (strategy == null)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Unknown strategy '{strategyName}'");
                AnsiConsole.MarkupLine("[grey]Valid strategies: random, recent, feedback-priority[/]");
                return 1;
            }

            // Generate default output path if not provided
            if (string.IsNullOrEmpty(outputPath))
            {
                var samplesDir = Path.Combine(projectRoot, ".mloop", "samples");
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                outputPath = Path.Combine(samplesDir, $"{modelName.ToLowerInvariant()}_{timestamp}.csv");
            }

            var sampler = new FileDataSampler(projectRoot);

            await AnsiConsole.Status()
                .StartAsync($"Sampling {sampleSize} entries from {modelName}...", async ctx =>
                {
                    var result = await sampler.SampleAsync(
                        modelName.ToLowerInvariant(),
                        sampleSize,
                        strategy.Value,
                        outputPath);

                    OutputCreateResult(result);
                });

            return 0;
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("No predictions"))
        {
            ErrorSuggestions.DisplayError(ex, "sample");
            return 1;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "sample");
            return 1;
        }
    }

    private static async Task<int> ExecuteStatsAsync(string modelName, bool jsonOutput)
    {
        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var sampler = new FileDataSampler(projectRoot);
            var stats = await sampler.GetStatisticsAsync(modelName.ToLowerInvariant());

            if (stats.TotalPredictions == 0)
            {
                AnsiConsole.MarkupLine($"[yellow]No predictions found for model '[cyan]{modelName}[/]'.[/]");
                AnsiConsole.MarkupLine("[grey]Log predictions using [blue]mloop predict --log[/] first.[/]");
                return 0;
            }

            if (jsonOutput)
            {
                OutputStatsAsJson(stats);
            }
            else
            {
                OutputStatsAsPanel(stats);
            }

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "sample");
            return 1;
        }
    }

    private static SamplingStrategy? ParseStrategy(string name)
    {
        return name.ToLowerInvariant() switch
        {
            "random" => SamplingStrategy.Random,
            "recent" => SamplingStrategy.Recent,
            "feedback-priority" or "feedbackpriority" => SamplingStrategy.FeedbackPriority,
            _ => null
        };
    }

    private static void OutputCreateResult(SamplingResult result)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[cyan]Sampling Complete[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var grid = new Grid();
        grid.AddColumn();
        grid.AddColumn();

        grid.AddRow("[grey]Output File:[/]", $"[white]{result.OutputPath}[/]");
        grid.AddRow("[grey]Samples Created:[/]", $"[green]{result.SampledCount:N0}[/]");
        grid.AddRow("[grey]Total Available:[/]", $"[white]{result.TotalAvailable:N0}[/]");
        grid.AddRow("[grey]Strategy Used:[/]", $"[yellow]{result.StrategyUsed}[/]");
        grid.AddRow("[grey]Created At:[/]", $"[grey]{result.CreatedAt.LocalDateTime:yyyy-MM-dd HH:mm:ss}[/]");

        AnsiConsole.Write(grid);
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[green]✓[/] Dataset ready for retraining at [blue]{result.OutputPath}[/]");
    }

    private static void OutputStatsAsJson(SamplingStatistics stats)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        Console.WriteLine(JsonSerializer.Serialize(new
        {
            stats.ModelName,
            stats.TotalPredictions,
            stats.PredictionsWithFeedback,
            stats.LowConfidenceCount,
            FeedbackPercentage = stats.TotalPredictions > 0
                ? (double)stats.PredictionsWithFeedback / stats.TotalPredictions
                : 0,
            OldestEntry = stats.OldestEntry != DateTimeOffset.MinValue
                ? stats.OldestEntry.ToString("o")
                : null,
            NewestEntry = stats.NewestEntry != DateTimeOffset.MinValue
                ? stats.NewestEntry.ToString("o")
                : null
        }, options));
    }

    private static void OutputStatsAsPanel(SamplingStatistics stats)
    {
        AnsiConsole.Write(new Rule($"[cyan]Sampling Statistics - {stats.ModelName}[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var grid = new Grid();
        grid.AddColumn();
        grid.AddColumn();

        grid.AddRow("[grey]Total Predictions:[/]", $"[white]{stats.TotalPredictions:N0}[/]");
        grid.AddRow("[grey]With Feedback:[/]", $"[green]{stats.PredictionsWithFeedback:N0}[/]");

        if (stats.TotalPredictions > 0)
        {
            var percentage = (double)stats.PredictionsWithFeedback / stats.TotalPredictions;
            grid.AddRow("[grey]Feedback Coverage:[/]", $"[yellow]{percentage:P1}[/]");
        }

        grid.AddRow("[grey]Low Confidence:[/]", $"[white]{stats.LowConfidenceCount:N0}[/]");

        if (stats.OldestEntry != DateTimeOffset.MinValue)
        {
            grid.AddRow("[grey]Date Range:[/]",
                $"[grey]{stats.OldestEntry.LocalDateTime:yyyy-MM-dd}[/] to [grey]{stats.NewestEntry.LocalDateTime:yyyy-MM-dd}[/]");
        }

        AnsiConsole.Write(grid);
        AnsiConsole.WriteLine();
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
