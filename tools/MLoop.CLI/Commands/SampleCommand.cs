using System.CommandLine;
using System.CommandLine.Parsing;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using Spectre.Console;
using MLoop.CLI.Infrastructure.Display;

namespace MLoop.CLI.Commands;

/// <summary>
/// CLI command for sampling data.
/// Two modes: prediction log sampling (--model) and general CSV sampling (--from).
/// </summary>
public static class SampleCommand
{
    public static Command Create()
    {
        var command = new Command("sample", "Sample data from CSV files or prediction logs");

        command.Subcommands.Add(CreateCreateCommand());
        command.Subcommands.Add(CreateStatsCommand());

        return command;
    }

    private static Command CreateCreateCommand()
    {
        var nameOption = new Option<string?>("--model", "-m")
        {
            Description = "Model name to sample from (prediction log mode)"
        };

        var fromOption = new Option<string?>("--from", "-f")
        {
            Description = "CSV file to sample from (general-purpose mode)"
        };

        var sizeOption = new Option<int>("--rows", "-n")
        {
            Description = "Number of rows to sample",
            DefaultValueFactory = _ => 1000
        };
        sizeOption.Aliases.Add("--size");

        var strategyOption = new Option<string>("--strategy", "-s")
        {
            Description = "Sampling strategy: random, head, stratified (CSV) / random, recent, feedback-priority, stratified, low-confidence (logs)",
            DefaultValueFactory = _ => "random"
        };

        var outputOption = new Option<string?>("--output", "-o")
        {
            Description = "Output CSV file path"
        };

        var labelOption = new Option<string?>("--label", "-l")
        {
            Description = "Label column for stratified sampling"
        };

        var command = new Command("create", "Create a sampled dataset from CSV file or prediction logs");
        command.Options.Add(nameOption);
        command.Options.Add(fromOption);
        command.Options.Add(sizeOption);
        command.Options.Add(strategyOption);
        command.Options.Add(outputOption);
        command.Options.Add(labelOption);

        command.SetAction((parseResult) =>
        {
            var modelName = parseResult.GetValue(nameOption);
            var fromPath = parseResult.GetValue(fromOption);
            var size = parseResult.GetValue(sizeOption);
            var strategy = parseResult.GetValue(strategyOption)!;
            var output = parseResult.GetValue(outputOption);
            var label = parseResult.GetValue(labelOption);

            if (!string.IsNullOrEmpty(fromPath))
                return ExecuteFromCsvAsync(fromPath, size, strategy, output, label);

            if (!string.IsNullOrEmpty(modelName))
                return ExecuteCreateAsync(modelName, size, strategy, output);

            ErrorConsole.Error(
                "Either --from <csv-file> or --model <name> is required.",
                "Examples: mloop sample create --from train.csv --rows 100, or mloop sample create --model default --rows 500");
            return Task.FromResult(1);
        });

        return command;
    }

    private static Command CreateStatsCommand()
    {
        var nameOption = new Option<string>("--model", "-m")
        {
            Description = "Model name",
            DefaultValueFactory = _ => ConfigDefaults.DefaultModelName
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

    private static async Task<int> ExecuteFromCsvAsync(
        string fromPath,
        int rows,
        string strategyName,
        string? outputPath,
        string? labelColumn)
    {
        try
        {
            var csvStrategy = ParseCsvStrategy(strategyName);
            if (csvStrategy == null)
            {
                ErrorConsole.Error($"Unknown strategy '{strategyName}'", "Valid strategies for CSV: random, head, stratified");
                return 1;
            }

            // Resolve input path
            var resolvedInput = Path.GetFullPath(fromPath);
            if (!File.Exists(resolvedInput))
            {
                ErrorConsole.Error($"File not found: {resolvedInput}", ErrorConsole.PathNotFoundTipCwd());
                return 1;
            }

            // Generate default output path if not provided
            if (string.IsNullOrEmpty(outputPath))
            {
                var dir = Path.GetDirectoryName(resolvedInput) ?? ".";
                var name = Path.GetFileNameWithoutExtension(resolvedInput);
                var ext = Path.GetExtension(resolvedInput);
                outputPath = Path.Combine(dir, $"{name}_sampled{ext}");
            }

            var sampler = new CsvSampler();

            CsvSamplingResult? result = null;
            await AnsiConsole.Status()
                .StartAsync($"Sampling {rows} rows from {Path.GetFileName(fromPath)}...", async ctx =>
                {
                    result = await sampler.SampleAsync(
                        resolvedInput,
                        outputPath,
                        rows,
                        csvStrategy.Value,
                        labelColumn);
                });

            OutputCsvResult(result!);
            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "sample");
            return 1;
        }
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
                ErrorConsole.Error($"Unknown strategy '{Markup.Escape(strategyName)}'",
                    "Valid strategies: random, recent, feedback-priority, stratified, low-confidence");
                return 1;
            }

            var columns = await ResolveRetrainingColumnsAsync(modelName.ToLowerInvariant());
            if (columns is null)
            {
                ErrorConsole.Error(
                    $"No label column is known for model '{Markup.Escape(modelName)}'.",
                    "The dataset is written under the label the model trains on — read from its production experiment, "
                    + "or from `label:` under the model in mloop.yaml.");
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
                        outputPath,
                        columns);

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
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        try
        {
            var projectRoot = FindProjectRoot();
            if (projectRoot == null) return 1;

            var sampler = new FileDataSampler(projectRoot);
            var stats = await sampler.GetStatisticsAsync(modelName.ToLowerInvariant());

            // Same as logs: an empty history is an ordinary state, and --json must still report it
            // as a payload (all-zero stats) instead of prose on stdout.
            if (stats.TotalPredictions == 0 && !jsonOutput)
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

    internal static SamplingStrategy? ParseStrategy(string name)
    {
        return name.ToLowerInvariant() switch
        {
            "random" => SamplingStrategy.Random,
            "recent" => SamplingStrategy.Recent,
            "feedback-priority" or "feedbackpriority" => SamplingStrategy.FeedbackPriority,
            "stratified" => SamplingStrategy.Stratified,
            "low-confidence" or "lowconfidence" => SamplingStrategy.LowConfidence,
            _ => null
        };
    }

    internal static CsvSamplingStrategy? ParseCsvStrategy(string name)
    {
        return name.ToLowerInvariant() switch
        {
            "random" => CsvSamplingStrategy.Random,
            "head" => CsvSamplingStrategy.Head,
            "stratified" => CsvSamplingStrategy.Stratified,
            _ => null
        };
    }

    private static void OutputCsvResult(CsvSamplingResult result)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[cyan]CSV Sampling Complete[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var grid = new Grid();
        grid.AddColumn();
        grid.AddColumn();

        grid.AddRow("[grey]Output File:[/]", $"[white]{result.OutputPath}[/]");
        grid.AddRow("[grey]Rows Sampled:[/]", $"[green]{result.SampledCount:N0}[/]");
        grid.AddRow("[grey]Total Rows:[/]", $"[white]{result.TotalRows:N0}[/]");
        grid.AddRow("[grey]Strategy:[/]", $"[yellow]{result.StrategyUsed}[/]");

        AnsiConsole.Write(grid);
        AnsiConsole.WriteLine();
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
        grid.AddRow("[grey]With Feedback:[/]", $"[white]{result.LabeledCount:N0}[/]");
        grid.AddRow("[grey]Total Available:[/]", $"[white]{result.TotalAvailable:N0}[/]");
        grid.AddRow("[grey]Strategy Used:[/]", $"[yellow]{result.StrategyUsed}[/]");
        if (result.LabelColumn is not null)
            grid.AddRow("[grey]Label Column:[/]", $"[white]{Markup.Escape(result.LabelColumn)}[/]");
        if (result.WeightColumn is not null)
            grid.AddRow("[grey]Weight Column:[/]", $"[white]{Markup.Escape(result.WeightColumn)}[/]");
        grid.AddRow("[grey]Created At:[/]", $"[grey]{TimestampDisplay.Local(result.CreatedAt)}[/]");

        AnsiConsole.Write(grid);
        AnsiConsole.WriteLine();
        ValueLine.Write("[green]✓[/] Dataset ready for retraining at ", result.OutputPath);
        if (result.LabeledCount < result.SampledCount)
            AnsiConsole.MarkupLine(
                $"[grey]{result.SampledCount - result.LabeledCount:N0} rows have no feedback yet — their label is empty, and training drops them.[/]");
        if (result.WeightColumn is not null)
            AnsiConsole.MarkupLine(
                $"[grey]Rows are weighted: train with [blue]--weight {Markup.Escape(result.WeightColumn)}[/] (or `weight:` in mloop.yaml) so they count as recorded.[/]");
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
                // Two dates, one zone: said once at the end rather than twice inline, the same
                // trade the column headings make.
                $"[grey]{TimestampDisplay.ZoneHeading($"{TimestampDisplay.LocalDate(stats.OldestEntry)} to {TimestampDisplay.LocalDate(stats.NewestEntry)}")}[/]");
        }

        AnsiConsole.Write(grid);
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// The columns the model trains with: its production experiment's label and weight, else the model's
    /// definition in mloop.yaml. <c>null</c> when neither names a label.
    /// </summary>
    private static async Task<RetrainingColumns?> ResolveRetrainingColumnsAsync(string modelName)
    {
        var fileSystem = new FileSystemManager();
        var discovery = new ProjectDiscovery(fileSystem);
        var store = new ExperimentStore(fileSystem, discovery);
        var registry = new ModelRegistry(fileSystem, discovery, store);

        var production = await registry.GetProductionAsync(modelName, CancellationToken.None);
        if (production?.ExperimentId is { } id && store.ExperimentExists(modelName, id))
        {
            var experiment = await store.LoadAsync(modelName, id, CancellationToken.None);
            if (!string.IsNullOrWhiteSpace(experiment.Config.LabelColumn))
                return new RetrainingColumns(experiment.Config.LabelColumn, experiment.Config.WeightColumn);
        }

        var config = await new ConfigLoader(fileSystem, discovery).LoadUserConfigAsync();
        return config.Models.TryGetValue(modelName, out var definition) && !string.IsNullOrWhiteSpace(definition.Label)
            ? new RetrainingColumns(definition.Label, definition.Weight)
            : null;
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
}
