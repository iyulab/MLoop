using System.CommandLine;
using Microsoft.ML;
using MLoop.Core.Models;
using MLoop.Core.Preprocessing;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Extensibility;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop train - Trains a model using AutoML
/// </summary>
public static class TrainCommand
{
    public static Command Create()
    {
        var dataFileArg = new Argument<string?>("data-file")
        {
            Description = "Path to training data (defaults to datasets/train.csv if omitted)",
            Arity = ArgumentArity.ZeroOrOne
        };

        var labelOption = new Option<string?>("--label")
        {
            Description = "Name of the label column"
        };

        var timeOption = new Option<int?>("--time")
        {
            Description = "Training time limit in seconds"
        };

        var metricOption = new Option<string?>("--metric")
        {
            Description = "Optimization metric (accuracy, auc, f1, etc.)"
        };

        var testSplitOption = new Option<double?>("--test-split")
        {
            Description = "Test data split ratio (0.0-1.0)"
        };

        var noPromoteOption = new Option<bool>("--no-promote")
        {
            Description = "Skip automatic promotion to production"
        };

        var command = new Command("train", "Train a model using AutoML")
        {
            dataFileArg,
            labelOption,
            timeOption,
            metricOption,
            testSplitOption,
            noPromoteOption
        };

        command.SetHandler(
            ExecuteAsync,
            dataFileArg,
            labelOption,
            timeOption,
            metricOption,
            testSplitOption,
            noPromoteOption);

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? dataFile,
        string? label,
        int? time,
        string? metric,
        double? testSplit,
        bool noPromote)
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

            // Resolve data file path (Convention over Configuration)
            string resolvedDataFile;

            if (string.IsNullOrEmpty(dataFile))
            {
                // Auto-discover datasets/train.csv
                var datasetDiscovery = new DatasetDiscovery(fileSystem);
                var datasets = datasetDiscovery.FindDatasets(projectRoot);

                if (datasets == null || string.IsNullOrEmpty(datasets.TrainPath))
                {
                    AnsiConsole.MarkupLine("[red]Error:[/] No data file specified and datasets/train.csv not found.");
                    AnsiConsole.MarkupLine("[yellow]Tip:[/] Create datasets/train.csv or specify a file: mloop train <data-file>");
                    return 1;
                }

                resolvedDataFile = datasets.TrainPath;
                AnsiConsole.MarkupLine($"[green]✓[/] Auto-detected: [cyan]{Path.GetRelativePath(projectRoot, resolvedDataFile)}[/]");
            }
            else
            {
                // Validate explicit data file
                resolvedDataFile = Path.IsPathRooted(dataFile)
                    ? dataFile
                    : Path.Combine(projectRoot, dataFile);

                if (!File.Exists(resolvedDataFile))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] Data file not found: {resolvedDataFile}");
                    return 1;
                }
            }

            // Execute preprocessing scripts if available (Phase 0)
            var preprocessingEngine = new PreprocessingEngine(
                projectRoot,
                new TrainCommandLogger());

            if (preprocessingEngine.HasPreprocessingScripts())
            {
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine("[blue]🔄 Running preprocessing scripts...[/]");

                try
                {
                    resolvedDataFile = await preprocessingEngine.ExecuteAsync(resolvedDataFile, label);
                    AnsiConsole.WriteLine();
                }
                catch (InvalidOperationException ex)
                {
                    AnsiConsole.MarkupLine("[red]Preprocessing failed:[/]");
                    AnsiConsole.WriteLine(ex.Message);
                    return 1;
                }
            }

            // Load and merge configuration
            var configLoader = new ConfigLoader(fileSystem, projectDiscovery);
            var configMerger = new ConfigMerger();

            var projectConfig = await configLoader.LoadProjectConfigAsync();
            var userConfig = await configLoader.LoadUserConfigAsync();
            var defaults = ConfigMerger.CreateDefaults();

            // CLI config from arguments
            var cliConfig = new MLoopConfig
            {
                LabelColumn = label,
                Training = new TrainingSettings
                {
                    TimeLimitSeconds = time ?? 0,
                    Metric = metric ?? string.Empty,
                    TestSplit = testSplit ?? 0
                }
            };

            var finalConfig = configMerger.Merge(cliConfig, userConfig, projectConfig, defaults);

            // Validate final configuration
            if (string.IsNullOrEmpty(finalConfig.LabelColumn))
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Label column not specified.");
                AnsiConsole.MarkupLine("Use --label option or set it in mloop.yaml");
                return 1;
            }

            if (string.IsNullOrEmpty(finalConfig.Task))
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Task type not specified in project config.");
                return 1;
            }

            // Display training configuration
            DisplayTrainingConfig(resolvedDataFile, finalConfig);

            // Initialize training components
            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var trainingEngine = new TrainingEngine(fileSystem, experimentStore);

            var trainingConfig = new TrainingConfig
            {
                DataFile = Path.GetFullPath(resolvedDataFile),
                LabelColumn = finalConfig.LabelColumn,
                Task = finalConfig.Task,
                TimeLimitSeconds = finalConfig.Training?.TimeLimitSeconds ?? 300,
                Metric = finalConfig.Training?.Metric ?? "accuracy",
                TestSplit = finalConfig.Training?.TestSplit ?? 0.2
            };

            // Train with progress display
            TrainingResult? result = null;

            await AnsiConsole.Progress()
                .AutoClear(false)
                .HideCompleted(false)
                .Columns(
                    new TaskDescriptionColumn(),
                    new ProgressBarColumn(),
                    new PercentageColumn(),
                    new ElapsedTimeColumn())
                .StartAsync(async ctx =>
                {
                    var task = ctx.AddTask("[green]Training model...[/]", maxValue: 100);

                    var progress = new Progress<TrainingProgress>(p =>
                    {
                        task.Description = $"[green]Trial {p.TrialNumber}:[/] {p.TrainerName} - {p.MetricName}={p.Metric:F4}";

                        // Estimate progress based on elapsed time
                        var progressPercent = Math.Min(
                            (p.ElapsedSeconds / trainingConfig.TimeLimitSeconds) * 100,
                            99);
                        task.Value = progressPercent;
                    });

                    result = await trainingEngine.TrainAsync(trainingConfig, progress, CancellationToken.None);
                    task.Value = 100;
                    task.StopTask();
                });

            if (result == null)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Training failed");
                return 1;
            }

            // Display results
            DisplayResults(result);

            // Auto-promote to production if enabled
            if (!noPromote)
            {
                AnsiConsole.WriteLine();
                await AnsiConsole.Status()
                    .Spinner(Spinner.Known.Dots)
                    .StartAsync("[yellow]Checking promotion eligibility...[/]", async ctx =>
                    {
                        var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);
                        var primaryMetric = trainingConfig.Metric;

                        var promoted = await modelRegistry.AutoPromoteAsync(
                            result.ExperimentId,
                            primaryMetric,
                            CancellationToken.None);

                        if (promoted)
                        {
                            ctx.Status("[green]Model promoted to production![/]");
                            AnsiConsole.MarkupLine("[green]🚀 Model promoted to production![/]");
                            AnsiConsole.WriteLine($"   Better {primaryMetric} than current production model");
                        }
                        else
                        {
                            ctx.Status("[yellow]Model saved to staging[/]");
                            AnsiConsole.MarkupLine("[yellow]ℹ️  Model saved to staging[/]");
                            AnsiConsole.WriteLine($"   Current production model has better {primaryMetric}");
                        }
                    });

                AnsiConsole.WriteLine();
            }

            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.Markup("[red]Error:[/] ");
            AnsiConsole.WriteLine(ex.Message);

            if (ex.InnerException != null)
            {
                AnsiConsole.Markup("[grey]Details:[/] ");
                AnsiConsole.WriteLine(ex.InnerException.Message);
            }

            return 1;
        }
    }

    private static void DisplayTrainingConfig(string dataFile, MLoopConfig config)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[blue]Training Configuration[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table()
            .BorderColor(Color.Grey)
            .AddColumn("Setting")
            .AddColumn("Value");

        table.AddRow("Project", config.ProjectName ?? "Unknown");
        table.AddRow("Task", config.Task ?? "Unknown");
        table.AddRow("Data File", dataFile);
        table.AddRow("Label Column", config.LabelColumn ?? "Unknown");
        table.AddRow("Time Limit", $"{config.Training?.TimeLimitSeconds ?? 0}s");
        table.AddRow("Metric", config.Training?.Metric ?? "Unknown");
        table.AddRow("Test Split", $"{(config.Training?.TestSplit ?? 0) * 100:F0}%");

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    private static void DisplayResults(TrainingResult result)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[green]Training Complete![/]").LeftJustified());
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine($"[green]✓[/] Experiment ID: [blue]{result.ExperimentId}[/]");
        AnsiConsole.MarkupLine($"[green]✓[/] Best Trainer: [yellow]{result.BestTrainer}[/]");
        AnsiConsole.MarkupLine($"[green]✓[/] Training Time: [cyan]{result.TrainingTimeSeconds:F2}s[/]");
        AnsiConsole.WriteLine();

        // Metrics table
        var metricsTable = new Table()
            .BorderColor(Color.Grey)
            .AddColumn("Metric")
            .AddColumn("Value");

        foreach (var (metricName, metricValue) in result.Metrics.OrderByDescending(m => m.Value))
        {
            var formattedValue = metricValue.ToString("F4");
            var color = metricValue >= 0.9 ? "green" :
                       metricValue >= 0.8 ? "yellow" :
                       metricValue >= 0.7 ? "orange1" : "red";

            metricsTable.AddRow(
                metricName.Replace("_", " ").ToUpperInvariant(),
                $"[{color}]{formattedValue}[/]");
        }

        AnsiConsole.Write(metricsTable);
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine($"[grey]Model saved to:[/] {result.ModelPath}");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
        AnsiConsole.MarkupLine($"  1. mloop evaluate {result.ExperimentId}/model.zip test-data.csv");
        AnsiConsole.MarkupLine($"  2. mloop predict {result.ExperimentId}/model.zip new-data.csv");
        AnsiConsole.MarkupLine($"  3. mloop model promote {result.ExperimentId} staging");
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// Logger implementation for preprocessing within TrainCommand
    /// </summary>
    private class TrainCommandLogger : ILogger
    {
        public void Debug(string message)
        {
            AnsiConsole.MarkupLine($"[grey]{message}[/]");
        }

        public void Info(string message)
        {
            AnsiConsole.WriteLine(message);
        }

        public void Warning(string message)
        {
            AnsiConsole.MarkupLine($"[yellow]{message}[/]");
        }

        public void Error(string message)
        {
            AnsiConsole.MarkupLine($"[red]{message}[/]");
        }
    }
}
