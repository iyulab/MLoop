using System.CommandLine;
using Microsoft.ML;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Hooks;
using MLoop.Core.Models;
using MLoop.Core.Preprocessing;
using MLoop.Extensibility;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Preprocessing;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop train - Trains a model using AutoML with multi-model support
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

        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')"
        };

        var labelOption = new Option<string?>("--label", "-l")
        {
            Description = "Name of the label column"
        };

        var taskOption = new Option<string?>("--task", "-t")
        {
            Description = "ML task type (binary-classification, multiclass-classification, regression)"
        };

        var timeOption = new Option<int?>("--time")
        {
            Description = "Training time limit in seconds"
        };

        var metricOption = new Option<string?>("--metric", "-m")
        {
            Description = "Optimization metric (accuracy, auc, f1, r_squared, etc.)"
        };

        var testSplitOption = new Option<double?>("--test-split")
        {
            Description = "Test data split ratio (0.0-1.0)"
        };

        var noPromoteOption = new Option<bool>("--no-promote")
        {
            Description = "Skip automatic promotion to production",
            DefaultValueFactory = _ => false
        };

        var command = new Command("train", "Train a model using AutoML");
        command.Arguments.Add(dataFileArg);
        command.Options.Add(nameOption);
        command.Options.Add(labelOption);
        command.Options.Add(taskOption);
        command.Options.Add(timeOption);
        command.Options.Add(metricOption);
        command.Options.Add(testSplitOption);
        command.Options.Add(noPromoteOption);

        command.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg);
            var name = parseResult.GetValue(nameOption);
            var label = parseResult.GetValue(labelOption);
            var task = parseResult.GetValue(taskOption);
            var time = parseResult.GetValue(timeOption);
            var metric = parseResult.GetValue(metricOption);
            var testSplit = parseResult.GetValue(testSplitOption);
            var noPromote = parseResult.GetValue(noPromoteOption);
            return ExecuteAsync(dataFile, name, label, task, time, metric, testSplit, noPromote);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? dataFile,
        string? modelName,
        string? label,
        string? task,
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

            // Resolve model name (defaults to "default")
            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? ConfigDefaults.DefaultModelName
                : modelName.Trim().ToLowerInvariant();

            // Load configuration
            var configLoader = new ConfigLoader(fileSystem, projectDiscovery);
            var configMerger = new ConfigMerger();

            var userConfig = await configLoader.LoadUserConfigAsync();

            // Build CLI-provided training settings
            TrainingSettings? cliTraining = null;
            if (time.HasValue || !string.IsNullOrEmpty(metric) || testSplit.HasValue)
            {
                cliTraining = new TrainingSettings
                {
                    TimeLimitSeconds = time,
                    Metric = metric,
                    TestSplit = testSplit
                };
            }

            // Get effective model definition (merges config with CLI overrides)
            ModelDefinition effectiveDefinition;
            try
            {
                effectiveDefinition = configMerger.GetEffectiveModelDefinition(
                    userConfig,
                    resolvedModelName,
                    cliLabel: label,
                    cliTask: task,
                    cliTraining: cliTraining);
            }
            catch (InvalidOperationException ex)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] {ex.Message}");
                return 1;
            }

            // Resolve data file path
            string resolvedDataFile;

            if (string.IsNullOrEmpty(dataFile))
            {
                // Try config data path, then auto-discover
                if (!string.IsNullOrEmpty(userConfig.Data?.Train))
                {
                    resolvedDataFile = Path.IsPathRooted(userConfig.Data.Train)
                        ? userConfig.Data.Train
                        : Path.Combine(projectRoot, userConfig.Data.Train);
                }
                else
                {
                    var datasetDiscovery = new DatasetDiscovery(fileSystem);
                    var datasets = datasetDiscovery.FindDatasets(projectRoot);

                    if (datasets == null || string.IsNullOrEmpty(datasets.TrainPath))
                    {
                        AnsiConsole.MarkupLine("[red]Error:[/] No data file specified and datasets/train.csv not found.");
                        AnsiConsole.MarkupLine("[yellow]Tip:[/] Create datasets/train.csv or specify a file: mloop train <data-file>");
                        return 1;
                    }

                    resolvedDataFile = datasets.TrainPath;
                }
                AnsiConsole.MarkupLine($"[green]>[/] Auto-detected: [cyan]{Path.GetRelativePath(projectRoot, resolvedDataFile)}[/]");
            }
            else
            {
                resolvedDataFile = Path.IsPathRooted(dataFile)
                    ? dataFile
                    : Path.Combine(projectRoot, dataFile);

                if (!File.Exists(resolvedDataFile))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] Data file not found: {resolvedDataFile}");
                    return 1;
                }
            }

            // Execute preprocessing scripts if available
            var preprocessingEngine = new PreprocessingEngine(
                projectRoot,
                new TrainCommandLogger());

            if (preprocessingEngine.HasPreprocessingScripts())
            {
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine("[blue]Running preprocessing scripts...[/]");

                try
                {
                    resolvedDataFile = await preprocessingEngine.ExecuteAsync(resolvedDataFile, effectiveDefinition.Label);
                    AnsiConsole.WriteLine();
                }
                catch (InvalidOperationException ex)
                {
                    AnsiConsole.MarkupLine("[red]Preprocessing failed:[/]");
                    AnsiConsole.WriteLine(ex.Message);
                    return 1;
                }
            }

            // Display training configuration
            DisplayTrainingConfig(resolvedDataFile, resolvedModelName, effectiveDefinition);

            // Validate label column exists
            await ValidateLabelColumnAsync(resolvedDataFile, effectiveDefinition.Label, resolvedModelName);

            // Initialize training components
            var modelNameResolver = new ModelNameResolver(fileSystem, projectDiscovery, configLoader);
            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var trainingEngine = new TrainingEngine(
                fileSystem,
                experimentStore,
                projectRoot,
                new TrainCommandLogger());

            // Ensure model directory structure exists
            await modelNameResolver.EnsureModelDirectoryAsync(resolvedModelName);

            var trainingConfig = new TrainingConfig
            {
                ModelName = resolvedModelName,
                DataFile = Path.GetFullPath(resolvedDataFile),
                LabelColumn = effectiveDefinition.Label,
                Task = effectiveDefinition.Task,
                TimeLimitSeconds = effectiveDefinition.Training?.TimeLimitSeconds ?? ConfigDefaults.DefaultTimeLimitSeconds,
                Metric = effectiveDefinition.Training?.Metric ?? ConfigDefaults.DefaultMetric,
                TestSplit = effectiveDefinition.Training?.TestSplit ?? ConfigDefaults.DefaultTestSplit
            };

            // Initialize hook engine
            var hookEngine = new HookEngine(projectRoot, new TrainCommandLogger());

            // Execute PreTrain hooks
            var mlContext = new MLContext(seed: 0);

            var preTrainContext = new HookContext
            {
                HookType = HookType.PreTrain,
                HookName = "pre-train",
                MLContext = mlContext,
                DataView = null,  // Hooks can load data themselves using DataFile from metadata
                Model = null,
                ExperimentResult = null,
                Metrics = null,
                ProjectRoot = projectRoot,
                Logger = new TrainCommandLogger(),
                Metadata = new Dictionary<string, object>
                {
                    ["DataFile"] = trainingConfig.DataFile,
                    ["LabelColumn"] = trainingConfig.LabelColumn,
                    ["TaskType"] = trainingConfig.Task,
                    ["ModelName"] = trainingConfig.ModelName,
                    ["TimeLimit"] = trainingConfig.TimeLimitSeconds
                }
            };

            var preTrainSuccess = await hookEngine.ExecuteHooksAsync(HookType.PreTrain, preTrainContext);
            if (!preTrainSuccess)
            {
                AnsiConsole.MarkupLine("[red]Training aborted by pre-train hook[/]");
                return 1;
            }

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
                    var progressTask = ctx.AddTask($"[green]Training {resolvedModelName}...[/]", maxValue: 100);

                    var progress = new Progress<TrainingProgress>(p =>
                    {
                        progressTask.Description = $"[green]Trial {p.TrialNumber}:[/] {p.TrainerName} - {p.MetricName}={p.Metric:F4}";

                        var progressPercent = Math.Min(
                            (p.ElapsedSeconds / trainingConfig.TimeLimitSeconds) * 100,
                            99);
                        progressTask.Value = progressPercent;
                    });

                    result = await trainingEngine.TrainAsync(trainingConfig, progress, CancellationToken.None);
                    progressTask.Value = 100;
                    progressTask.StopTask();
                });

            if (result == null)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Training failed for model '[cyan]{resolvedModelName}[/]'");
                return 1;
            }

            // Execute PostTrain hooks
            var postTrainContext = new HookContext
            {
                HookType = HookType.PostTrain,
                HookName = "post-train",
                MLContext = mlContext,
                DataView = null,  // Training data not needed for post-train
                Model = null,  // Model not directly available, use ModelPath from metadata
                ExperimentResult = result,
                Metrics = result.Metrics,
                ProjectRoot = projectRoot,
                Logger = new TrainCommandLogger(),
                Metadata = new Dictionary<string, object>
                {
                    ["LabelColumn"] = trainingConfig.LabelColumn,
                    ["TaskType"] = trainingConfig.Task,
                    ["ModelName"] = trainingConfig.ModelName,
                    ["ExperimentId"] = result.ExperimentId,
                    ["BestTrainer"] = result.BestTrainer,
                    ["ModelPath"] = result.ModelPath,
                    ["TrainingTimeSeconds"] = result.TrainingTimeSeconds
                }
            };

            var postTrainSuccess = await hookEngine.ExecuteHooksAsync(HookType.PostTrain, postTrainContext);
            if (!postTrainSuccess)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] Post-train hook aborted, but model was saved successfully");
                // Don't return 1 here - training succeeded, only post-processing failed
            }

            // Display results
            DisplayResults(result, resolvedModelName);

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
                            resolvedModelName,
                            result.ExperimentId,
                            primaryMetric,
                            CancellationToken.None);

                        if (promoted)
                        {
                            ctx.Status("[green]Model promoted to production![/]");
                            AnsiConsole.MarkupLine($"[green]Model promoted to production![/]");
                            AnsiConsole.WriteLine($"   Better {primaryMetric} than current production model");
                        }
                        else
                        {
                            ctx.Status("[yellow]Model saved to staging[/]");
                            AnsiConsole.MarkupLine("[yellow]Model saved to staging[/]");
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

    private static void DisplayTrainingConfig(string dataFile, string modelName, ModelDefinition definition)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[blue]Training Configuration[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table()
            .BorderColor(Color.Grey)
            .AddColumn("Setting")
            .AddColumn("Value");

        table.AddRow("Model", $"[cyan]{modelName}[/]");
        table.AddRow("Task", definition.Task);
        table.AddRow("Data File", dataFile);
        table.AddRow("Label Column", definition.Label);
        table.AddRow("Time Limit", $"{definition.Training?.TimeLimitSeconds ?? ConfigDefaults.DefaultTimeLimitSeconds}s");
        table.AddRow("Metric", definition.Training?.Metric ?? ConfigDefaults.DefaultMetric);
        table.AddRow("Test Split", $"{(definition.Training?.TestSplit ?? ConfigDefaults.DefaultTestSplit) * 100:F0}%");

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    private static void DisplayResults(TrainingResult result, string modelName)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[green]Training Complete![/]").LeftJustified());
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{modelName}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Experiment ID: [blue]{result.ExperimentId}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Best Trainer: [yellow]{result.BestTrainer}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Training Time: [cyan]{result.TrainingTimeSeconds:F2}s[/]");
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
        AnsiConsole.MarkupLine($"  mloop list --name {modelName}");
        AnsiConsole.MarkupLine($"  mloop predict data.csv --name {modelName}");
        AnsiConsole.MarkupLine($"  mloop promote {result.ExperimentId} --name {modelName}");
        AnsiConsole.WriteLine();
    }

    private static async Task ValidateLabelColumnAsync(string dataFilePath, string labelColumn, string modelName)
    {
        var csvHelper = new MLoop.Core.Data.CsvHelperImpl();
        var data = await csvHelper.ReadAsync(dataFilePath);

        if (data.Count == 0)
        {
            throw new InvalidOperationException($"Data file is empty: {dataFilePath}");
        }

        var firstRow = data[0];
        var availableColumns = firstRow.Keys.ToArray();

        if (!firstRow.ContainsKey(labelColumn))
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[red]Error:[/] Label column not found in data for model '[cyan]{modelName}[/]'");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"  [yellow]Label specified:[/] '{labelColumn}'");
            AnsiConsole.MarkupLine($"  [yellow]Available columns:[/] {string.Join(", ", availableColumns)}");
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[yellow]Tip:[/] Update the label in mloop.yaml or use --label option for --name {modelName}");
            AnsiConsole.WriteLine();

            throw new ArgumentException(
                $"Label column '{labelColumn}' not found in data for model '{modelName}'.\n" +
                $"Available columns: {string.Join(", ", availableColumns)}",
                nameof(labelColumn));
        }
    }

    private class TrainCommandLogger : ILogger
    {
        public void Debug(string message) => AnsiConsole.MarkupLine($"[grey]{message}[/]");
        public void Info(string message) => AnsiConsole.WriteLine(message);
        public void Warning(string message) => AnsiConsole.MarkupLine($"[yellow]{message}[/]");
        public void Error(string message) => AnsiConsole.MarkupLine($"[red]{message}[/]");
        public void Error(string message, Exception exception)
        {
            AnsiConsole.MarkupLine($"[red]{message}[/]");
            AnsiConsole.WriteException(exception);
        }
    }
}
