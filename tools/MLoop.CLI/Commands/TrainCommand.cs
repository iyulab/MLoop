using System.CommandLine;
using Microsoft.ML;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.DataQuality;
using MLoop.Core.Diagnostics;
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

        var analyzeDataOption = new Option<bool>("--analyze-data")
        {
            Description = "Analyze data quality without training (shows preprocessing recommendations)",
            DefaultValueFactory = _ => false
        };

        var generateScriptOption = new Option<string?>("--generate-script")
        {
            Description = "Generate preprocessing script from data quality analysis at specified path"
        };

        var autoMergeOption = new Option<bool>("--auto-merge")
        {
            Description = "Automatically detect and merge CSV files with same schema in datasets/ directory",
            DefaultValueFactory = _ => false
        };

        var dropMissingLabelsOption = new Option<bool?>("--drop-missing-labels")
        {
            Description = "Drop rows with missing label values (default: true for classification, false for regression)"
        };

        var dataOption = new Option<string[]?>("--data", "-d")
        {
            Description = "Path(s) to training data file(s). Supports external paths and multiple files for auto-merge",
            AllowMultipleArgumentsPerToken = true
        };

        var balanceOption = new Option<string?>("--balance", "-b")
        {
            Description = "Class balancing strategy: 'auto' (balance to 10:1 if ratio > 10:1), 'none' (no balancing), or target ratio (e.g., '5' for 5:1). Use without value for 'auto'.",
            Arity = ArgumentArity.ZeroOrOne
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
        command.Options.Add(analyzeDataOption);
        command.Options.Add(generateScriptOption);
        command.Options.Add(autoMergeOption);
        command.Options.Add(dropMissingLabelsOption);
        command.Options.Add(dataOption);
        command.Options.Add(balanceOption);

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
            var analyzeData = parseResult.GetValue(analyzeDataOption);
            var generateScript = parseResult.GetValue(generateScriptOption);
            var autoMerge = parseResult.GetValue(autoMergeOption);
            var dropMissingLabels = parseResult.GetValue(dropMissingLabelsOption);
            var dataPaths = parseResult.GetValue(dataOption);
            var balance = parseResult.GetValue(balanceOption);
            // Handle --balance without argument: default to "auto"
            if (balance == null && parseResult.Tokens.Any(t => t.Value == "--balance" || t.Value == "-b"))
            {
                balance = "auto";
            }
            return ExecuteAsync(dataFile, name, label, task, time, metric, testSplit, noPromote, analyzeData, generateScript, autoMerge, dropMissingLabels, dataPaths, balance);
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
        bool noPromote,
        bool analyzeData,
        string? generateScript,
        bool autoMerge,
        bool? dropMissingLabels,
        string[]? dataPaths,
        string? balance)
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

            // Validate required fields (defensive guard for nullable flow analysis)
            if (string.IsNullOrEmpty(effectiveDefinition.Task) || string.IsNullOrEmpty(effectiveDefinition.Label))
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Task and Label must be specified. Use --task and <label> arguments, or define them in mloop.yaml");
                return 1;
            }

            // Resolve data file path
            string? resolvedDataFile;
            var datasetDiscovery = new DatasetDiscovery(fileSystem);

            // T4.3: Handle --data option for external data paths
            if (dataPaths is { Length: > 0 })
            {
                var csvHelper = new CsvHelperImpl();

                // Resolve all paths (support relative and absolute)
                var resolvedPaths = new List<string>();
                foreach (var path in dataPaths)
                {
                    var resolved = Path.IsPathRooted(path) ? path : Path.GetFullPath(Path.Combine(projectRoot, path));
                    if (!File.Exists(resolved))
                    {
                        AnsiConsole.MarkupLine($"[red]Error:[/] Data file not found: {path}");
                        return 1;
                    }
                    resolvedPaths.Add(resolved);
                }

                if (resolvedPaths.Count == 1)
                {
                    // Single file - use directly
                    resolvedDataFile = resolvedPaths[0];
                    AnsiConsole.MarkupLine($"[green]>[/] Using external data: [cyan]{dataPaths[0]}[/]");
                }
                else
                {
                    // Multiple files - validate and merge
                    AnsiConsole.MarkupLine($"[blue]Merging {resolvedPaths.Count} external data files...[/]");

                    foreach (var file in resolvedPaths)
                    {
                        AnsiConsole.MarkupLine($"    [grey]• {Path.GetFileName(file)}[/]");
                    }

                    var csvMerger = new CsvMerger(csvHelper);

                    // Validate schemas
                    var validation = await csvMerger.ValidateSchemaCompatibilityAsync(resolvedPaths);
                    if (!validation.IsCompatible)
                    {
                        AnsiConsole.MarkupLine($"[red]Error:[/] Schema mismatch between files: {validation.Message}");
                        return 1;
                    }

                    // Merge to temp file in datasets directory
                    var datasetsPath = datasetDiscovery.GetDatasetsPath(projectRoot);
                    if (!Directory.Exists(datasetsPath))
                    {
                        Directory.CreateDirectory(datasetsPath);
                    }

                    var mergedPath = Path.Combine(datasetsPath, "merged_train.csv");
                    var mergeResult = await csvMerger.MergeAsync(resolvedPaths, mergedPath);

                    if (!mergeResult.Success)
                    {
                        AnsiConsole.MarkupLine($"[red]Error:[/] Failed to merge files: {mergeResult.Error}");
                        return 1;
                    }

                    AnsiConsole.MarkupLine($"[green]✓[/] Merged [cyan]{mergeResult.TotalRows}[/] rows from {resolvedPaths.Count} files");
                    foreach (var (fileName, rowCount) in mergeResult.RowsPerFile)
                    {
                        AnsiConsole.MarkupLine($"    [grey]• {fileName}: {rowCount} rows[/]");
                    }
                    AnsiConsole.WriteLine();

                    resolvedDataFile = mergedPath;
                }
            }
            // Auto-merge if requested and no specific data file provided
            else if (autoMerge && string.IsNullOrEmpty(dataFile))
            {
                var datasetsPath = datasetDiscovery.GetDatasetsPath(projectRoot);
                if (Directory.Exists(datasetsPath))
                {
                    var csvHelper = new CsvHelperImpl();
                    var csvMerger = new CsvMerger(csvHelper);

                    AnsiConsole.MarkupLine("[blue]Scanning for mergeable CSV files...[/]");
                    var mergeGroups = await csvMerger.DiscoverMergeableCsvsAsync(datasetsPath);

                    if (mergeGroups.Count > 0)
                    {
                        var primaryGroup = mergeGroups.First();
                        AnsiConsole.MarkupLine($"[green]>[/] Found [cyan]{primaryGroup.FilePaths.Count}[/] files with same schema (pattern: [yellow]{primaryGroup.DetectedPattern}[/])");

                        foreach (var file in primaryGroup.FilePaths)
                        {
                            AnsiConsole.MarkupLine($"    [grey]• {Path.GetFileName(file)}[/]");
                        }

                        // Merge to train.csv
                        var mergedPath = Path.Combine(datasetsPath, "train.csv");
                        var mergeResult = await csvMerger.MergeAsync(primaryGroup.FilePaths, mergedPath);

                        if (mergeResult.Success)
                        {
                            AnsiConsole.MarkupLine($"[green]✓[/] Merged [cyan]{mergeResult.TotalRows}[/] rows into [cyan]train.csv[/]");
                            foreach (var (fileName, rowCount) in mergeResult.RowsPerFile)
                            {
                                AnsiConsole.MarkupLine($"    [grey]• {fileName}: {rowCount} rows[/]");
                            }
                            AnsiConsole.WriteLine();

                            resolvedDataFile = mergedPath;
                        }
                        else
                        {
                            AnsiConsole.MarkupLine($"[yellow]Warning:[/] Auto-merge failed: {mergeResult.Error}");
                            AnsiConsole.MarkupLine("[yellow]Falling back to standard discovery...[/]");
                            resolvedDataFile = await TrainDataValidator.ResolveDataFileAsync(dataFile, userConfig, projectRoot, datasetDiscovery, fileSystem);
                        }
                    }
                    else
                    {
                        AnsiConsole.MarkupLine("[grey]No mergeable CSV groups found, using standard discovery[/]");
                        resolvedDataFile = await TrainDataValidator.ResolveDataFileAsync(dataFile, userConfig, projectRoot, datasetDiscovery, fileSystem);
                    }
                }
                else
                {
                    resolvedDataFile = await TrainDataValidator.ResolveDataFileAsync(dataFile, userConfig, projectRoot, datasetDiscovery, fileSystem);
                }
            }
            else
            {
                resolvedDataFile = await TrainDataValidator.ResolveDataFileAsync(dataFile, userConfig, projectRoot, datasetDiscovery, fileSystem);
            }

            if (resolvedDataFile == null)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] No data file specified and datasets/train.csv not found.");
                AnsiConsole.MarkupLine("[yellow]Tip:[/] Create datasets/train.csv or specify a file: mloop train <data-file>");
                return 1;
            }

            AnsiConsole.MarkupLine($"[green]>[/] Using data: [cyan]{Path.GetRelativePath(projectRoot, resolvedDataFile)}[/]");

            // Track all data files used during pipeline (for unused file scanner)
            var allDataFilesUsed = new List<string> { resolvedDataFile };

            // Handle missing label values (T4.2)
            // Default behavior: drop missing labels for classification tasks
            var isClassificationTask = effectiveDefinition.Task.ToLowerInvariant() switch
            {
                "binary-classification" => true,
                "multiclass-classification" => true,
                "binaryclassification" => true,  // Legacy format
                "multiclassclassification" => true,  // Legacy format
                "classification" => true,
                _ => false
            };

            // Use explicit parameter if provided, otherwise default to true for classification
            var shouldDropMissingLabels = dropMissingLabels ?? isClassificationTask;

            if (shouldDropMissingLabels && !string.IsNullOrEmpty(effectiveDefinition.Label))
            {
                var csvHelper = new CsvHelperImpl();
                var labelHandler = new LabelValueHandler(csvHelper, new TrainCommandLogger());

                var labelAnalysis = await labelHandler.AnalyzeLabelColumnAsync(
                    resolvedDataFile,
                    effectiveDefinition.Label);

                if (!string.IsNullOrEmpty(labelAnalysis.Error))
                {
                    AnsiConsole.MarkupLine($"[yellow]Warning:[/] Label analysis error: {labelAnalysis.Error}");
                }
                else if (labelAnalysis.HasMissingValues)
                {
                    AnsiConsole.MarkupLine($"[yellow]Warning:[/] Found {labelAnalysis.MissingCount}/{labelAnalysis.TotalRows} rows ({labelAnalysis.MissingPercentage:F1}%) with missing labels");

                    // Create cleaned data file
                    var cleanedDataPath = Path.Combine(
                        Path.GetDirectoryName(resolvedDataFile)!,
                        $"{Path.GetFileNameWithoutExtension(resolvedDataFile)}_cleaned{Path.GetExtension(resolvedDataFile)}");

                    var cleanResult = await labelHandler.DropMissingLabelsAsync(
                        resolvedDataFile,
                        cleanedDataPath,
                        effectiveDefinition.Label);

                    if (cleanResult.Success)
                    {
                        AnsiConsole.MarkupLine($"[green]>[/] Dropped {cleanResult.DroppedRowCount} rows with missing labels");
                        AnsiConsole.MarkupLine($"[green]>[/] Using cleaned data: [cyan]{Path.GetRelativePath(projectRoot, cleanResult.OutputPath!)}[/]");
                        resolvedDataFile = cleanResult.OutputPath!;
                        allDataFilesUsed.Add(resolvedDataFile);
                    }
                    else
                    {
                        AnsiConsole.MarkupLine($"[red]Error:[/] Failed to clean label data: {cleanResult.Error}");
                        return 1;
                    }
                }
                else
                {
                    AnsiConsole.MarkupLine($"[green]✓[/] Label column '{effectiveDefinition.Label}' has no missing values ({labelAnalysis.ValidCount} rows, {labelAnalysis.UniqueValueCount} unique values)");
                }
            }

            // T4.5: Class distribution analysis for classification tasks
            if (isClassificationTask && !string.IsNullOrEmpty(effectiveDefinition.Label))
            {
                var csvHelperForDistribution = new CsvHelperImpl();
                var distributionAnalyzer = new ClassDistributionAnalyzer(csvHelperForDistribution);

                try
                {
                    var distributionResult = await distributionAnalyzer.AnalyzeAsync(
                        resolvedDataFile,
                        effectiveDefinition.Label);

                    TrainPresenter.DisplayClassDistribution(distributionResult);
                }
                catch (Exception ex)
                {
                    AnsiConsole.MarkupLine($"[yellow]Warning:[/] Class distribution analysis failed: {ex.Message}");
                }
            }

            // Data quality analysis if requested
            if (analyzeData || !string.IsNullOrEmpty(generateScript))
            {
                var analyzer = new DataQualityAnalyzer(new TrainCommandLogger());
                var issues = await analyzer.AnalyzeAsync(resolvedDataFile, effectiveDefinition.Label);

                TrainPresenter.DisplayDataQualityIssues(issues);

                // Generate preprocessing script if requested
                if (!string.IsNullOrEmpty(generateScript))
                {
                    var scriptGenerator = new PreprocessingScriptGenerator(new TrainCommandLogger());
                    var generated = await scriptGenerator.AnalyzeAndGenerateAsync(
                        resolvedDataFile,
                        generateScript,
                        effectiveDefinition.Label);

                    if (generated)
                    {
                        AnsiConsole.MarkupLine($"[green]✓[/] Generated preprocessing script: [cyan]{generateScript}[/]");
                        AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
                        AnsiConsole.MarkupLine($"  1. Review the generated script: {generateScript}");
                        AnsiConsole.MarkupLine($"  2. Move it to .mloop/scripts/preprocess/ directory");
                        AnsiConsole.MarkupLine($"  3. Run 'mloop train' again to apply preprocessing automatically");
                        AnsiConsole.WriteLine();
                    }
                }

                // If only analyzing (not training), exit here
                if (analyzeData)
                {
                    return 0;
                }
            }

            // Execute YAML-defined preprocessing pipeline (FilePrepper DataPipeline API)
            if (effectiveDefinition.Prep is { Count: > 0 })
            {
                AnsiConsole.WriteLine();
                AnsiConsole.Write(new Rule("[blue]Preprocessing Pipeline[/]").LeftJustified());
                AnsiConsole.WriteLine();

                try
                {
                    var pipelineExecutor = new DataPipelineExecutor(new TrainCommandLogger());
                    var prepOutputPath = Path.Combine(
                        Path.GetDirectoryName(resolvedDataFile)!,
                        $"{Path.GetFileNameWithoutExtension(resolvedDataFile)}_prep{Path.GetExtension(resolvedDataFile)}");

                    resolvedDataFile = await pipelineExecutor.ExecuteAsync(
                        resolvedDataFile,
                        effectiveDefinition.Prep,
                        prepOutputPath);

                    allDataFilesUsed.Add(resolvedDataFile);
                    AnsiConsole.MarkupLine($"[green]>[/] Preprocessed data: [cyan]{Path.GetRelativePath(projectRoot, resolvedDataFile)}[/]");
                    AnsiConsole.WriteLine();
                }
                catch (Exception ex)
                {
                    AnsiConsole.MarkupLine("[red]Preprocessing pipeline failed:[/]");
                    AnsiConsole.WriteLine(ex.Message);
                    return 1;
                }
            }

            // Execute preprocessing scripts if available (Roslyn .cs scripts)
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

            // Apply class balancing if requested (only for classification tasks)
            string? balancedFilePath = null;
            if (!string.IsNullOrEmpty(balance) && isClassificationTask && !string.IsNullOrEmpty(effectiveDefinition.Label))
            {
                var dataBalancer = new DataBalancer();
                var balanceResult = dataBalancer.Balance(resolvedDataFile, effectiveDefinition.Label, balance);

                if (balanceResult.Applied && balanceResult.BalancedFilePath != null)
                {
                    AnsiConsole.WriteLine();
                    AnsiConsole.Write(new Rule("[blue]Class Balancing[/]").LeftJustified());
                    AnsiConsole.WriteLine();
                    AnsiConsole.MarkupLine($"[green]✓[/] {balanceResult.Message}");
                    AnsiConsole.MarkupLine($"[green]>[/] Using balanced data: [cyan]{Path.GetRelativePath(projectRoot, balanceResult.BalancedFilePath)}[/]");

                    // IMP-5: Warn about overfitting risk from high replication
                    var replicationRatio = (double)balanceResult.NewMinorityCount / balanceResult.OriginalMinorityCount;
                    if (replicationRatio > 10)
                    {
                        AnsiConsole.MarkupLine($"[yellow]Warning:[/] Minority class replicated {replicationRatio:F0}x — high risk of overfitting with duplicated samples.");
                        AnsiConsole.MarkupLine("[grey]  Consider using independent test data for evaluation.[/]");
                    }
                    AnsiConsole.WriteLine();

                    resolvedDataFile = balanceResult.BalancedFilePath;
                    allDataFilesUsed.Add(resolvedDataFile);
                    balancedFilePath = balanceResult.BalancedFilePath;
                }
                else if (!string.IsNullOrEmpty(balanceResult.Message))
                {
                    AnsiConsole.MarkupLine($"[grey]Balance:[/] {balanceResult.Message}");
                }
            }
            else if (!string.IsNullOrEmpty(balance) && !isClassificationTask)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] --balance option is only applicable to classification tasks");
            }

            // Display data summary
            TrainPresenter.DisplayDataSummary(resolvedDataFile, effectiveDefinition.Label);

            // Display training configuration
            TrainPresenter.DisplayTrainingConfig(resolvedDataFile, resolvedModelName, effectiveDefinition);

            // Validate label column exists
            await TrainDataValidator.ValidateLabelColumnAsync(resolvedDataFile, effectiveDefinition.Label, resolvedModelName);

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
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLine("[yellow]Suggestions:[/]");
                AnsiConsole.MarkupLine("  [blue]>[/] Try increasing the time limit: [cyan]--time 120[/]");
                AnsiConsole.MarkupLine("  [blue]>[/] Check data quality: [cyan]mloop analyze[/]");
                AnsiConsole.MarkupLine("  [blue]>[/] Verify label column exists and has valid values");
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

            // Sync mloop.yaml if CLI overrode label or task
            await SyncYamlConfigAsync(configLoader, userConfig, resolvedModelName, effectiveDefinition, label, task);

            // Display results
            TrainPresenter.DisplayResults(result, resolvedModelName);

            // T4.4: Performance diagnostics
            var performanceDiagnostics = new PerformanceDiagnostics();
            var diagnosticResult = performanceDiagnostics.Analyze(
                trainingConfig.Task,
                result.Metrics);

            TrainPresenter.DisplayDiagnostics(diagnosticResult);

            // Auto-promote to production if enabled
            if (!noPromote)
            {
                AnsiConsole.WriteLine();
                var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);
                var primaryMetric = trainingConfig.Metric;

                // Get current production model for comparison
                var production = await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None);

                // Show comparison if production model exists
                if (production?.Metrics != null && result.Metrics != null)
                {
                    TrainPresenter.DisplayProductionComparison(result, production);
                }

                var promoted = await modelRegistry.AutoPromoteAsync(
                    resolvedModelName,
                    result.ExperimentId,
                    primaryMetric,
                    CancellationToken.None);

                // Resolve class count for quality gate threshold
                int? classCount = null;
                if (!promoted)
                {
                    try
                    {
                        var expData = await experimentStore.LoadAsync(resolvedModelName, result.ExperimentId, CancellationToken.None);
                        var labelSchema = expData.Config?.InputSchema?.Columns?
                            .FirstOrDefault(s => s.Name.Equals(trainingConfig.LabelColumn, StringComparison.OrdinalIgnoreCase));
                        if (labelSchema?.UniqueValueCount > 0)
                            classCount = labelSchema.UniqueValueCount;
                    }
                    catch { /* schema unavailable, use default threshold */ }
                }

                var minThreshold = ModelRegistry.GetMinimumMetricThreshold(primaryMetric, classCount);
                TrainPresenter.DisplayPromotionResult(promoted, primaryMetric, result, production, minThreshold);
            }

            // T4.6: Unused data warning
            var dataDirectory = Path.GetDirectoryName(resolvedDataFile);
            if (!string.IsNullOrEmpty(dataDirectory) && Directory.Exists(dataDirectory))
            {
                var unusedDataScanner = new UnusedDataScanner();
                var scanResult = unusedDataScanner.Scan(dataDirectory, allDataFilesUsed);
                TrainPresenter.DisplayUnusedDataWarning(scanResult);
            }

            return 0;
        }
        catch (Exception ex)
        {
            // T8.3: Enhanced error messaging with actionable suggestions
            ErrorSuggestions.DisplayError(ex, "training");
            return 1;
        }
    }

    /// <summary>
    /// Updates mloop.yaml when CLI-specified label or task differs from the stored config.
    /// This ensures predict can find the correct label after training with --label override.
    /// </summary>
    internal static async Task SyncYamlConfigAsync(
        ConfigLoader configLoader,
        MLoopConfig userConfig,
        string modelName,
        ModelDefinition effectiveDefinition,
        string? cliLabel,
        string? cliTask)
    {
        // Only sync if CLI actually overrode something
        if (string.IsNullOrEmpty(cliLabel) && string.IsNullOrEmpty(cliTask))
            return;

        var yamlModel = userConfig.Models.TryGetValue(modelName, out var m) ? m : null;

        bool needsUpdate = false;
        if (!string.IsNullOrEmpty(cliLabel) && (yamlModel == null || !string.Equals(yamlModel.Label, cliLabel, StringComparison.Ordinal)))
            needsUpdate = true;
        if (!string.IsNullOrEmpty(cliTask) && (yamlModel == null || !string.Equals(yamlModel.Task, cliTask, StringComparison.OrdinalIgnoreCase)))
            needsUpdate = true;

        if (!needsUpdate)
            return;

        if (yamlModel != null)
        {
            // Update existing model definition
            if (!string.IsNullOrEmpty(cliLabel))
                yamlModel.Label = cliLabel;
            if (!string.IsNullOrEmpty(cliTask))
                yamlModel.Task = cliTask;
        }
        else
        {
            // Create new model definition in yaml
            userConfig.Models[modelName] = new ModelDefinition
            {
                Task = effectiveDefinition.Task,
                Label = effectiveDefinition.Label,
                Training = effectiveDefinition.Training
            };
        }

        try
        {
            await configLoader.SaveUserConfigAsync(userConfig);
            AnsiConsole.MarkupLine($"[grey]Updated mloop.yaml: model '{modelName}' → label={effectiveDefinition.Label}, task={effectiveDefinition.Task}[/]");
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[yellow]Warning:[/] Could not update mloop.yaml: {ex.Message}");
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
