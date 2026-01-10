using System.CommandLine;
using Microsoft.ML;
using MLoop.CLI.Infrastructure.Configuration;
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
            return ExecuteAsync(dataFile, name, label, task, time, metric, testSplit, noPromote, analyzeData, generateScript, autoMerge, dropMissingLabels, dataPaths);
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
        string[]? dataPaths)
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
                            resolvedDataFile = await ResolveDataFileAsync(dataFile, userConfig, projectRoot, datasetDiscovery, fileSystem);
                        }
                    }
                    else
                    {
                        AnsiConsole.MarkupLine("[grey]No mergeable CSV groups found, using standard discovery[/]");
                        resolvedDataFile = await ResolveDataFileAsync(dataFile, userConfig, projectRoot, datasetDiscovery, fileSystem);
                    }
                }
                else
                {
                    resolvedDataFile = await ResolveDataFileAsync(dataFile, userConfig, projectRoot, datasetDiscovery, fileSystem);
                }
            }
            else
            {
                resolvedDataFile = await ResolveDataFileAsync(dataFile, userConfig, projectRoot, datasetDiscovery, fileSystem);
            }

            if (resolvedDataFile == null)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] No data file specified and datasets/train.csv not found.");
                AnsiConsole.MarkupLine("[yellow]Tip:[/] Create datasets/train.csv or specify a file: mloop train <data-file>");
                return 1;
            }

            AnsiConsole.MarkupLine($"[green]>[/] Using data: [cyan]{Path.GetRelativePath(projectRoot, resolvedDataFile)}[/]");

            // Handle missing label values (T4.2)
            // Default behavior: drop missing labels for classification tasks
            var isClassificationTask = effectiveDefinition.Task?.ToLowerInvariant() switch
            {
                "binaryclassification" => true,
                "multiclassclassification" => true,
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

                    if (distributionResult.Error == null)
                    {
                        AnsiConsole.WriteLine();
                        AnsiConsole.Write(new Rule("[blue]Class Distribution[/]").LeftJustified());
                        AnsiConsole.WriteLine();
                        AnsiConsole.WriteLine(distributionResult.DistributionVisualization);
                        AnsiConsole.WriteLine();

                        if (distributionResult.NeedsAttention)
                        {
                            AnsiConsole.MarkupLine($"[yellow]⚠ {distributionResult.Summary}[/]");

                            foreach (var warning in distributionResult.Warnings)
                            {
                                AnsiConsole.MarkupLine($"[yellow]  • {warning}[/]");
                            }

                            if (distributionResult.Suggestions.Count > 0)
                            {
                                AnsiConsole.MarkupLine("[grey]Suggestions:[/]");
                                foreach (var suggestion in distributionResult.Suggestions)
                                {
                                    AnsiConsole.MarkupLine($"[grey]  • {suggestion}[/]");
                                }
                            }
                            AnsiConsole.WriteLine();
                        }
                        else
                        {
                            AnsiConsole.MarkupLine($"[green]✓[/] {distributionResult.Summary}");
                            AnsiConsole.WriteLine();
                        }
                    }
                }
                catch (Exception ex)
                {
                    AnsiConsole.MarkupLine($"[yellow]Warning:[/] Class distribution analysis failed: {ex.Message}");
                }
            }

            // Data quality analysis if requested
            if (analyzeData || !string.IsNullOrEmpty(generateScript))
            {
                AnsiConsole.WriteLine();
                AnsiConsole.Write(new Rule("[blue]Data Quality Analysis[/]").LeftJustified());
                AnsiConsole.WriteLine();

                var analyzer = new DataQualityAnalyzer(new TrainCommandLogger());
                var issues = await analyzer.AnalyzeAsync(resolvedDataFile, effectiveDefinition.Label);

                if (issues.Count == 0)
                {
                    AnsiConsole.MarkupLine("[green]✓[/] No data quality issues detected!");
                    AnsiConsole.WriteLine();
                }
                else
                {
                    // Display issues grouped by severity
                    var criticalIssues = issues.Where(i => i.Severity == IssueSeverity.Critical).ToList();
                    var highIssues = issues.Where(i => i.Severity == IssueSeverity.High).ToList();
                    var mediumIssues = issues.Where(i => i.Severity == IssueSeverity.Medium).ToList();
                    var lowIssues = issues.Where(i => i.Severity == IssueSeverity.Low).ToList();

                    if (criticalIssues.Any())
                    {
                        AnsiConsole.MarkupLine("[red]CRITICAL Issues:[/]");
                        foreach (var issue in criticalIssues)
                        {
                            AnsiConsole.MarkupLine($"  [red]•[/] {issue.Description}");
                            if (!string.IsNullOrEmpty(issue.SuggestedFix))
                            {
                                AnsiConsole.MarkupLine($"    [grey]Fix: {issue.SuggestedFix.Replace("\n", "\n    ")}[/]");
                            }
                        }
                        AnsiConsole.WriteLine();
                    }

                    if (highIssues.Any())
                    {
                        AnsiConsole.MarkupLine("[yellow]HIGH Priority Issues:[/]");
                        foreach (var issue in highIssues)
                        {
                            AnsiConsole.MarkupLine($"  [yellow]•[/] {issue.Description}");
                            if (!string.IsNullOrEmpty(issue.SuggestedFix))
                            {
                                AnsiConsole.MarkupLine($"    [grey]Fix: {issue.SuggestedFix.Replace("\n", "\n    ")}[/]");
                            }
                        }
                        AnsiConsole.WriteLine();
                    }

                    if (mediumIssues.Any())
                    {
                        AnsiConsole.MarkupLine("[blue]MEDIUM Priority Issues:[/]");
                        foreach (var issue in mediumIssues)
                        {
                            AnsiConsole.MarkupLine($"  [blue]•[/] {issue.Description}");
                        }
                        AnsiConsole.WriteLine();
                    }

                    if (lowIssues.Any())
                    {
                        AnsiConsole.MarkupLine("[grey]LOW Priority Issues: {0}[/]", lowIssues.Count);
                        AnsiConsole.WriteLine();
                    }
                }

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

            // T4.4: Performance diagnostics
            var performanceDiagnostics = new PerformanceDiagnostics();
            var diagnosticResult = performanceDiagnostics.Analyze(
                trainingConfig.Task,
                result.Metrics);

            if (diagnosticResult.NeedsAttention)
            {
                AnsiConsole.WriteLine();
                AnsiConsole.Write(new Rule("[yellow]Performance Diagnostics[/]").LeftJustified());
                AnsiConsole.WriteLine();

                var levelColor = diagnosticResult.OverallAssessment switch
                {
                    PerformanceLevel.Poor => "red",
                    PerformanceLevel.Low => "yellow",
                    _ => "orange1"
                };

                AnsiConsole.MarkupLine($"[{levelColor}]⚠ {diagnosticResult.Summary}[/]");
                AnsiConsole.WriteLine();

                if (diagnosticResult.Warnings.Count > 0)
                {
                    AnsiConsole.MarkupLine("[yellow]Warnings:[/]");
                    foreach (var warning in diagnosticResult.Warnings)
                    {
                        AnsiConsole.MarkupLine($"  [yellow]•[/] {warning}");
                    }
                    AnsiConsole.WriteLine();
                }

                if (diagnosticResult.Suggestions.Count > 0)
                {
                    AnsiConsole.MarkupLine("[blue]Suggestions to improve performance:[/]");
                    foreach (var suggestion in diagnosticResult.Suggestions)
                    {
                        AnsiConsole.MarkupLine($"  [grey]•[/] {suggestion}");
                    }
                    AnsiConsole.WriteLine();
                }
            }

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

            // T4.6: Unused data warning
            var dataDirectory = Path.GetDirectoryName(resolvedDataFile);
            if (!string.IsNullOrEmpty(dataDirectory) && Directory.Exists(dataDirectory))
            {
                var unusedDataScanner = new UnusedDataScanner();
                var usedFiles = new List<string> { resolvedDataFile };

                var scanResult = unusedDataScanner.Scan(dataDirectory, usedFiles);

                if (scanResult.HasUnusedFiles)
                {
                    AnsiConsole.Write(new Rule("[grey]Data Directory Summary[/]").LeftJustified());
                    AnsiConsole.WriteLine();
                    AnsiConsole.MarkupLine($"[grey]{scanResult.Summary}[/]");

                    // Show warnings if any
                    foreach (var warning in scanResult.Warnings.Take(3))
                    {
                        AnsiConsole.MarkupLine($"[yellow]  ⚠ {warning}[/]");
                    }

                    // Show suggestions if any
                    foreach (var suggestion in scanResult.Suggestions.Take(2))
                    {
                        AnsiConsole.MarkupLine($"[grey]  💡 {suggestion}[/]");
                    }

                    // List unused files (limit to 5)
                    if (scanResult.UnusedFiles.Count <= 5)
                    {
                        AnsiConsole.MarkupLine("[grey]  Unused files:[/]");
                        foreach (var file in scanResult.UnusedFiles)
                        {
                            AnsiConsole.MarkupLine($"[grey]    • {file.FileName} ({file.SizeFormatted})[/]");
                        }
                    }
                    else
                    {
                        var remaining = scanResult.UnusedFiles.Count - 3;
                        AnsiConsole.MarkupLine("[grey]  Unused files:[/]");
                        foreach (var file in scanResult.UnusedFiles.Take(3))
                        {
                            AnsiConsole.MarkupLine($"[grey]    • {file.FileName} ({file.SizeFormatted})[/]");
                        }
                        AnsiConsole.MarkupLine($"[grey]    ... and {remaining} more[/]");
                    }
                    AnsiConsole.WriteLine();
                }
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

    /// <summary>
    /// Resolves the data file path from various sources (explicit path, config, or auto-discovery).
    /// </summary>
    private static Task<string?> ResolveDataFileAsync(
        string? dataFile,
        MLoopConfig userConfig,
        string projectRoot,
        IDatasetDiscovery datasetDiscovery,
        IFileSystemManager fileSystem)
    {
        if (!string.IsNullOrEmpty(dataFile))
        {
            var resolvedPath = Path.IsPathRooted(dataFile)
                ? dataFile
                : Path.Combine(projectRoot, dataFile);

            return File.Exists(resolvedPath)
                ? Task.FromResult<string?>(resolvedPath)
                : Task.FromResult<string?>(null);
        }

        // Try config data path
        if (!string.IsNullOrEmpty(userConfig.Data?.Train))
        {
            var configPath = Path.IsPathRooted(userConfig.Data.Train)
                ? userConfig.Data.Train
                : Path.Combine(projectRoot, userConfig.Data.Train);

            if (File.Exists(configPath))
            {
                return Task.FromResult<string?>(configPath);
            }
        }

        // Auto-discover datasets/train.csv
        var datasets = datasetDiscovery.FindDatasets(projectRoot);
        return Task.FromResult(datasets?.TrainPath);
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
