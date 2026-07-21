using System.CommandLine;
using Microsoft.ML;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.AutoML;
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
            Description = "ML task type (binary-classification, multiclass-classification, regression, anomaly-detection, clustering, ranking, forecasting, time-series-anomaly, recommendation, image-classification, object-detection, text-classification, sentence-similarity, ner, question-answering)"
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

        var noAutoTimeOption = new Option<bool>("--no-auto-time")
        {
            Description = "Disable automatic time estimation (use default 300s)",
            DefaultValueFactory = _ => false
        };

        var autoTimeOption = new Option<bool>("--auto-time")
        {
            Description = "Force automatic time estimation based on data size, overriding any time_limit_seconds set in mloop.yaml",
            DefaultValueFactory = _ => false
        };

        var balanceOption = new Option<string?>("--balance", "-b")
        {
            Description = "Class balancing strategy: 'auto' (balance to 10:1 if ratio > 10:1), 'none' (no balancing), or target ratio (e.g., '5' for 5:1). Use without value for 'auto'.",
            Arity = ArgumentArity.ZeroOrOne
        };
        var groupColumnOption = new Option<string?>("--group-column")
        {
            Description = "Group column for ranking task (groups rows into query contexts)"
        };

        var maxRowsOption = new Option<int?>("--max-rows")
        {
            Description = "Maximum training rows. Data exceeding this is auto-sampled (random for regression/anomaly, stratified for classification)"
        };

        var samplingStrategyOption = new Option<string?>("--sampling-strategy")
        {
            Description = "Sampling strategy when --max-rows is used: 'random' or 'stratified' (default: task-aware)"
        };

        var samplingStrategyAliasOption = new Option<string?>("--sampling")
        {
            Description = "Alias for --sampling-strategy"
        };
        samplingStrategyAliasOption.Hidden = true;

        var seedOption = new Option<int?>("--seed")
        {
            Description = "Random seed for sampling reproducibility (default: 42)"
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Emit newline-delimited JSON events (phase/trial/warning/result/error) on stdout instead of the rich display",
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
        command.Options.Add(analyzeDataOption);
        command.Options.Add(generateScriptOption);
        command.Options.Add(autoMergeOption);
        command.Options.Add(dropMissingLabelsOption);
        command.Options.Add(dataOption);
        command.Options.Add(noAutoTimeOption);
        command.Options.Add(autoTimeOption);
        command.Options.Add(balanceOption);
        command.Options.Add(groupColumnOption);
        command.Options.Add(maxRowsOption);
        command.Options.Add(samplingStrategyOption);
        command.Options.Add(samplingStrategyAliasOption);
        command.Options.Add(seedOption);
        command.Options.Add(jsonOption);

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
            var noAutoTime = parseResult.GetValue(noAutoTimeOption);
            var autoTime = parseResult.GetValue(autoTimeOption);
            var balance = parseResult.GetValue(balanceOption);
            var groupColumn = parseResult.GetValue(groupColumnOption);
            var maxRows = parseResult.GetValue(maxRowsOption);
            var samplingStrategy = parseResult.GetValue(samplingStrategyOption) ?? parseResult.GetValue(samplingStrategyAliasOption);
            var seed = parseResult.GetValue(seedOption);
            var json = parseResult.GetValue(jsonOption);
            // Handle --balance without argument: default to "auto"
            if (balance == null && parseResult.Tokens.Any(t => t.Value == "--balance" || t.Value == "-b"))
            {
                balance = "auto";
            }
            return ExecuteAsync(dataFile, name, label, task, time, metric, testSplit, noPromote, analyzeData, generateScript, autoMerge, dropMissingLabels, dataPaths, noAutoTime, autoTime, balance, groupColumn, maxRows, samplingStrategy, seed, json);
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
        bool noAutoTime,
        bool autoTime,
        string? balance,
        string? groupColumn,
        int? maxRows = null,
        string? samplingStrategy = null,
        int? seed = null,
        bool json = false)
    {
        // --json reserves stdout for the event stream: every renderer and narrator in the training
        // path is silenced for the duration, and the real stdout is handed to the emitter. Disposed
        // in the finally below so the console is restored even on the error paths.
        using var machineOutput = json ? new MachineOutputScope() : null;
        var events = machineOutput is null ? null : new TrainJsonEmitter(machineOutput.Stdout);
        if (machineOutput is not null && events is not null)
        {
            // Every failure path in this command already ends at the stderr diagnostics sink — one
            // top-level catch plus a dozen validation early-returns. Hooking the sink emits the error
            // event from all of them instead of tagging each return site.
            machineOutput.ErrorSink = events.Error;
            machineOutput.WarningSink = events.Warning;
        }

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
                ErrorConsole.Error("Not inside a MLoop project.");
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
                ErrorConsole.Error($"{ex.Message}");
                return 1;
            }

            // Apply CLI --group-column override
            if (!string.IsNullOrEmpty(groupColumn))
                effectiveDefinition.GroupColumn = groupColumn;

            // Validate required fields (defensive guard for nullable flow analysis).
            // AutoMLRunner.RequiresLabel is the single source — previously this inline set omitted
            // time-series-anomaly, so a label-less ts-anomaly project that merge/init accept failed here.
            var requiresLabel = AutoMLRunner.RequiresLabel(effectiveDefinition.Task);

            if (string.IsNullOrEmpty(effectiveDefinition.Task) ||
                (requiresLabel && string.IsNullOrEmpty(effectiveDefinition.Label)))
            {
                ErrorConsole.Error("Task is required. Label is required for supervised tasks. Use --task and <label> arguments, or define them in mloop.yaml");
                return 1;
            }

            // Ranking requires group column
            if (effectiveDefinition.Task.Equals("ranking", StringComparison.OrdinalIgnoreCase) &&
                string.IsNullOrEmpty(effectiveDefinition.GroupColumn))
            {
                ErrorConsole.Error("Ranking task requires a group column. Use --group-column or set 'group_column' in mloop.yaml");
                return 1;
            }

            // Recommendation requires user_column and item_column
            if (effectiveDefinition.Task.Equals("recommendation", StringComparison.OrdinalIgnoreCase))
            {
                if (string.IsNullOrEmpty(effectiveDefinition.UserColumn) || string.IsNullOrEmpty(effectiveDefinition.ItemColumn))
                {
                    ErrorConsole.Error("Recommendation task requires user_column and item_column. Set them in mloop.yaml");
                    return 1;
                }
            }

            // Forecasting requires horizon
            if (effectiveDefinition.Task.Equals("forecasting", StringComparison.OrdinalIgnoreCase) &&
                (effectiveDefinition.Horizon ?? 0) <= 0)
            {
                ErrorConsole.Error("Forecasting task requires horizon > 0. Set 'horizon' in mloop.yaml");
                return 1;
            }

            // Resolve data file path
            string? resolvedDataFile;
            var datasetDiscovery = new DatasetDiscovery(fileSystem);
            var isDirectoryBased = DataLoaderFactory.IsDirectoryBased(effectiveDefinition.Task);

            // Validate --auto-time opt-in against contradictory inputs before doing any work.
            if (autoTime)
            {
                if (noAutoTime)
                {
                    ErrorConsole.Error("--auto-time and --no-auto-time cannot be used together.");
                    return 1;
                }
                if (time.HasValue)
                {
                    ErrorConsole.Error("--auto-time cannot be combined with an explicit --time value.");
                    return 1;
                }
                if (isDirectoryBased)
                {
                    ErrorConsole.Error("--auto-time is not supported for image/directory training (time estimation probes CSV rows).");
                    return 1;
                }
            }

            if (isDirectoryBased)
            {
                // Directory-based tasks consume a directory, not a CSV file: image
                // classification reads class subfolders (folder name = label); object
                // detection reads a COCO annotations file plus the referenced images.
                resolvedDataFile = ResolveDirectoryDataset(effectiveDefinition.Task, dataFile, dataPaths, projectRoot);
                if (resolvedDataFile == null)
                {
                    if (effectiveDefinition.Task.Equals("object-detection", StringComparison.OrdinalIgnoreCase))
                    {
                        ErrorConsole.Error("Object-detection dataset not found.");
                        ErrorConsole.Tip("Put a COCO annotations.json plus images under datasets/coco/, or pass a directory: mloop train --task object-detection <dir>");
                    }
                    else
                    {
                        ErrorConsole.Error("Image dataset directory not found.");
                        ErrorConsole.Tip("Lay images out as datasets/images/<class>/<files>, or pass a directory: mloop train --task image-classification <dir>");
                    }
                    return 1;
                }
            }
            // T4.3: Handle --data option for external data paths
            else if (dataPaths is { Length: > 0 })
            {
                var csvHelper = new CsvHelperImpl();

                // Resolve all paths (support relative and absolute)
                var resolvedPaths = new List<string>();
                foreach (var path in dataPaths)
                {
                    var resolved = Path.IsPathRooted(path) ? path : Path.GetFullPath(Path.Combine(projectRoot, path));
                    if (!File.Exists(resolved))
                    {
                        ErrorConsole.Error($"Data file not found: {path}");
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
                        ErrorConsole.Error($"Schema mismatch between files: {validation.Message}");
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
                        ErrorConsole.Error($"Failed to merge files: {mergeResult.Error}");
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
                ErrorConsole.Error("No data file specified and datasets/train.csv not found.");
                ErrorConsole.Tip("Create datasets/train.csv or specify a file: mloop train <data-file>");
                return 1;
            }

            AnsiConsole.MarkupLine($"[green]>[/] Using data: [cyan]{Path.GetRelativePath(projectRoot, resolvedDataFile)}[/]");

            // testDataFile may be produced by stratified split below (CSV path only),
            // and is read later when building the training config — declare it here so it
            // survives the directory-based bypass. allDataFilesUsed is consumed by the
            // unused-data scan after training and is likewise declared outside the bypass.
            string? testDataFile = null;
            var allDataFilesUsed = new List<string> { resolvedDataFile };
            IEstimator<ITransformer>? preFeaturizer = null;
            List<string> preFeaturizerColumns = new();

            // CSV preprocessing (flatten, sampling, cleaning, class analysis, preprocessing,
            // balancing, splitting) applies only to tabular tasks. Image classification loads
            // a directory and skips this entire block.
            if (!isDirectoryBased)
            {
            // Flatten multi-line quoted fields early so all downstream line-by-line processing is safe
            resolvedDataFile = CsvDataLoader.FlattenMultiLineQuotedFields(resolvedDataFile);

            // Auto-sampling for large datasets (--max-rows)
            if (maxRows.HasValue && maxRows.Value > 0)
            {
                resolvedDataFile = await TrainDataValidator.ApplySamplingIfNeededAsync(
                    resolvedDataFile,
                    maxRows.Value,
                    effectiveDefinition.Task,
                    effectiveDefinition.Label,
                    samplingStrategy,
                    seed ?? 42,
                    projectRoot);
            }

            // (allDataFilesUsed is declared above so it survives the directory-based bypass.)

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
                    WarningConsole.Warn($"Found {labelAnalysis.MissingCount}/{labelAnalysis.TotalRows} rows ({labelAnalysis.MissingPercentage:F1}%) with missing labels");

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
                        ErrorConsole.Error($"Failed to clean label data: {cleanResult.Error}");
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

                    // Single-class early termination: cannot train a classifier with one class
                    if (distributionResult.ClassCount <= 1)
                    {
                        ErrorConsole.Error("Cannot train a classifier with only one class.");
                        ErrorConsole.Tip("Check if the correct label column is specified, or if the data contains only one category.");
                        return 1;
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

            // Execute YAML-defined preprocessing pipeline (누수 안전: 통계 변환은 preFeaturizer로 이전)
            if (effectiveDefinition.Prep is { Count: > 0 })
            {
                AnsiConsole.WriteLine();
                AnsiConsole.Write(new Rule("[blue]Preprocessing Pipeline[/]").LeftJustified());
                AnsiConsole.WriteLine();
                try
                {
                    var mlCtxForPrep = new MLContext(seed: 42);
                    List<string> prepWarnings;
                    (resolvedDataFile, preFeaturizer, prepWarnings, preFeaturizerColumns) =
                        await ApplyPrepAsync(resolvedDataFile, effectiveDefinition.Prep, mlCtxForPrep, effectiveDefinition.Task);

                    allDataFilesUsed.Add(resolvedDataFile);
                    foreach (var w in prepWarnings)
                        AnsiConsole.MarkupLine($"[yellow]![/] {w}");
                    AnsiConsole.MarkupLine($"[green]>[/] Preprocessed data: [cyan]{Path.GetRelativePath(projectRoot, resolvedDataFile)}[/]");
                    if (preFeaturizer != null)
                        AnsiConsole.MarkupLine("[green]>[/] 통계 변환은 학습 중 fold-내 fit으로 적용됩니다(누수 안전).");
                    AnsiConsole.WriteLine();
                }
                catch (Exception ex)
                {
                    ErrorConsole.Out.MarkupLine("[red]Preprocessing pipeline failed:[/]");
                    ErrorConsole.Out.WriteLine(ex.Message);
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
                    ErrorConsole.Out.MarkupLine("[red]Preprocessing failed:[/]");
                    ErrorConsole.Out.WriteLine(ex.Message);
                    return 1;
                }
            }

            // Apply class balancing if requested (only for classification tasks)
            // HD-02 fix: split BEFORE balancing to prevent data leakage
            string? balancedFilePath = null;
            if (!string.IsNullOrEmpty(balance) && isClassificationTask && !string.IsNullOrEmpty(effectiveDefinition.Label))
            {
                var effectiveTestSplit = effectiveDefinition.Training?.TestSplit ?? ConfigDefaults.DefaultTestSplit;

                if (effectiveTestSplit > 0)
                {
                    // Step 1: Stratified split BEFORE balancing
                    var splitter = new CsvSplitter();
                    var splitResult = splitter.StratifiedSplit(resolvedDataFile, effectiveDefinition.Label, effectiveTestSplit);

                    AnsiConsole.WriteLine();
                    AnsiConsole.Write(new Rule("[blue]Pre-Balance Split[/]").LeftJustified());
                    AnsiConsole.WriteLine();
                    AnsiConsole.MarkupLine($"[green]✓[/] Stratified split: train={splitResult.TrainRows}, test={splitResult.TestRows}");
                    AnsiConsole.MarkupLine("[grey]  Test set is isolated from balancing to prevent data leakage.[/]");
                    AnsiConsole.WriteLine();

                    allDataFilesUsed.Add(splitResult.TrainFile);
                    allDataFilesUsed.Add(splitResult.TestFile);
                    testDataFile = splitResult.TestFile;

                    // Step 2: Balance only the train split
                    var dataBalancer = new DataBalancer();
                    var balanceResult = dataBalancer.Balance(splitResult.TrainFile, effectiveDefinition.Label, balance);

                    if (balanceResult.Applied && balanceResult.BalancedFilePath != null)
                    {
                        AnsiConsole.Write(new Rule("[blue]Class Balancing (train only)[/]").LeftJustified());
                        AnsiConsole.WriteLine();
                        AnsiConsole.MarkupLine($"[green]✓[/] {balanceResult.Message}");
                        AnsiConsole.MarkupLine($"[green]>[/] Balanced train: [cyan]{Path.GetRelativePath(projectRoot, balanceResult.BalancedFilePath)}[/]");
                        AnsiConsole.MarkupLine($"[green]>[/] Clean test: [cyan]{Path.GetRelativePath(projectRoot, splitResult.TestFile)}[/]");

                        var replicationRatio = (double)balanceResult.NewMinorityCount / balanceResult.OriginalMinorityCount;
                        if (replicationRatio > 10)
                        {
                            AnsiConsole.MarkupLine($"[yellow]Warning:[/] Minority class replicated {replicationRatio:F0}x in train set.");
                        }
                        AnsiConsole.WriteLine();

                        resolvedDataFile = balanceResult.BalancedFilePath;
                        allDataFilesUsed.Add(resolvedDataFile);
                        balancedFilePath = balanceResult.BalancedFilePath;
                    }
                    else
                    {
                        // Balancing not applied (e.g., ratio within threshold) — use train split as-is
                        if (!string.IsNullOrEmpty(balanceResult.Message))
                        {
                            AnsiConsole.MarkupLine($"[grey]Balance:[/] {balanceResult.Message}");
                        }
                        resolvedDataFile = splitResult.TrainFile;
                    }
                }
                else
                {
                    // testSplit == 0: no split needed, balance entire dataset
                    var dataBalancer = new DataBalancer();
                    var balanceResult = dataBalancer.Balance(resolvedDataFile, effectiveDefinition.Label, balance);

                    if (balanceResult.Applied && balanceResult.BalancedFilePath != null)
                    {
                        AnsiConsole.WriteLine();
                        AnsiConsole.Write(new Rule("[blue]Class Balancing[/]").LeftJustified());
                        AnsiConsole.WriteLine();
                        AnsiConsole.MarkupLine($"[green]✓[/] {balanceResult.Message}");
                        AnsiConsole.MarkupLine($"[green]>[/] Using balanced data: [cyan]{Path.GetRelativePath(projectRoot, balanceResult.BalancedFilePath)}[/]");
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
            }
            else if (!string.IsNullOrEmpty(balance) && !isClassificationTask)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] --balance option is only applicable to classification tasks");
            }
            } // end if (!isDirectoryBased) — CSV preprocessing block

            // Display data summary (CSV row/column counts — not applicable to an image directory)
            if (!isDirectoryBased)
                TrainPresenter.DisplayDataSummary(resolvedDataFile, effectiveDefinition.Label);

            // Determine auto-time eligibility:
            // - If --auto-time was explicitly specified, force auto-time even when mloop.yaml
            //   sets time_limit_seconds (the project-workflow opt-in; contradictions already rejected above).
            // - If --time was explicitly specified, use that value (no auto-time)
            // - If --no-auto-time was specified, use default 300s (no auto-time)
            // - If neither --time nor yaml time_limit_seconds is set, enable auto-time
            var yamlHasTimeLimit = effectiveDefinition.Training?.TimeLimitSeconds != null;
            // Auto-time probing samples CSV rows, which does not apply to image directories.
            var useAutoTime = autoTime
                || (!time.HasValue && !yamlHasTimeLimit && !noAutoTime && !isDirectoryBased);

            // Display training configuration (auto-time shows an "auto" limit, not the yaml/default seconds,
            // so the summary matches what the engine actually does — honest record vs. actual)
            TrainPresenter.DisplayTrainingConfig(resolvedDataFile, resolvedModelName, effectiveDefinition, testDataFile, useAutoTime);

            // Validate label column exists (skip for unsupervised and directory-based tasks)
            if (requiresLabel && !isDirectoryBased)
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
                TestSplit = testDataFile != null ? 0 : (effectiveDefinition.Training?.TestSplit ?? ConfigDefaults.DefaultTestSplit),
                TestDataFile = testDataFile != null ? Path.GetFullPath(testDataFile) : null,
                UseAutoTime = useAutoTime,
                ColumnOverrides = effectiveDefinition.Columns?.ToDictionary(
                    kvp => kvp.Key,
                    kvp => kvp.Value.Type),
                NumClusters = effectiveDefinition.NumClusters ?? 0,
                GroupColumn = effectiveDefinition.GroupColumn,
                Horizon = effectiveDefinition.Horizon ?? 0,
                WindowSize = effectiveDefinition.WindowSize ?? 0,
                SeriesLength = effectiveDefinition.SeriesLength ?? 0,
                UserColumn = effectiveDefinition.UserColumn,
                ItemColumn = effectiveDefinition.ItemColumn,
                PreFeaturizer = preFeaturizer,
                PreFeaturizerColumns = preFeaturizerColumns
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
                ErrorConsole.Out.MarkupLine("[red]Training aborted by pre-train hook[/]");
                return 1;
            }

            // If auto-time, display static estimate before progress bar starts
            if (useAutoTime)
            {
                var (rowCount, colCount, hasText, classCount) = TrainingEngine.CollectDataStats(
                    trainingConfig.DataFile, trainingConfig.LabelColumn, trainingConfig.Task);
                var staticEstimate = TimeEstimator.EstimateStatic(rowCount, colCount, trainingConfig.Task, classCount, hasText);
                TrainPresenter.DisplayAutoTimeEstimate(staticEstimate, rowCount, colCount, trainingConfig.Task, classCount);
                AnsiConsole.WriteLine();
            }

            // Train with progress display
            TrainingResult? result = null;
            TrainingProgress? lastAutoTimeEvent = null;

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

                    var progressTracker = new TrainingProgressTracker(trainingConfig.TimeLimitSeconds);

                    var progress = new Progress<TrainingProgress>(p =>
                    {
                        if (p.Phase.HasValue)
                        {
                            events?.Phase(p);
                            // The post-run probe summary keys off the last *probe* phase; the
                            // window boundaries (MainStart/Complete) every run now reports would
                            // otherwise overwrite it and silently drop the summary.
                            if (p.Phase is TrainingPhase.ProbeStart or TrainingPhase.ProbeComplete or TrainingPhase.ProbeConverged)
                                lastAutoTimeEvent = p;
                            progressTracker.EnterPhase(p);
                            progressTask.Description = p.Phase switch
                            {
                                TrainingPhase.ProbeStart => $"[cyan]Phase 1:[/] Probe ({p.ProbeTimeSeconds}s)...",
                                TrainingPhase.ProbeComplete => $"[cyan]Phase 2:[/] Main training ({p.FinalTimeSeconds}s)...",
                                TrainingPhase.ProbeConverged => "[green]Converged[/] in probe phase",
                                TrainingPhase.Complete => $"[green]Finalizing {resolvedModelName}...[/]",
                                _ => progressTask.Description
                            };
                            return;
                        }

                        events?.Trial(p);

                        var trainer = TrainingProgressTracker.ShortTrainerName(p.TrainerName);
                        progressTask.Description = $"[green]Trial {p.TrialNumber}:[/] {trainer} - {p.MetricName}={p.Metric:F4}";

                        if (progressTracker.PercentFor(p) is { } percent)
                            progressTask.Value = percent;
                    });

                    result = await trainingEngine.TrainAsync(trainingConfig, progress, CancellationToken.None);
                    progressTask.Value = 100;
                    progressTask.StopTask();
                });

            // Display auto-time phase summary after progress bar completes
            if (lastAutoTimeEvent?.Phase == TrainingPhase.ProbeComplete)
            {
                TrainPresenter.DisplayProbeResult(
                    lastAutoTimeEvent.ProbeTimeSeconds,
                    lastAutoTimeEvent.Metric,
                    lastAutoTimeEvent.TrialNumber,
                    lastAutoTimeEvent.FinalTimeSeconds);
                AnsiConsole.WriteLine();
            }
            else if (lastAutoTimeEvent?.Phase == TrainingPhase.ProbeConverged)
            {
                TrainPresenter.DisplayProbeConverged(
                    lastAutoTimeEvent.ProbeTimeSeconds,
                    lastAutoTimeEvent.Metric);
                AnsiConsole.WriteLine();
            }

            if (result == null)
            {
                ErrorConsole.Error($"Training failed for model '[cyan]{resolvedModelName}[/]'");
                ErrorConsole.Out.WriteLine();
                ErrorConsole.Out.MarkupLine("[yellow]Suggestions:[/]");
                ErrorConsole.Out.MarkupLine("  [blue]>[/] Try increasing the time limit: [cyan]--time 120[/]");
                ErrorConsole.Out.MarkupLine("  [blue]>[/] Check data quality: [cyan]mloop analyze[/]");
                ErrorConsole.Out.MarkupLine("  [blue]>[/] Verify label column exists and has valid values");
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
                WarningConsole.Warn("Post-train hook aborted, but model was saved successfully");
                // Don't return 1 here - training succeeded, only post-processing failed
            }

            // Sync mloop.yaml if CLI overrode label or task
            await SyncYamlConfigAsync(configLoader, userConfig, resolvedModelName, effectiveDefinition, label, task);

            // Emitted here, before auto-promote: the experiment is finished and recorded at this
            // point. A promotion that fails afterwards adds an error event and a non-zero exit, so a
            // consumer must not read "a result arrived" as "the command succeeded" — the exit code
            // is what says that.
            events?.Result(result, resolvedModelName);

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
                    catch (Exception) { /* schema unavailable, use default threshold */ }
                }

                // Display the threshold the gate actually applied: resolve "auto"/aliases to the
                // canonical key first (matches ShouldPromoteAsync), so image tasks show their 1/N
                // floor instead of a blank (BUG-46).
                var displayMetricKey = result.Metrics != null
                    ? MetricPolicy.ResolveCanonicalMetricKey(primaryMetric, trainingConfig.Task, result.Metrics.Keys)
                    : null;
                var minThreshold = displayMetricKey != null
                    ? MetricPolicy.GetMinimumMetricThreshold(displayMetricKey, classCount)
                    : null;
                // Report the resolved key, not the raw request: the presenter both names the metric
                // in its messages and looks it up in result.Metrics. An unresolved sentinel made
                // every TryGetValue miss, so the staging *reason* silently vanished instead of
                // being shown.
                TrainPresenter.DisplayPromotionResult(promoted, displayMetricKey ?? primaryMetric, result, production, minThreshold);
            }

            // T4.6: Unused data warning - only scan project's datasets/ directory.
            // The scan targets CSV datasets, so skip it for image directory training.
            var datasetsDir = datasetDiscovery.GetDatasetsPath(projectRoot);
            if (!isDirectoryBased && Directory.Exists(datasetsDir))
            {
                var unusedDataScanner = new UnusedDataScanner();
                var scanResult = unusedDataScanner.Scan(datasetsDir, allDataFilesUsed);
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
    /// Resolves the dataset directory for directory-based tasks. Prefers an explicit positional
    /// argument or <c>--data</c> path, then falls back to a task-specific convention:
    /// object detection uses <c>datasets/coco</c>, image classification uses <c>datasets/images</c>,
    /// both finally falling back to <c>datasets</c>. Returns null if no directory exists.
    /// </summary>
    private static string? ResolveDirectoryDataset(string task, string? dataFile, string[]? dataPaths, string projectRoot)
    {
        static string Full(string root, string p) =>
            Path.IsPathRooted(p) ? p : Path.GetFullPath(Path.Combine(root, p));

        if (!string.IsNullOrWhiteSpace(dataFile))
        {
            var p = Full(projectRoot, dataFile);
            return Directory.Exists(p) ? p : null;
        }

        if (dataPaths is { Length: > 0 } && !string.IsNullOrWhiteSpace(dataPaths[0]))
        {
            var p = Full(projectRoot, dataPaths[0]);
            return Directory.Exists(p) ? p : null;
        }

        var conventionDir = task.Equals("object-detection", StringComparison.OrdinalIgnoreCase)
            ? "coco"
            : "images";

        foreach (var candidate in new[]
        {
            Path.Combine(projectRoot, "datasets", conventionDir),
            Path.Combine(projectRoot, "datasets")
        })
        {
            if (Directory.Exists(candidate))
                return candidate;
        }

        return null;
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

    /// <summary>
    /// prep 스텝을 누수 안전하게 라우팅한다. preFeaturizer를 소비하는 태스크
    /// (binary/multiclass/regression)에서만 통계 변환을 preFeaturizer로(fold-내 fit) 보낸다.
    /// 그 외 태스크(clustering/anomaly 등)는 preFeaturizer를 무시하므로 통계 변환을 CSV로 굽고
    /// (적용은 되나 누수) 경고를 남긴다 — 무성 누락 회귀 방지.
    /// 반환: (굽힌 데이터 경로, preFeaturizer, 경고 목록, preFeaturizer 컬럼).
    /// </summary>
    internal static async Task<(string dataFile, IEstimator<ITransformer>? preFeaturizer, List<string> warnings, List<string> preFeaturizerColumns)>
        ApplyPrepAsync(string dataFile, List<PrepStep> prep, MLContext ctx, string task)
    {
        var route = new PrepRouter().Route(ctx, prep, AutoMLRunner.SupportsPreFeaturizer(task));

        var prepOutputPath = Path.Combine(
            Path.GetDirectoryName(dataFile)!,
            $"{Path.GetFileNameWithoutExtension(dataFile)}_prep{Path.GetExtension(dataFile)}");

        // 통계 변환을 제외한 CSV 스텝만 적용(없으면 원본 경로 유지)
        var outFile = dataFile;
        if (route.CsvSteps.Count > 0)
        {
            var executor = new DataPipelineExecutor(new TrainCommandLogger());
            outFile = await executor.ExecuteAsync(dataFile, route.CsvSteps, prepOutputPath);
        }

        return (outFile, route.PreFeaturizer, route.Warnings, route.PreFeaturizerColumns);
    }

    private class TrainCommandLogger : ILogger
    {
        public void Debug(string message) => AnsiConsole.MarkupLine($"[grey]{message}[/]");
        public void Info(string message) => AnsiConsole.WriteLine(message);
        public void Warning(string message) => AnsiConsole.MarkupLine($"[yellow]{message}[/]");
        // Error/Warning go to stderr: these surface while training is already streaming progress to
        // stdout, and a subprocess consumer reading stderr on a non-zero exit must find the cause
        // there. Debug/Info stay on stdout — they are narration, not diagnostics.
        public void Error(string message) => ErrorConsole.Out.MarkupLine($"[red]{message}[/]");
        public void Error(string message, Exception exception)
        {
            ErrorConsole.Out.MarkupLine($"[red]{message}[/]");
            ErrorConsole.Out.WriteException(exception);
        }
    }
}
