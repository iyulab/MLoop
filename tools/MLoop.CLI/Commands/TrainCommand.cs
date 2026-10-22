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
using MLoop.Core.Evaluation;
using MLoop.Core.Hooks;
using MLoop.Core.Models;
using MLoop.Core.Preprocessing;
using MLoop.Core.Storage;
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
            Description = $"ML task type ({TaskTypes.Listed})"
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
            Description = "Drop rows with missing label values (default: true wherever the label is a target to learn; "
                + "a time series keeps its rows)"
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
        var weightOption = new Option<string?>("--weight")
        {
            Description = "Column that weights the rows (relative frequency, e.g. 1/π for sampled labels); never a feature. Binary, multiclass and regression"
        };
        var portableOption = new Option<bool>("--portable")
        {
            Description = "Train only with steps ONNX can express, so 'mloop export onnx' can export the model (text columns use character n-grams). Binary, multiclass and regression"
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
        command.Options.Add(weightOption);
        command.Options.Add(portableOption);
        command.Options.Add(maxRowsOption);
        command.Options.Add(samplingStrategyOption);
        command.Options.Add(samplingStrategyAliasOption);
        command.Options.Add(seedOption);
        command.Options.Add(jsonOption);
        var recordsOption = DataFileOptions.Records();
        command.Options.Add(recordsOption);

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
            return ExecuteAsync(dataFile, name, label, task, time, metric, testSplit, noPromote, analyzeData, generateScript, autoMerge, dropMissingLabels, dataPaths, noAutoTime, autoTime, balance, groupColumn, maxRows, samplingStrategy, seed, json, parseResult.GetValue(recordsOption), parseResult.GetValue(weightOption), parseResult.GetValue(portableOption));
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
        bool json = false,
        string? records = null,
        string? weight = null,
        bool portable = false)
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

        // The task, once resolved, so a failure's advice can fit it.
        string? failedTask = task;

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
                ErrorConsole.Error(
                    ProjectDiscovery.NotInsideProjectCause,
                    ProjectDiscovery.NotInsideProjectGuidance);
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
                failedTask = effectiveDefinition.Task;
            }
            catch (InvalidOperationException ex)
            {
                ErrorConsole.Error($"{ex.Message}");
                return 1;
            }

            // Apply CLI --group-column override
            if (!string.IsNullOrEmpty(groupColumn))
                effectiveDefinition.GroupColumn = groupColumn;

            // Apply CLI --weight override, then refuse a weight the run could not honour — a weight
            // that silently did nothing would train an unweighted model the user believes weighted.
            if (!string.IsNullOrEmpty(weight))
                effectiveDefinition.Weight = weight;
            if ((TrainingInput.BalanceConflict(effectiveDefinition, balance)
                 ?? ModelTrainingConfig.WeightConflict(effectiveDefinition, balance)) is { } weightError)
            {
                ErrorConsole.Error(weightError);
                return 1;
            }

            if (portable)
                effectiveDefinition.Portable = true;
            if (ModelTrainingConfig.PortableConflict(effectiveDefinition) is { } portableError)
            {
                ErrorConsole.Error(portableError);
                return 1;
            }

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

            // `train` had never checked the task against the vocabulary — `TaskTypes` was consulted
            // here only for the --task help text, so a misspelled task reached whichever predicate
            // asked about it first and was answered "no" by all of them, producing a run that
            // trained something nobody had asked for. `init` and `validate` have always checked.
            // The spelling is already canonical by now (ConfigMerger), so a name that fails here is
            // one no spelling of any known task produces.
            if (!TaskTypes.IsValid(effectiveDefinition.Task))
            {
                ErrorConsole.Error(
                    $"Invalid task type '{effectiveDefinition.Task}'.",
                    $"Valid options: {TaskTypes.Listed}");
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

            // Refuse a metric the task's optimizer cannot take before any work starts — the run would
            // otherwise train for minutes on the task default while recording the name given here.
            var metricRejection = MetricNames.Rejection(
                effectiveDefinition.Task, effectiveDefinition.Training?.Metric ?? ConfigDefaults.DefaultMetric);
            if (metricRejection != null)
            {
                ErrorConsole.Error(Markup.Escape(metricRejection), "Set 'metric' in mloop.yaml or pass --metric with one of the names above.");
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
                        ErrorConsole.Error(
                            "Object-detection dataset not found.",
                            "Put a COCO annotations.json plus images under datasets/coco/, or pass a directory: mloop train --task object-detection <dir>");
                    }
                    else
                    {
                        ErrorConsole.Error(
                            "Image dataset directory not found.",
                            "Lay images out as datasets/images/<class>/<files>, or pass a directory: mloop train --task image-classification <dir>");
                    }
                    return 1;
                }
            }
            // T4.3: Handle --data option for external data paths
            else if (dataPaths is { Length: > 0 })
            {
                // Resolve all paths (support relative and absolute)
                var resolvedPaths = new List<string>();
                foreach (var path in dataPaths)
                {
                    var resolved = Path.IsPathRooted(path) ? path : Path.GetFullPath(Path.Combine(projectRoot, path));
                    if (!TabularDataFile.Exists(resolved))
                    {
                        ErrorConsole.Error($"Data file not found: {path}", ErrorConsole.PathNotFoundTip(projectRoot));
                        return 1;
                    }
                    resolvedPaths.Add(resolved);
                }

                if (resolvedPaths.Count == 1)
                {
                    // Single file - use directly
                    resolvedDataFile = resolvedPaths[0];
                    ValueLine.Write("[green]>[/] Using external data: ", dataPaths[0]);
                }
                else
                {
                    // Multiple files - validate and merge
                    AnsiConsole.MarkupLine($"[blue]Merging {resolvedPaths.Count} external data files...[/]");

                    foreach (var file in resolvedPaths)
                    {
                        ValueLine.Write("    [grey]• [/]", Path.GetFileName(file));
                    }

                    // Merge to temp file in datasets directory
                    var datasetsPath = datasetDiscovery.GetDatasetsPath(projectRoot);
                    if (!Directory.Exists(datasetsPath))
                    {
                        Directory.CreateDirectory(datasetsPath);
                    }

                    var merge = await TrainingDataFiles.MergeAsync(
                        resolvedPaths, effectiveDefinition.Weight, Path.Combine(datasetsPath, "merged_train.csv"), records);
                    if (merge.Path is null)
                    {
                        ErrorConsole.Error(Markup.Escape(merge.Error!), Markup.Escape(merge.Guidance!));
                        return 1;
                    }

                    AnsiConsole.MarkupLine($"[green]✓[/] Merged [cyan]{merge.TotalRows}[/] rows from {resolvedPaths.Count} files");
                    foreach (var (fileName, rowCount) in merge.RowsPerFile)
                    {
                        AnsiConsole.MarkupLine($"    [grey]• {Markup.Escape(fileName)}: {rowCount} rows[/]");
                    }
                    AnsiConsole.WriteLine();

                    resolvedDataFile = merge.Path;
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
                            ValueLine.Write("    [grey]• [/]", Path.GetFileName(file));
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
                ErrorConsole.Error(
                    "No data file specified and datasets/train.csv not found.",
                    "Create datasets/train.csv or specify a file: mloop train <data-file>");
                return 1;
            }

            ValueLine.Write("[green]>[/] Using data: ", Path.GetRelativePath(projectRoot, resolvedDataFile));

            // Every step below may swap resolvedDataFile for a converted, flattened or split copy;
            // what the user is told they trained on is still the file they named.
            var sourceDataFile = resolvedDataFile;

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
                // A data file in another format (Parquet, Excel, JSON, TSV) is read as a table from here on.
                resolvedDataFile = await TabularDataFile.AsCsvAsync(resolvedDataFile, records);

                // Flatten multi-line quoted fields early so all downstream line-by-line processing is safe
                resolvedDataFile = CsvDataLoader.FlattenMultiLineQuotedFields(resolvedDataFile, CoreNarration.Sink);

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

                // Execute YAML-defined preprocessing pipeline (누수 안전: 통계 변환은 preFeaturizer로 이전).
                // It runs before the label checks below: a step can drop, add or reshape rows (filter-rows,
                // sample, explode), so missing labels and classes are counted on the rows training gets.
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
                        ValueLine.Write("[green]>[/] Preprocessed data: ", Path.GetRelativePath(projectRoot, resolvedDataFile));
                        if (preFeaturizer != null)
                            AnsiConsole.MarkupLine("[green]>[/] Statistical steps are fitted inside training on the training rows (no leak).");
                        AnsiConsole.WriteLine();
                    }
                    catch (Exception ex)
                    {
                        ErrorConsole.Out.MarkupLine("[red]Preprocessing pipeline failed:[/]");
                        ErrorConsole.Out.WriteLine(ex.Message);
                        return 1;
                    }
                }

                // Handle missing label values (T4.2)
                // Default behavior: drop rows without a target wherever the label is one (DropsMissingLabelsByDefault)
                // The legacy spellings this used to list are folded before they get here
                // (TaskTypes.Canonical, applied in the config merge), so the question left is the
                // composed one: classes that live in a column, which is what dropping rows with a
                // missing label and balancing both act on.
                var isClassificationTask =
                    AutoMLRunner.IsClassification(effectiveDefinition.Task)
                    && !DataLoaderFactory.IsDirectoryBased(effectiveDefinition.Task);

                // Use explicit parameter if provided, otherwise the task's default (see DropsMissingLabelsByDefault)
                var shouldDropMissingLabels = dropMissingLabels ?? TrainingInput.DropsMissingLabelsByDefault(effectiveDefinition.Task);

                if (shouldDropMissingLabels && !string.IsNullOrEmpty(effectiveDefinition.Label))
                {
                    var csvHelper = new CsvHelperImpl();
                    var labelHandler = new LabelValueHandler(csvHelper, new ConsoleLogger());

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
                            ValueLine.Write("[green]>[/] Using cleaned data: ", Path.GetRelativePath(projectRoot, cleanResult.OutputPath!));
                            resolvedDataFile = cleanResult.OutputPath!;
                            allDataFilesUsed.Add(resolvedDataFile);
                        }
                        else
                        {
                            ErrorConsole.Error(
                                $"Failed to clean label data: {cleanResult.Error}",
                                $"Run [blue]mloop info {Markup.Escape(Path.GetRelativePath(projectRoot, resolvedDataFile))}[/] to check the label column '{Markup.Escape(effectiveDefinition.Label)}' exists and has values.");
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
                    try
                    {
                        var distributionResult = await TrainingInput.ClassesAsync(resolvedDataFile, effectiveDefinition.Label);

                        TrainPresenter.DisplayClassDistribution(distributionResult);

                        // Single-class early termination: cannot train a classifier with one class
                        if (distributionResult.ClassCount <= 1)
                        {
                            ErrorConsole.Error(TrainingInput.SingleClassRefusal, TrainingInput.SingleClassGuidance);
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
                    var analyzer = new DataQualityAnalyzer(new ConsoleLogger());
                    var issues = await analyzer.AnalyzeAsync(resolvedDataFile, effectiveDefinition.Label);

                    TrainPresenter.DisplayDataQualityIssues(issues);

                    // Generate preprocessing script if requested
                    if (!string.IsNullOrEmpty(generateScript))
                    {
                        var scriptGenerator = new PreprocessingScriptGenerator(new ConsoleLogger());
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

                // Execute preprocessing scripts if available (Roslyn .cs scripts)
                var preprocessingEngine = new PreprocessingEngine(
                    projectRoot,
                    new ConsoleLogger());

                if (preprocessingEngine.HasPreprocessingScripts())
                {
                    AnsiConsole.WriteLine();

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

                // Apply class balancing if requested (classification only — BalanceConflict refused the rest).
                // Split BEFORE balancing to prevent data leakage
                if (!string.IsNullOrEmpty(balance) && isClassificationTask && !string.IsNullOrEmpty(effectiveDefinition.Label))
                {
                    var balanced = TrainingInput.Balance(resolvedDataFile, effectiveDefinition.Label, balance,
                        effectiveDefinition.Training?.TestSplit ?? ConfigDefaults.DefaultTestSplit);
                    var balanceResult = balanced.Result;

                    if (balanced.Split is { } splitResult)
                    {
                        AnsiConsole.WriteLine();
                        AnsiConsole.Write(new Rule("[blue]Pre-Balance Split[/]").LeftJustified());
                        AnsiConsole.WriteLine();
                        AnsiConsole.MarkupLine($"[green]✓[/] Stratified split: train={splitResult.TrainRows}, test={splitResult.TestRows}");
                        AnsiConsole.MarkupLine("[grey]  Test set is isolated from balancing to prevent data leakage.[/]");
                        AnsiConsole.WriteLine();

                        allDataFilesUsed.Add(splitResult.TrainFile);
                        allDataFilesUsed.Add(splitResult.TestFile);
                        testDataFile = splitResult.TestFile;
                    }

                    if (balanceResult.Applied && balanceResult.BalancedFilePath != null)
                    {
                        AnsiConsole.WriteLine();
                        AnsiConsole.Write(new Rule(balanced.Split is null ? "[blue]Class Balancing[/]" : "[blue]Class Balancing (train only)[/]").LeftJustified());
                        AnsiConsole.WriteLine();
                        AnsiConsole.MarkupLine($"[green]✓[/] {balanceResult.Message}");
                        ValueLine.Write(balanced.Split is null ? "[green]>[/] Using balanced data: " : "[green]>[/] Balanced train: ",
                            Path.GetRelativePath(projectRoot, balanceResult.BalancedFilePath));
                        if (balanced.TestFile is { } cleanTest)
                            ValueLine.Write("[green]>[/] Clean test: ", Path.GetRelativePath(projectRoot, cleanTest));

                        var replicationRatio = (double)balanceResult.NewMinorityCount / balanceResult.OriginalMinorityCount;
                        if (replicationRatio > 10)
                            AnsiConsole.MarkupLine($"[yellow]Warning:[/] Minority class replicated {replicationRatio:F0}x in train set.");
                        AnsiConsole.WriteLine();

                        allDataFilesUsed.Add(balanceResult.BalancedFilePath);
                    }
                    else if (!string.IsNullOrEmpty(balanceResult.Message))
                    {
                        AnsiConsole.MarkupLine($"[grey]Balance:[/] {balanceResult.Message}");
                    }
                    resolvedDataFile = balanced.TrainFile;
                }
            } // end if (!isDirectoryBased) — CSV preprocessing block

            // Display data summary (CSV row/column counts — not applicable to an image directory)
            if (!isDirectoryBased)
                TrainPresenter.DisplayDataSummary(resolvedDataFile, effectiveDefinition.Label, sourceDataFile);

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
            TrainPresenter.DisplayTrainingConfig(resolvedDataFile, resolvedModelName, effectiveDefinition, testDataFile, useAutoTime, sourceDataFile);

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
                new ConsoleLogger());

            // Ensure model directory structure exists
            await modelNameResolver.EnsureModelDirectoryAsync(resolvedModelName);

            var trainingConfig = ModelTrainingConfig.From(
                effectiveDefinition,
                resolvedModelName,
                Path.GetFullPath(resolvedDataFile),
                effectiveDefinition.Training?.TimeLimitSeconds ?? ConfigDefaults.DefaultTimeLimitSeconds,
                testDataFile != null ? 0 : (effectiveDefinition.Training?.TestSplit ?? ConfigDefaults.DefaultTestSplit),
                testDataFile != null ? Path.GetFullPath(testDataFile) : null,
                useAutoTime,
                preFeaturizer,
                preFeaturizerColumns);

            // The project's pre- and post-train hooks run inside TrainingEngine, the one place for
            // `mloop train` and `POST /train` alike. They used to run here as well, so each ran twice.

            // If auto-time, display static estimate before progress bar starts. A fit that takes no
            // budget (deep learning, time series) would have an estimate describing time nothing uses.
            if (useAutoTime && AutoMLRunner.TimeBudgetUnused(trainingConfig.Task) is null)
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
                    var progressTask = ctx.AddTask(TrainingProgressTracker.StartDescription(resolvedModelName), maxValue: 100);

                    var progressTracker = new TrainingProgressTracker(trainingConfig.TimeLimitSeconds);

                    var progress = new Progress<TrainingProgress>(p =>
                    {
                        if (p.Phase.HasValue)
                        {
                            events?.Phase(p);
                            // The post-run probe summary keys off the last *probe* phase; the
                            // window boundaries (MainStart/Complete) every run now reports would
                            // otherwise overwrite it and silently drop the summary.
                            if (p.Phase is TrainingPhase.ProbeStart or TrainingPhase.ProbeComplete
                                         or TrainingPhase.ProbeConverged or TrainingPhase.ProbeFellBack)
                                lastAutoTimeEvent = p;
                            progressTracker.EnterPhase(p);
                            progressTask.Description =
                                TrainingProgressTracker.PhaseDescription(p, resolvedModelName) ?? progressTask.Description;
                            if (TrainingProgressTracker.FitPercent(p) is { } fitPercent)
                                progressTask.Value = fitPercent;
                            return;
                        }

                        events?.Trial(p);

                        progressTask.Description = TrainingProgressTracker.TrialDescription(p);

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
            else if (lastAutoTimeEvent?.Phase == TrainingPhase.ProbeFellBack)
            {
                TrainPresenter.DisplayProbeFellBack(
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
                ErrorConsole.Out.MarkupLine("  [blue]>[/] Check data quality: [cyan]mloop analyze profile <data-file>[/]");
                ErrorConsole.Out.MarkupLine("  [blue]>[/] Verify label column exists and has valid values");
                return 1;
            }

            // Sync mloop.yaml if CLI overrode label or task
            await SyncYamlConfigAsync(configLoader, userConfig, resolvedModelName, effectiveDefinition, label, task, groupColumn);

            // Emitted here, before auto-promote: the experiment is finished and recorded at this
            // point. A promotion that fails afterwards adds an error event and a non-zero exit, so a
            // consumer must not read "a result arrived" as "the command succeeded" — the exit code
            // is what says that.
            events?.Result(result, resolvedModelName);

            // Display results
            TrainPresenter.DisplayResults(result, resolvedModelName, promotionFollows: !noPromote, records, effectiveDefinition.Task,
                trainingConfig.WeightColumn);

            // T4.4: Performance diagnostics
            var performanceDiagnostics = new PerformanceDiagnostics();
            var diagnosticResult = performanceDiagnostics.Analyze(
                trainingConfig.Task,
                result.Metrics,
                rowCount: (int)Math.Min(result.RowCount, int.MaxValue));

            TrainPresenter.DisplayDiagnostics(diagnosticResult);

            // A post-train hook that refused the model keeps it from being promoted automatically.
            if (!noPromote && result.HookRefusal is { } hookRefusal)
            {
                AnsiConsole.WriteLine();
                WarningConsole.Warn(
                    $"Kept in staging: a post-train hook refused it — {Markup.Escape(hookRefusal)}. " +
                    $"[blue]mloop promote {result.ExperimentId}[/] promotes it anyway.");
            }
            // Auto-promote to production if enabled
            else if (!noPromote)
            {
                AnsiConsole.WriteLine();
                var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);
                var primaryMetric = trainingConfig.CanonicalMetric;

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
                double? majorityClassRatio = null;
                string? gateRefusal = null;
                string? incomparable = null;
                string? policyRefusal = null;
                if (!promoted)
                {
                    try
                    {
                        var expData = await experimentStore.LoadAsync(resolvedModelName, result.ExperimentId, CancellationToken.None);
                        gateRefusal = ModelRegistry.QualityGateRefusal(expData, primaryMetric);
                        incomparable = await modelRegistry.IncomparableWithProductionAsync(resolvedModelName, expData, production);
                        // When the metrics alone would have promoted it, the gate policy is what held it back —
                        // asked again so the staging line names the rule (an evaluation, only when one is configured).
                        if (await modelRegistry.ShouldPromoteOnMetricsAsync(resolvedModelName, result.ExperimentId, primaryMetric))
                            policyRefusal = await modelRegistry.GatePolicyRefusalAsync(resolvedModelName, result.ExperimentId);
                        var labelSchema = expData.Config?.InputSchema?.Columns?
                            .FirstOrDefault(s => s.Name.Equals(trainingConfig.LabelColumn, StringComparison.OrdinalIgnoreCase));
                        if (labelSchema?.UniqueValueCount > 0)
                            classCount = labelSchema.UniqueValueCount;
                        // Read alongside the class count so the threshold shown is the one the gate
                        // applied — a display that omits it would understate why the model was held.
                        majorityClassRatio = labelSchema?.MajorityClassRatio;
                    }
                    catch (Exception) { /* schema unavailable, use default threshold */ }
                }

                // Display the threshold the gate actually applied: resolve "auto"/aliases to the
                // canonical key first (matches ShouldPromoteAsync), so image tasks show their 1/N
                // floor instead of a blank.
                var displayMetricKey = result.Metrics != null
                    ? MetricPolicy.ResolveCanonicalMetricKey(primaryMetric, trainingConfig.Task, result.Metrics.Keys)
                    : null;
                var minThreshold = displayMetricKey != null
                    ? MetricPolicy.GetMinimumMetricThreshold(displayMetricKey, classCount, majorityClassRatio)
                    : null;
                // Report the resolved key, not the raw request: the presenter both names the metric
                // in its messages and looks it up in result.Metrics. An unresolved sentinel made
                // every TryGetValue miss, so the staging *reason* silently vanished instead of
                // being shown.
                TrainPresenter.DisplayPromotionResult(promoted, displayMetricKey ?? primaryMetric, result, production, minThreshold, majorityClassRatio, gateRefusal, incomparable, policyRefusal);

                // Promoted or not, a model that ranks but answers one class at its threshold is told so.
                if (result.Metrics != null && MetricPolicy.SingleClassAtThreshold(result.Metrics) is { } gateWarning)
                    WarningConsole.Warn($"Quality: {gateWarning}.");
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
            // DisplayTrainingError has existed since 0.28.0 (its CHANGELOG entry describes it as
            // already wired here) but was never actually called from this catch — every training
            // failure rendered through the generic DisplayError instead, losing the model name in
            // the heading and the "mloop analyze"/"mloop status" diagnostic hints the dedicated
            // path was built to add. resolvedModelName is recomputed here (pure function of
            // modelName, safe to redo) since the try-scoped local isn't visible in catch.
            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? ConfigDefaults.DefaultModelName
                : modelName.Trim().ToLowerInvariant();
            ErrorSuggestions.DisplayTrainingError(ex, resolvedModelName, dataFile, failedTask);
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
            Path.Combine(projectRoot, ProjectLayout.DatasetsDirectory, conventionDir),
            Path.Combine(projectRoot, ProjectLayout.DatasetsDirectory)
        })
        {
            if (Directory.Exists(candidate))
                return candidate;
        }

        return null;
    }

    /// <summary>
    /// Updates mloop.yaml when CLI-specified label, task or ranking group column differs from the
    /// stored config, so the next command reads the columns this model was trained with.
    /// </summary>
    internal static async Task SyncYamlConfigAsync(
        ConfigLoader configLoader,
        MLoopConfig userConfig,
        string modelName,
        ModelDefinition effectiveDefinition,
        string? cliLabel,
        string? cliTask,
        string? cliGroupColumn = null)
    {
        // Only sync if CLI actually overrode something
        if (string.IsNullOrEmpty(cliLabel) && string.IsNullOrEmpty(cliTask) && string.IsNullOrEmpty(cliGroupColumn))
            return;

        var yamlModel = userConfig.Models.TryGetValue(modelName, out var m) ? m : null;

        bool needsUpdate = false;
        if (!string.IsNullOrEmpty(cliLabel) && (yamlModel == null || !string.Equals(yamlModel.Label, cliLabel, StringComparison.Ordinal)))
            needsUpdate = true;
        if (!string.IsNullOrEmpty(cliTask) && (yamlModel == null || !string.Equals(yamlModel.Task, cliTask, StringComparison.OrdinalIgnoreCase)))
            needsUpdate = true;
        if (!string.IsNullOrEmpty(cliGroupColumn) && (yamlModel == null || !string.Equals(yamlModel.GroupColumn, cliGroupColumn, StringComparison.Ordinal)))
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
            if (!string.IsNullOrEmpty(cliGroupColumn))
                yamlModel.GroupColumn = cliGroupColumn;
        }
        else
        {
            // Create new model definition in yaml
            userConfig.Models[modelName] = new ModelDefinition
            {
                Task = effectiveDefinition.Task,
                Label = effectiveDefinition.Label,
                GroupColumn = effectiveDefinition.GroupColumn,
                Training = effectiveDefinition.Training
            };
        }

        try
        {
            await configLoader.SaveUserConfigAsync(userConfig);
            var group = string.IsNullOrEmpty(effectiveDefinition.GroupColumn) ? "" : $", group_column={effectiveDefinition.GroupColumn}";
            var text = $"Updated mloop.yaml: model '{modelName}' → label={effectiveDefinition.Label}, task={effectiveDefinition.Task}{group}";
            AnsiConsole.MarkupLine($"[grey]{Markup.Escape(text)}[/]");
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
            var executor = new DataPipelineExecutor(new ConsoleLogger());
            outFile = await executor.ExecuteAsync(dataFile, route.CsvSteps, prepOutputPath);
        }

        return (outFile, route.PreFeaturizer, route.Warnings, route.PreFeaturizerColumns);
    }
}
