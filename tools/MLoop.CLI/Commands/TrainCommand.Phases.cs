using System.CommandLine;
using Microsoft.ML;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.DataQuality;
using MLoop.Core.Diagnostics;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;
using MLoop.Core.Preprocessing;
using MLoop.Core.Storage;
using MLoop.Core.Training;
using MLoop.Extensibility;
using Spectre.Console;

namespace MLoop.CLI.Commands;

// The phases of `mloop train`, in the order ExecuteAsync runs them. Each reports its own failure
// (already printed) by returning null or an exit code, so ExecuteAsync reads as the sequence.
public static partial class TrainCommand
{
    private sealed record TrainSetup(
        string ProjectRoot,
        string ModelName,
        ConfigLoader ConfigLoader,
        MLoopConfig UserConfig,
        ModelDefinition Definition,
        bool RequiresLabel);

    /// <summary>The rows training will read, after the tabular preparation steps; <see cref="ExitCode"/> is set when the run ends there.</summary>
    private sealed record PreparedData(
        string DataFile,
        string? TestDataFile,
        List<string> AllDataFilesUsed,
        IEstimator<ITransformer>? PreFeaturizer,
        List<string> PreFeaturizerColumns,
        int? ExitCode)
    {
        public static PreparedData Exit(int code) => new("", null, [], null, [], code);
    }

    /// <summary>Project, model name, configuration and the validated effective definition — null after an error was shown.</summary>
    private static async Task<TrainSetup?> ResolveSetupAsync(
        string? modelName, string? label, string? task, int? time, int? maxTrials, string? metric, double? testSplit,
        string? groupColumn, string? weight, string? balance, bool portable, string? encoder, int? bankSize,
        FileSystemManager fileSystem, ProjectDiscovery projectDiscovery, Action<string?> noteTask)
    {
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
            return null;
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
        if (time.HasValue || maxTrials.HasValue || !string.IsNullOrEmpty(metric) || testSplit.HasValue)
        {
            cliTraining = new TrainingSettings
            {
                TimeLimitSeconds = time,
                MaxTrials = maxTrials,
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
            noteTask(effectiveDefinition.Task);
        }
        catch (InvalidOperationException ex)
        {
            ErrorConsole.Error($"{ex.Message}");
            return null;
        }

        // Apply CLI --group-column override
        if (!string.IsNullOrEmpty(groupColumn))
            effectiveDefinition.GroupColumn = groupColumn;

        // Apply CLI --weight override, then refuse a weight the run could not honour — a weight
        // that silently did nothing would train an unweighted model the user believes weighted.
        if (!string.IsNullOrEmpty(weight))
            effectiveDefinition.Weight = weight;
        if ((ModelTrainingConfig.MaxTrialsConflict(effectiveDefinition)
             ?? TrainingInput.BalanceConflict(effectiveDefinition, balance)
             ?? ModelTrainingConfig.WeightConflict(effectiveDefinition, balance)) is { } weightError)
        {
            ErrorConsole.Error(weightError);
            return null;
        }

        if (!string.IsNullOrWhiteSpace(encoder))
            effectiveDefinition.Encoder = encoder;
        if (bankSize is not null)
            effectiveDefinition.BankSize = bankSize;

        if (portable)
            effectiveDefinition.Portable = true;
        if (ModelTrainingConfig.PortableConflict(effectiveDefinition) is { } portableError)
        {
            ErrorConsole.Error(portableError);
            return null;
        }

        // Validate required fields (defensive guard for nullable flow analysis).
        // TrainingRunner.RequiresLabel is the single source — previously this inline set omitted
        // time-series-anomaly, so a label-less ts-anomaly project that merge/init accept failed here.
        var requiresLabel = TrainingRunner.RequiresLabel(effectiveDefinition.Task);

        if (string.IsNullOrEmpty(effectiveDefinition.Task) ||
            (requiresLabel && string.IsNullOrEmpty(effectiveDefinition.Label)))
        {
            ErrorConsole.Error("Task is required. Label is required for supervised tasks. Use --task and <label> arguments, or define them in mloop.yaml");
            return null;
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
            return null;
        }


        // Ranking requires group column
        if (effectiveDefinition.Task.Equals("ranking", StringComparison.OrdinalIgnoreCase) &&
            string.IsNullOrEmpty(effectiveDefinition.GroupColumn))
        {
            ErrorConsole.Error("Ranking task requires a group column. Use --group-column or set 'group_column' in mloop.yaml");
            return null;
        }

        // Recommendation requires user_column and item_column
        if (effectiveDefinition.Task.Equals("recommendation", StringComparison.OrdinalIgnoreCase))
        {
            if (string.IsNullOrEmpty(effectiveDefinition.UserColumn) || string.IsNullOrEmpty(effectiveDefinition.ItemColumn))
            {
                ErrorConsole.Error("Recommendation task requires user_column and item_column. Set them in mloop.yaml");
                return null;
            }
        }

        // Forecasting requires horizon
        if (effectiveDefinition.Task.Equals("forecasting", StringComparison.OrdinalIgnoreCase) &&
            (effectiveDefinition.Horizon ?? 0) <= 0)
        {
            ErrorConsole.Error("Forecasting task requires horizon > 0. Set 'horizon' in mloop.yaml");
            return null;
        }

        // Refuse a metric the task's optimizer cannot take before any work starts — the run would
        // otherwise train for minutes on the task default while recording the name given here.
        var metricRejection = MetricNames.Rejection(
            effectiveDefinition.Task, effectiveDefinition.Training?.Metric ?? ConfigDefaults.DefaultMetric);
        if (metricRejection != null)
        {
            ErrorConsole.Error(Markup.Escape(metricRejection), "Set 'metric' in mloop.yaml or pass --metric with one of the names above.");
            return null;
        }

        return new TrainSetup(projectRoot, resolvedModelName, configLoader, userConfig, effectiveDefinition, requiresLabel);
    }

    /// <summary>The file or directory the run trains on, named and announced — null after an error was shown.</summary>
    private static async Task<string?> ResolveDataFileAsync(
        ModelDefinition effectiveDefinition, string? dataFile, string[]? dataPaths, bool autoMerge,
        bool autoTime, bool noAutoTime, int? time, string? records, string projectRoot,
        MLoopConfig userConfig, FileSystemManager fileSystem, DatasetDiscovery datasetDiscovery)
    {
        // Resolve data file path
        string? resolvedDataFile;
        var isDirectoryBased = DataLoaderFactory.IsDirectoryBased(effectiveDefinition.Task);

        // Validate --auto-time opt-in against contradictory inputs before doing any work.
        if (autoTime)
        {
            if (noAutoTime)
            {
                ErrorConsole.Error("--auto-time and --no-auto-time cannot be used together.");
                return null;
            }
            if (time.HasValue)
            {
                ErrorConsole.Error("--auto-time cannot be combined with an explicit --time value.");
                return null;
            }
            if (isDirectoryBased)
            {
                ErrorConsole.Error("--auto-time is not supported for image/directory training (time estimation probes CSV rows).");
                return null;
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
                return null;
            }
        }
        // T4.3: Handle --data option for external data paths
        else if (dataPaths is { Length: > 0 })
        {
            // Resolve all paths (support relative and absolute)
            var resolvedPaths = new List<string>();
            foreach (var path in dataPaths)
            {
                var resolved = ProjectPaths.Resolve(path, projectRoot);
                if (!TabularDataFile.Exists(resolved))
                {
                    ErrorConsole.Error($"Data file not found: {path}", ErrorConsole.PathNotFoundTip(projectRoot));
                    return null;
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
                    return null;
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
                            AnsiConsole.MarkupLineInterpolated($"    [grey]• {fileName}: {rowCount} rows[/]");
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

        if (resolvedDataFile == null && !string.IsNullOrEmpty(dataFile))
        {
            ErrorConsole.Error(Markup.Escape($"Data file not found: {dataFile}"), ErrorConsole.PathNotFoundTip(projectRoot));
            return null;
        }
        if (resolvedDataFile == null)
        {
            ErrorConsole.Error(
                "No data file specified and datasets/train.csv not found.",
                "Create datasets/train.csv or specify a file: mloop train <data-file>");
            return null;
        }

        ValueLine.Write("[green]>[/] Using data: ", Path.GetRelativePath(projectRoot, resolvedDataFile));

        return resolvedDataFile;
    }

    /// <summary>
    /// CSV preprocessing (flatten, sampling, cleaning, class analysis, preprocessing, balancing, splitting) —
    /// tabular tasks only; image classification loads a directory and skips it.
    /// </summary>
    private static async Task<PreparedData> PrepareTabularDataAsync(
        string resolvedDataFile, ModelDefinition effectiveDefinition, string projectRoot, string? records,
        int? maxRows, string? samplingStrategy, int? seed, bool? dropMissingLabels, bool analyzeData,
        string? generateScript, string? balance)
    {
        var allDataFilesUsed = new List<string> { resolvedDataFile };
        string? testDataFile = null;
        IEstimator<ITransformer>? preFeaturizer = null;
        List<string> preFeaturizerColumns = new();

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
                return PreparedData.Exit(1);
            }
        }

        // Handle missing label values (T4.2)
        // Default behavior: drop rows without a target wherever the label is one (DropsMissingLabelsByDefault)
        // The legacy spellings this used to list are folded before they get here
        // (TaskTypes.Canonical, applied in the config merge), so the question left is the
        // composed one: classes that live in a column, which is what dropping rows with a
        // missing label and balancing both act on.
        var isClassificationTask =
            TrainingRunner.IsClassification(effectiveDefinition.Task)
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
                AnsiConsole.MarkupLineInterpolated($"[yellow]Warning:[/] Label analysis error: {labelAnalysis.Error}");
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
                        $"Run [blue]mloop info {Path.GetRelativePath(projectRoot, resolvedDataFile)}[/] to check the label column '{effectiveDefinition.Label}' exists and has values.");
                    return PreparedData.Exit(1);
                }
            }
            else
            {
                AnsiConsole.MarkupLineInterpolated($"[green]✓[/] Label column '{effectiveDefinition.Label}' has no missing values ({labelAnalysis.ValidCount} rows, {labelAnalysis.UniqueValueCount} unique values)");
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
                    return PreparedData.Exit(1);
                }
            }
            catch (Exception ex)
            {
                AnsiConsole.MarkupLineInterpolated($"[yellow]Warning:[/] Class distribution analysis failed: {ex.Message}");
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
                return PreparedData.Exit(0);
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
                return PreparedData.Exit(1);
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
                AnsiConsole.MarkupLineInterpolated($"[green]✓[/] {balanceResult.Message}");
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
                AnsiConsole.MarkupLineInterpolated($"[grey]Balance:[/] {balanceResult.Message}");
            }
            resolvedDataFile = balanced.TrainFile;
        }

        return new PreparedData(resolvedDataFile, testDataFile, allDataFilesUsed, preFeaturizer, preFeaturizerColumns, null);
    }

    /// <summary>Fits under the progress display and returns the result; null when training produced none.</summary>
    private static async Task<TrainingResult?> RunTrainingAsync(
        TrainingEngine trainingEngine, TrainingConfig trainingConfig, string resolvedModelName, TrainJsonEmitter? events)
    {
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
                                     or TrainingPhase.ProbeConverged)
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

        return result;
    }

    /// <summary>
    /// The automatic promotion and what is said about it: the comparison with production, the gate's verdict
    /// and the threshold it applied, and the matching event for a stream consumer.
    /// </summary>
    private static async Task AutoPromoteAndReportAsync(
        string resolvedModelName, TrainingResult result, TrainingConfig trainingConfig,
        FileSystemManager fileSystem, ProjectDiscovery projectDiscovery, ExperimentStore experimentStore,
        TrainJsonEmitter? events)
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

        // Asked only of a stream consumer: the reason can run the gate policy's evaluation again.
        if (events != null)
            events.Promotion(result.ExperimentId, promoted, promoted
                ? null
                : await modelRegistry.KeptInStagingReasonAsync(resolvedModelName, result.ExperimentId, primaryMetric, CancellationToken.None)
                  ?? "it does not beat production");

        // Promoted or not, a model that ranks but answers one class at its threshold is told so.
        if (result.Metrics != null && MetricPolicy.SingleClassAtThreshold(result.Metrics) is { } gateWarning)
            WarningConsole.Warn($"Quality: {gateWarning}.");
    }
}
