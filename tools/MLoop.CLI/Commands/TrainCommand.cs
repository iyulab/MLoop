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
using MLoop.Core.Hooks;
using MLoop.Core.Models;
using MLoop.Core.Preprocessing;
using MLoop.Core.Storage;
using MLoop.Core.Training;
using MLoop.Extensibility;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Preprocessing;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop train - Trains a model using AutoML with multi-model support
/// </summary>
public static partial class TrainCommand
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

        var maxTrialsOption = new Option<int?>("--max-trials")
        {
            Description = "Stop the trainer search after this many trials (within --time) — the same search on a fast machine and a slow one"
        };

        var epochsOption = new Option<int?>("--epochs")
        {
            Description = "Epochs a deep-learning task trains for (default 10) — the time limit does not shorten such a fit; text, sentence-similarity, NER, question-answering and object-detection tasks"
        };

        var metricOption = new Option<string?>("--metric")
        {
            Description = "Optimization metric (accuracy, auc, f1, r_squared, etc.). Default auto: AUC for binary, log-loss for multiclass classification, the task's primary metric otherwise"
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

        var balanceOption = new Option<string?>("--balance")
        {
            Description = "Class balancing strategy: 'auto' (balance to 10:1 if ratio > 10:1), 'none' (no balancing), or target ratio (e.g., '5' for 5:1). Use without value for 'auto'.",
            Arity = ArgumentArity.ZeroOrOne
        };
        var weightOption = new Option<string?>("--weight")
        {
            Description = "Column that weights the rows (relative frequency, e.g. 1/π for sampled labels); never a feature. Binary, multiclass and regression"
        };
        var encoderOption = new Option<string?>("--encoder")
        {
            Description = "Image encoder an image-anomaly model embeds its images with (an installed encoder's id; default: the only one installed)"
        };
        var bankSizeOption = new Option<int?>("--bank-size")
        {
            Description = "The most patches an image-anomaly model's bank holds (default: every patch of the images; with a limit, a greedy k-center coreset keeps the patches that cover what the images showed, chosen with a fixed seed)"
        };
        var portableOption = new Option<bool>("--portable")
        {
            Description = "Train only with steps ONNX can express, so 'mloop export onnx' can export the model (text columns use character n-grams). Binary, multiclass and regression"
        };
        var ensembleOption = new Option<bool>("--ensemble")
        {
            Description = "Answer with a weighted blend of the best trials when it scores better out of fold (binary classification and regression). Slower to fit and to score, and cannot be exported to ONNX; the search is given 70 % of the time limit and the rest fits the blend"
        };
        var groupColumnOption = new Option<string?>("--group-column")
        {
            Description = "Group column for ranking task (groups rows into query contexts)"
        };

        var userColumnOption = new Option<string?>("--user-column")
        {
            Description = "User column for the recommendation task (who rated)"
        };
        var itemColumnOption = new Option<string?>("--item-column")
        {
            Description = "Item column for the recommendation task (what was rated)"
        };

        var maxRowsOption = new Option<int?>("--max-rows")
        {
            Description = "Maximum training rows. Data exceeding this is auto-sampled (random for regression/anomaly, stratified for classification)"
        };

        var samplingStrategyOption = new Option<string?>("--sampling-strategy")
        {
            Description = $"Sampling strategy when --max-rows is used: {string.Join(" or ", SamplingStrategies.Names.Select(n => $"'{n}'"))} (default: task-aware)"
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
            Description = "Emit newline-delimited JSON events (phase/trial/warning/result/promotion/error) on stdout instead of the rich display",
            DefaultValueFactory = _ => false
        };

        var command = new Command("train", "Train a model using AutoML");
        command.Arguments.Add(dataFileArg);
        command.Options.Add(nameOption);
        command.Options.Add(labelOption);
        command.Options.Add(taskOption);
        command.Options.Add(timeOption);
        command.Options.Add(maxTrialsOption);
        command.Options.Add(epochsOption);
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
        command.Options.Add(userColumnOption);
        command.Options.Add(itemColumnOption);
        command.Options.Add(weightOption);
        command.Options.Add(encoderOption);
        command.Options.Add(bankSizeOption);
        command.Options.Add(portableOption);
        command.Options.Add(ensembleOption);
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
            var userColumn = parseResult.GetValue(userColumnOption);
            var itemColumn = parseResult.GetValue(itemColumnOption);
            var maxRows = parseResult.GetValue(maxRowsOption);
            var samplingStrategy = parseResult.GetValue(samplingStrategyOption) ?? parseResult.GetValue(samplingStrategyAliasOption);
            var seed = parseResult.GetValue(seedOption);
            var json = parseResult.GetValue(jsonOption);
            // Handle --balance without argument: default to "auto"
            if (balance == null && parseResult.Tokens.Any(t => t.Value == "--balance" || t.Value == "-b"))
            {
                balance = "auto";
            }
            return ExecuteAsync(dataFile, name, label, task, time, metric, testSplit, noPromote, analyzeData, generateScript, autoMerge, dropMissingLabels, dataPaths, noAutoTime, autoTime, balance, groupColumn, maxRows, samplingStrategy, seed, json, parseResult.GetValue(recordsOption), parseResult.GetValue(weightOption), parseResult.GetValue(portableOption), parseResult.GetValue(encoderOption), parseResult.GetValue(bankSizeOption), parseResult.GetValue(maxTrialsOption), parseResult.GetValue(epochsOption), parseResult.GetValue(ensembleOption), userColumn, itemColumn);
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
        bool portable = false,
        string? encoder = null,
        int? bankSize = null,
        int? maxTrials = null,
        int? epochs = null,
        bool ensemble = false,
        string? userColumn = null,
        string? itemColumn = null)
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
            if (SamplingStrategies.Refusal(samplingStrategy) is { } unknownStrategy)
            {
                ErrorConsole.Error(Markup.Escape(unknownStrategy), SamplingStrategies.RefusalTip);
                return 1;
            }

            // Initialize components
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);

            var setup = await ResolveSetupAsync(
                modelName, label, task, time, maxTrials, epochs, metric, testSplit, groupColumn, weight, balance, portable, ensemble, encoder, bankSize,
                userColumn, itemColumn, fileSystem, projectDiscovery, resolved => failedTask = resolved);
            if (setup is null)
                return 1;
            var (projectRoot, resolvedModelName, configLoader, userConfig, effectiveDefinition, requiresLabel) = setup;

            var datasetDiscovery = new DatasetDiscovery(fileSystem);
            var isDirectoryBased = DataLoaderFactory.IsDirectoryBased(effectiveDefinition.Task);

            // Options that shape a table's rows do nothing to a directory of images; say so rather than let them be believed.
            var tableOnly = new List<string>();
            if (maxRows is not null) tableOnly.Add("--max-rows");
            if (!string.IsNullOrEmpty(samplingStrategy)) tableOnly.Add("--sampling-strategy");
            if (seed is not null) tableOnly.Add("--seed");
            if (!string.IsNullOrEmpty(records)) tableOnly.Add("--records");
            if (autoMerge) tableOnly.Add("--auto-merge");
            if (dropMissingLabels is not null) tableOnly.Add("--drop-missing-labels");
            if (analyzeData) tableOnly.Add("--analyze-data");
            if (!string.IsNullOrEmpty(generateScript)) tableOnly.Add("--generate-script");
            if (dataPaths is { Length: > 1 }) tableOnly.Add("--data with several paths");
            if (TrainingInput.TableOnlyOptionsRefusal(effectiveDefinition.Task, tableOnly) is { } tableOnlyRefusal)
            {
                ErrorConsole.Error(tableOnlyRefusal);
                return 1;
            }

            var resolvedDataFile = await ResolveDataFileAsync(
                effectiveDefinition, dataFile, dataPaths, autoMerge, autoTime, noAutoTime, time, records,
                projectRoot, userConfig, fileSystem, datasetDiscovery);
            if (resolvedDataFile is null)
                return 1;

            // Every step below may swap resolvedDataFile for a converted, flattened or split copy;
            // what the user is told they trained on is still the file they named.
            var sourceDataFile = resolvedDataFile;

            // The prepared tabular file, its held-out test file when a pre-balance split made one, and
            // every file the run touched (read by the unused-data scan after training). A directory of
            // images skips the tabular preparation.
            string? testDataFile = null;
            var allDataFilesUsed = new List<string> { resolvedDataFile };
            IEstimator<ITransformer>? preFeaturizer = null;
            List<string> preFeaturizerColumns = new();

            if (!isDirectoryBased)
            {
                var prepared = await PrepareTabularDataAsync(
                    resolvedDataFile, effectiveDefinition, projectRoot, records, maxRows, samplingStrategy, seed,
                    dropMissingLabels, analyzeData, generateScript, balance, events);
                if (prepared.ExitCode is { } exitCode)
                    return exitCode;
                resolvedDataFile = prepared.DataFile;
                testDataFile = prepared.TestDataFile;
                allDataFilesUsed = prepared.AllDataFilesUsed;
                preFeaturizer = prepared.PreFeaturizer;
                preFeaturizerColumns = prepared.PreFeaturizerColumns;
            }

            // Display data summary (CSV row/column counts — not applicable to an image directory)
            if (!isDirectoryBased)
                TrainPresenter.DisplayDataSummary(resolvedDataFile, effectiveDefinition.Label, sourceDataFile);

            // Determine auto-time eligibility:
            // - If --auto-time was explicitly specified, force auto-time even when mloop.yaml
            //   sets time_limit_seconds (the project-workflow opt-in; contradictions already rejected above).
            // - If --time was explicitly specified, use that value (no auto-time)
            // - If --no-auto-time was specified, use default 300s (no auto-time)
            // - If a trial limit is set (--max-trials / training.max_trials), no auto-time: the time is its ceiling
            // - If neither --time nor yaml time_limit_seconds is set, enable auto-time
            var yamlHasTimeLimit = effectiveDefinition.Training?.TimeLimitSeconds != null;
            // A trial limit bounds the work itself — the time is then only its ceiling (the default, or --time):
            // an estimated budget would run the same fixed-seed search twice.
            var hasTrialLimit = effectiveDefinition.Training?.MaxTrials != null;
            // Auto-time probing samples CSV rows, which does not apply to image directories.
            var useAutoTime = autoTime
                || (!time.HasValue && !yamlHasTimeLimit && !hasTrialLimit && !noAutoTime && !isDirectoryBased);

            // Display training configuration (auto-time shows an "auto" limit, not the yaml/default seconds,
            // so the summary matches what the engine actually does — honest record vs. actual)
            TrainPresenter.DisplayTrainingConfig(resolvedDataFile, resolvedModelName, effectiveDefinition, testDataFile, useAutoTime, sourceDataFile, balance);

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
            // A directory of images is not a file to balance up front: the trainer oversamples its own training split.
            if (DataLoaderFactory.IsDirectoryBased(trainingConfig.Task) && ClassBalance.Requested(balance))
                trainingConfig = trainingConfig with { Balance = balance };

            // The project's pre- and post-train hooks run inside TrainingEngine, the one place for
            // `mloop train` and `POST /train` alike. They used to run here as well, so each ran twice.

            // If auto-time, display static estimate before progress bar starts. A fit that takes no
            // budget (deep learning, time series) would have an estimate describing time nothing uses.
            if (useAutoTime && TrainingRunner.TimeBudgetUnused(trainingConfig.Task) is null)
            {
                var (rowCount, colCount, hasText, classCount) = TrainingEngine.CollectDataStats(
                    trainingConfig.DataFile, trainingConfig.LabelColumn, trainingConfig.Task);
                var staticEstimate = TimeEstimator.EstimateStatic(rowCount, colCount, trainingConfig.Task, classCount, hasText);
                TrainPresenter.DisplayAutoTimeEstimate(staticEstimate, rowCount, colCount, trainingConfig.Task, classCount);
                AnsiConsole.WriteLine();
            }

            // Train with progress display
            var result = await RunTrainingAsync(trainingEngine, trainingConfig, resolvedModelName, events);

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
            await SyncYamlConfigAsync(configLoader, userConfig, resolvedModelName, effectiveDefinition, label, task, groupColumn, userColumn, itemColumn);

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

            // The block above is narration, silent in machine mode; its warnings are not. A --json
            // consumer gets each as a warning event, as it gets every other warning of the run.
            foreach (var warning in diagnosticResult.Warnings)
                MachineOutputScope.ReportWarning(Markup.Escape(warning));

            // A post-train hook that refused the model keeps it from being promoted automatically.
            if (noPromote)
            {
                events?.Promotion(result.ExperimentId, promoted: false,
                    $"--no-promote was given — mloop promote {result.ExperimentId} promotes it");
            }
            else if (result.HookRefusal is { } hookRefusal)
            {
                AnsiConsole.WriteLine();
                WarningConsole.Warn(
                    $"Kept in staging: a post-train hook refused it — {hookRefusal}. " +
                    $"[blue]mloop promote {result.ExperimentId}[/] promotes it anyway.");
                events?.Promotion(result.ExperimentId, promoted: false, $"a post-train hook refused it ({hookRefusal})");
            }
            // Auto-promote to production
            else
            {
                await AutoPromoteAndReportAsync(resolvedModelName, result, trainingConfig, fileSystem, projectDiscovery, experimentStore, events);
            }

            // T4.6: Unused data warning - only scan project's datasets/ directory.
            // The scan targets CSV datasets, so skip it for image directory training.
            var datasetsDir = datasetDiscovery.GetDatasetsPath(projectRoot);
            if (!isDirectoryBased && Directory.Exists(datasetsDir))
            {
                // The files evaluate and predict read — by convention, or where mloop.yaml points them — are not
                // unused: training is not what they are for.
                var declaredElsewhere = new List<string>
                {
                    Path.Combine(datasetsDir, ProjectLayout.TestFileName),
                    Path.Combine(datasetsDir, ProjectLayout.PredictFileName),
                };
                if (!string.IsNullOrWhiteSpace(userConfig.Data?.Test))
                    declaredElsewhere.Add(ProjectPaths.Resolve(userConfig.Data.Test, projectRoot));
                if (!string.IsNullOrWhiteSpace(userConfig.Data?.Predict))
                    declaredElsewhere.Add(ProjectPaths.Resolve(userConfig.Data.Predict, projectRoot));

                var unusedDataScanner = new UnusedDataScanner();
                var scanResult = unusedDataScanner.Scan(datasetsDir, allDataFilesUsed, declaredElsewhere);
                TrainPresenter.DisplayUnusedDataWarning(scanResult);
            }

            return 0;
        }
        catch (MLoop.Core.Storage.InvalidModelNameException ex)
        {
            // A name that cannot be a directory is refused before anything is read or created.
            ErrorConsole.Error(ex.Message, "Choose a name of lowercase letters, digits and single hyphens, e.g. --name my-model.");
            return 1;
        }
        catch (Exception ex)
        {
            // DisplayTrainingError has existed since 0.28.0 (its CHANGELOG entry describes it as
            // already wired here) but was never actually called from this catch — every training
            // failure rendered through the generic DisplayError instead, losing the model name in
            // the heading and the "mloop analyze"/"mloop status" diagnostic hints the dedicated
            // path was built to add. resolvedModelName is recomputed here (pure function of
            // modelName, safe to redo) since the try-scoped local isn't visible in catch.
            var resolvedModelName = MLoop.Core.Storage.ModelName.Resolve(modelName);
            ErrorSuggestions.DisplayTrainingError(ex, resolvedModelName, dataFile, failedTask);
            return 1;
        }
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
        string? cliGroupColumn = null,
        string? cliUserColumn = null,
        string? cliItemColumn = null)
    {
        // Only sync if CLI actually overrode something
        if (string.IsNullOrEmpty(cliLabel) && string.IsNullOrEmpty(cliTask) && string.IsNullOrEmpty(cliGroupColumn)
            && string.IsNullOrEmpty(cliUserColumn) && string.IsNullOrEmpty(cliItemColumn))
            return;

        var yamlModel = userConfig.Models.TryGetValue(modelName, out var m) ? m : null;

        bool needsUpdate = false;
        if (!string.IsNullOrEmpty(cliLabel) && (yamlModel == null || !string.Equals(yamlModel.Label, cliLabel, StringComparison.Ordinal)))
            needsUpdate = true;
        if (!string.IsNullOrEmpty(cliTask) && (yamlModel == null || !string.Equals(yamlModel.Task, cliTask, StringComparison.OrdinalIgnoreCase)))
            needsUpdate = true;
        if (!string.IsNullOrEmpty(cliGroupColumn) && (yamlModel == null || !string.Equals(yamlModel.GroupColumn, cliGroupColumn, StringComparison.Ordinal)))
            needsUpdate = true;
        if (!string.IsNullOrEmpty(cliUserColumn) && (yamlModel == null || !string.Equals(yamlModel.UserColumn, cliUserColumn, StringComparison.Ordinal)))
            needsUpdate = true;
        if (!string.IsNullOrEmpty(cliItemColumn) && (yamlModel == null || !string.Equals(yamlModel.ItemColumn, cliItemColumn, StringComparison.Ordinal)))
            needsUpdate = true;

        if (!needsUpdate)
            return;

        var changes = new List<(string Key, string Value)>();
        if (yamlModel != null)
        {
            // Update existing model definition
            if (!string.IsNullOrEmpty(cliLabel))
            {
                yamlModel.Label = cliLabel;
                changes.Add(("label", cliLabel));
            }
            if (!string.IsNullOrEmpty(cliTask))
            {
                yamlModel.Task = cliTask;
                changes.Add(("task", cliTask));
            }
            if (!string.IsNullOrEmpty(cliGroupColumn))
            {
                yamlModel.GroupColumn = cliGroupColumn;
                changes.Add(("group_column", cliGroupColumn));
            }
            if (!string.IsNullOrEmpty(cliUserColumn))
            {
                yamlModel.UserColumn = cliUserColumn;
                changes.Add(("user_column", cliUserColumn));
            }
            if (!string.IsNullOrEmpty(cliItemColumn))
            {
                yamlModel.ItemColumn = cliItemColumn;
                changes.Add(("item_column", cliItemColumn));
            }
        }
        else
        {
            // Create new model definition in yaml
            userConfig.Models[modelName] = new ModelDefinition
            {
                Task = effectiveDefinition.Task,
                Label = effectiveDefinition.Label,
                GroupColumn = effectiveDefinition.GroupColumn,
                UserColumn = effectiveDefinition.UserColumn,
                ItemColumn = effectiveDefinition.ItemColumn,
                Training = effectiveDefinition.Training
            };
        }

        try
        {
            if (!await configLoader.SaveModelAsync(userConfig, modelName, changes))
                WarningConsole.Warn(ConfigLoader.RewrittenWarning);
            var group = string.IsNullOrEmpty(effectiveDefinition.GroupColumn) ? "" : $", group_column={effectiveDefinition.GroupColumn}";
            var users = string.IsNullOrEmpty(effectiveDefinition.UserColumn) ? "" : $", user_column={effectiveDefinition.UserColumn}";
            var items = string.IsNullOrEmpty(effectiveDefinition.ItemColumn) ? "" : $", item_column={effectiveDefinition.ItemColumn}";
            var text = $"Updated mloop.yaml: model '{modelName}' → label={effectiveDefinition.Label}, task={effectiveDefinition.Task}{group}{users}{items}";
            AnsiConsole.MarkupLine($"[grey]{Markup.Escape(text)}[/]");
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLineInterpolated($"[yellow]Warning:[/] Could not update mloop.yaml: {ex.Message}");
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
        var route = new PrepRouter().Route(ctx, prep, TrainingRunner.SupportsPreFeaturizer(task));

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
