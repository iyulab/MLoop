using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.AutoML;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Evaluation;
using MLoop.Core.Hooks;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Complete training engine that orchestrates AutoML and experiment storage
/// </summary>
public class TrainingEngine : ITrainingEngine
{
    private readonly MLContext _mlContext;
    private readonly IDataProvider _dataLoader;
    private readonly IExperimentStore _experimentStore;
    private readonly IFileSystemManager _fileSystem;
    private readonly AutoMLRunner _autoMLRunner;
    private readonly HookEngine? _hookEngine;
    private readonly string? _projectRoot;

    public TrainingEngine(
        IFileSystemManager fileSystem,
        IExperimentStore experimentStore,
        string? projectRoot = null,
        ILogger? logger = null)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
        _experimentStore = experimentStore ?? throw new ArgumentNullException(nameof(experimentStore));
        _projectRoot = projectRoot;

        // Initialize ML.NET components
        _mlContext = new MLContext(seed: 42);
        _dataLoader = new CsvDataLoader(_mlContext);
        _autoMLRunner = new AutoMLRunner(_mlContext, _dataLoader);

        // Initialize HookEngine if project root provided
        if (!string.IsNullOrEmpty(projectRoot) && logger != null)
        {
            _hookEngine = new HookEngine(projectRoot, logger);
        }
    }

    public async Task<TrainingResult> TrainAsync(
        TrainingConfig config,
        IProgress<TrainingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var modelName = config.ModelName;

        // Generate experiment ID for this model
        var experimentId = await _experimentStore.GenerateIdAsync(modelName, cancellationToken);
        var experimentPath = _experimentStore.GetExperimentPath(modelName, experimentId);

        // Track original data file and temp file for cleanup (encoding conversion)
        string originalDataFile = config.DataFile;
        string? tempEncodingFile = null;
        string? stratifiedSplitDirectory = null;

        try
        {
            // Create experiment directory
            await _fileSystem.CreateDirectoryAsync(experimentPath, cancellationToken);

            string dataFilePath = config.DataFile;
            InputSchemaInfo? inputSchema;

            if (DataLoaderFactory.IsDirectoryBased(config.Task))
            {
                // Image classification: config.DataFile is a directory whose subfolders are
                // class labels (folder name = label). CSV preprocessing \u2014 encoding detection,
                // multi-line flattening, CSV quality validation, and column inference \u2014 does
                // not apply to an image directory, so it is bypassed entirely here.
                if (!Directory.Exists(dataFilePath))
                {
                    throw new DirectoryNotFoundException(
                        $"Image classification expects a dataset directory, but it was not found: {dataFilePath}\n" +
                        "Lay images out as one subfolder per class (e.g. datasets/images/OK, datasets/images/NG).");
                }

                // For image classification the class count is the number of class subfolders;
                // it feeds the promotion quality gate's 1/N threshold (BUG-46). Object detection's
                // classes live in annotations, not folders, so leave it unknown there.
                int? classCount = string.Equals(config.Task, "image-classification", StringComparison.OrdinalIgnoreCase)
                    ? ImageDirectoryLoader.CountClasses(dataFilePath)
                    : null;
                inputSchema = BuildDirectoryInputSchema(config.LabelColumn, config.Task, classCount);
            }
            else
            {
                // Handle encoding detection/conversion for non-UTF8 files (e.g., CP949/EUC-KR)
                // This ensures Korean and other non-ASCII text is read correctly throughout training
                var (convertedPath, detection) = EncodingDetector.ConvertToUtf8WithBom(config.DataFile);

                // Use converted path for all training operations, original path for metadata
                if (detection.WasConverted)
                {
                    tempEncodingFile = convertedPath;
                    dataFilePath = convertedPath;
                    Console.WriteLine($"[Info] Converted {detection.EncodingName} \u2192 UTF-8: {Path.GetFileName(originalDataFile)}");
                }

                // Flatten multi-line quoted fields in data rows (RFC 4180 multiline support)
                dataFilePath = CsvDataLoader.FlattenMultiLineQuotedFields(dataFilePath);

                // Flatten multi-line quoted headers (ML.NET doesn't support them)
                dataFilePath = CsvDataLoader.FlattenMultiLineHeaders(dataFilePath);

                // Update config DataFile so AutoMLRunner uses the processed file
                if (dataFilePath != config.DataFile)
                {
                    // `with` rather than a re-listed constructor: the previous hand-written copy
                    // omitted TestDataFile and the pre-featurizer fields, so any dataset needing an
                    // encoding conversion silently lost its pre-split test set and prep featurizer.
                    config = config with { DataFile = dataFilePath };
                }

                // Validate data quality before training (label column + dataset size)
                var dataQualityValidator = new DataQualityValidator(_mlContext);
                var qualityResult = dataQualityValidator.ValidateTrainingData(dataFilePath, config.LabelColumn, config.Task);

                if (!qualityResult.IsValid)
                {
                    // Data quality issue detected - fail fast with clear error
                    throw new InvalidOperationException(
                        $"{qualityResult.ErrorMessage}\n" +
                        $"{qualityResult.ErrorMessageEn ?? ""}\n" +
                        $"{string.Join("\n", qualityResult.Suggestions)}");
                }

                // Show warnings if any
                foreach (var warning in qualityResult.Warnings)
                {
                    Console.WriteLine($"[Warning] {warning}");
                }

                // Show suggestions if any
                if (qualityResult.Suggestions.Any())
                {
                    Console.WriteLine("[Suggestions]");
                    foreach (var suggestion in qualityResult.Suggestions)
                    {
                        Console.WriteLine($"  {suggestion}");
                    }
                    Console.WriteLine();
                }

                // Decide once, here, which columns featurization drops. Everything downstream — the
                // saved schema below, the train partition, the test partition — consumes this one
                // decision instead of re-deriving it from whatever slice it happens to hold. Deciding
                // per slice is unsound because the rules are data-dependent: a column that is
                // constant only inside one partition narrows that partition alone, and the pipeline
                // fitted on one width then fails on the other ("Schema mismatch for feature column
                // 'Features': expected Vector<Single, 30>, got Vector<Single, 29>").
                //
                // The full dataset is the deciding slice, matching the schema capture below: a column
                // is dropped because it carries no signal in the data as a whole, not because one
                // random partition happened to flatten it.
                var featureExclusions = CsvDataLoader.DetermineExcludedColumns(dataFilePath, config.LabelColumn);
                config = config with { FeatureExclusions = featureExclusions.Select(c => c.Name).ToList() };

                // Capture input schema before training (using enhanced detection).
                // Deliberately from the full dataset, not the train split — the schema must describe
                // every category the model may be asked to predict on.
                inputSchema = CaptureInputSchemaEnhanced(dataFilePath, config.LabelColumn, config.Task, config.ColumnOverrides, featureExclusions);

                // Stratify the train/test split for classification. The unstratified alternative is
                // ML.NET's TrainTestSplit (DataProviderBase.SplitData), which draws rows uniformly and
                // therefore leaves a rare class out of the test partition entirely with high probability
                // — 3 positives in 357 rows lands a positive-free test set about half the time, and every
                // classification metric that needs both classes is then undefined, which is what the
                // AUC-fallback chain was built to paper over.
                //
                // Note ML.NET's samplingKeyColumnName is NOT stratification: it is grouping (keeping equal
                // keys on one side to prevent leakage), so passing the label there would guarantee the
                // failure instead of preventing it. The split has to be done on the rows themselves.
                if (RequiresStratifiedSplit(config))
                {
                    stratifiedSplitDirectory = Path.Combine(
                        Path.GetTempPath(), "mloop-split-" + Guid.NewGuid().ToString("N"));

                    var split = new CsvSplitter().StratifiedSplit(
                        dataFilePath, config.LabelColumn, config.TestSplit,
                        outputDirectory: stratifiedSplitDirectory);

                    // TestSplit stays on the config for provenance (the experiment records the fraction
                    // that was actually requested); AutoMLRunner switches to the pre-split path on the
                    // presence of TestDataFile alone.
                    config = config with { DataFile = split.TrainFile, TestDataFile = split.TestFile };

                    Console.WriteLine(
                        $"[Info] Stratified split: train={split.TrainRows}, test={split.TestRows} " +
                        $"({string.Join(", ", split.PerClass.OrderBy(c => c.Value.Train + c.Value.Test)
                            .Select(c => $"'{c.Key}' {c.Value.Train}/{c.Value.Test}"))} as train/test)");
                }
            }

            // Execute PreTrain hooks
            if (_hookEngine != null && _hookEngine.HasHooks(HookType.PreTrain))
            {
                // Load training data for hook context. Image classification reads a
                // directory of class subfolders; tabular tasks read the CSV file.
                IDataView trainData;
                if (DataLoaderFactory.IsDirectoryBased(config.Task))
                {
                    trainData = DataLoaderFactory
                        .Create(config.Task, _mlContext)
                        .LoadData(dataFilePath, config.LabelColumn, config.Task);
                }
                else
                {
                    var loader = _mlContext.Data.CreateTextLoader(
                        new[] { new TextLoader.Column("Features", DataKind.Single, 0, int.MaxValue) },
                        separatorChar: ',',
                        hasHeader: true);
                    trainData = loader.Load(dataFilePath);
                }

                var hookContext = new HookContext
                {
                    HookType = HookType.PreTrain,
                    HookName = string.Empty,  // Will be set by HookEngine
                    MLContext = _mlContext,
                    DataView = trainData,
                    Model = null,
                    ExperimentResult = null,
                    Metrics = null,
                    ProjectRoot = _projectRoot!,
                    Logger = new TrainingEngineLogger(),
                    Metadata = new Dictionary<string, object>
                    {
                        ["LabelColumn"] = config.LabelColumn,
                        ["TaskType"] = config.Task,
                        ["ModelName"] = modelName,
                        ["ExperimentId"] = experimentId,
                        ["TimeLimit"] = config.TimeLimitSeconds
                    }
                };

                var hooksPassed = await _hookEngine.ExecuteHooksAsync(HookType.PreTrain, hookContext);
                if (!hooksPassed)
                {
                    throw new InvalidOperationException("Training aborted by PreTrain hook");
                }
            }

            // Run AutoML (with optional auto-time two-phase training).
            // Auto-time probing samples CSV rows, which does not apply to image
            // directories — image classification always uses the single-pass path.
            AutoMLResult autoMLResult;

            if (config.UseAutoTime && !DataLoaderFactory.IsDirectoryBased(config.Task))
            {
                autoMLResult = await RunAutoTimeTrainingAsync(config, dataFilePath, progress, cancellationToken);
            }
            else
            {
                // Fixed-budget runs get the same start boundary auto-time announces via its probe
                // phases: without it a consumer capturing the event stream sees nothing between the
                // pre-training summary and the first completed trial.
                progress?.Report(new TrainingProgress
                {
                    TrialNumber = 0, TrainerName = "", Metric = 0, MetricName = "", ElapsedSeconds = 0,
                    Phase = TrainingPhase.MainStart,
                    FinalTimeSeconds = config.TimeLimitSeconds
                });

                autoMLResult = await _autoMLRunner.RunAsync(config, progress, cancellationToken);
            }

            stopwatch.Stop();

            // The uniform end-of-training-window marker, on every path (fixed budget, auto-time
            // main, auto-time converged): post-training steps — save, evaluate, promote — still run
            // after this, so a consumer can tell "still training" from "finalizing".
            progress?.Report(new TrainingProgress
            {
                TrialNumber = autoMLResult.Trials.Count, TrainerName = "", Metric = 0, MetricName = "",
                ElapsedSeconds = stopwatch.Elapsed.TotalSeconds,
                Phase = TrainingPhase.Complete
            });

            // Save model
            var modelPath = _fileSystem.CombinePath(experimentPath, ExperimentLayout.ModelFileName);
            _mlContext.Model.Save(autoMLResult.Model, null, modelPath);

            // ② regression wave (heteroscedastic): persist the optional σ(x) model beside the main one
            // so predict/serve can widen the conformal band per row. Absent for non-regression tasks
            // and homoscedastic fallbacks — predict then uses the constant-width band.
            if (autoMLResult.ResidualModel != null)
            {
                var residualPath = _fileSystem.CombinePath(experimentPath, ExperimentLayout.ResidualModelFileName);
                _mlContext.Model.Save(autoMLResult.ResidualModel, null, residualPath);
            }

            // Execute PostTrain hooks
            if (_hookEngine != null && _hookEngine.HasHooks(HookType.PostTrain))
            {
                var hookContext = new HookContext
                {
                    HookType = HookType.PostTrain,
                    HookName = string.Empty,  // Will be set by HookEngine
                    MLContext = _mlContext,
                    DataView = null,
                    Model = autoMLResult.Model,
                    ExperimentResult = autoMLResult,
                    Metrics = autoMLResult.Metrics,
                    ProjectRoot = _projectRoot!,
                    Logger = new TrainingEngineLogger(),
                    Metadata = new Dictionary<string, object>
                    {
                        ["LabelColumn"] = config.LabelColumn,
                        ["TaskType"] = config.Task,
                        ["ModelName"] = modelName,
                        ["ExperimentId"] = experimentId,
                        ["TimeLimit"] = config.TimeLimitSeconds,
                        ["BestTrainer"] = autoMLResult.BestTrainer,
                        ["ModelPath"] = modelPath
                    }
                };

                var hooksPassed = await _hookEngine.ExecuteHooksAsync(HookType.PostTrain, hookContext);
                if (!hooksPassed)
                {
                    // PostTrain hooks can't abort (model already trained), just log warning
                    Console.WriteLine("[Warning] PostTrain hook requested abort, but model is already trained");
                }
            }

            // Prepare experiment data
            var experimentData = new ExperimentData
            {
                ModelName = modelName,
                ExperimentId = experimentId,
                Timestamp = DateTime.UtcNow,
                Status = "completed",
                Task = config.Task,
                Config = new ExperimentConfig
                {
                    DataFile = originalDataFile, // Store original path, not converted temp file
                    LabelColumn = config.LabelColumn,
                    TimeLimitSeconds = config.TimeLimitSeconds,
                    Metric = config.Metric,
                    TestSplit = config.TestSplit,
                    InputSchema = inputSchema,
                    GroupColumn = config.GroupColumn,
                    UserColumn = config.UserColumn,
                    ItemColumn = config.ItemColumn
                },
                Result = new ExperimentResult
                {
                    BestTrainer = autoMLResult.BestTrainer,
                    TrainingTimeSeconds = stopwatch.Elapsed.TotalSeconds
                },
                Metrics = autoMLResult.Metrics,
                Trials = autoMLResult.Trials,
                RankingMetric = autoMLResult.RankingMetric
            };

            // Save experiment metadata
            await _experimentStore.SaveAsync(modelName, experimentData, cancellationToken);

            return new TrainingResult
            {
                ExperimentId = experimentId,
                BestTrainer = autoMLResult.BestTrainer,
                Metrics = autoMLResult.Metrics,
                TrainingTimeSeconds = stopwatch.Elapsed.TotalSeconds,
                ModelPath = modelPath,
                Schema = inputSchema ?? autoMLResult.Schema,
                RowCount = autoMLResult.RowCount
            };
        }
        catch (Exception ex)
        {
            stopwatch.Stop();

            // Save failed experiment
            var experimentData = new ExperimentData
            {
                ModelName = modelName,
                ExperimentId = experimentId,
                Timestamp = DateTime.UtcNow,
                Status = "failed",
                Task = config.Task,
                Config = new ExperimentConfig
                {
                    DataFile = originalDataFile, // Store original path, not converted temp file
                    LabelColumn = config.LabelColumn,
                    TimeLimitSeconds = config.TimeLimitSeconds,
                    Metric = config.Metric,
                    TestSplit = config.TestSplit,
                    GroupColumn = config.GroupColumn,
                    UserColumn = config.UserColumn,
                    ItemColumn = config.ItemColumn
                },
                Result = new ExperimentResult
                {
                    BestTrainer = "none",
                    TrainingTimeSeconds = stopwatch.Elapsed.TotalSeconds
                }
            };

            await _experimentStore.SaveAsync(modelName, experimentData, cancellationToken);

            throw new InvalidOperationException(
                $"Training failed for experiment {modelName}/{experimentId}: {ex.Message}",
                ex);
        }
        finally
        {
            // Clean up temp file from encoding conversion
            // ML.NET may have lazily loaded data, but by now training is complete
            if (tempEncodingFile != null && File.Exists(tempEncodingFile))
            {
                try { File.Delete(tempEncodingFile); } catch (IOException) { /* Ignore cleanup errors */ }
            }

            // The stratified split writes to a temp directory rather than datasets/, so routine
            // training leaves no artifacts behind for the user to wonder about or accidentally commit.
            if (stratifiedSplitDirectory != null && Directory.Exists(stratifiedSplitDirectory))
            {
                try { Directory.Delete(stratifiedSplitDirectory, recursive: true); }
                catch (IOException) { /* Ignore cleanup errors */ }
                catch (UnauthorizedAccessException) { /* Ignore cleanup errors */ }
            }
        }
    }

    /// <summary>
    /// Whether this run should replace ML.NET's uniform random split with a stratified one.
    /// </summary>
    /// <remarks>
    /// Only classification cares about class proportions. A caller that already supplied a test file
    /// (the <c>--balance</c> flow, which stratify-splits upstream before oversampling) is left alone —
    /// splitting again would discard its work. <c>TestSplit == 0</c> means the caller opted out of a
    /// holdout entirely and AutoML validates internally.
    /// </remarks>
    private static bool RequiresStratifiedSplit(TrainingConfig config) =>
        config.TestSplit > 0
        && string.IsNullOrEmpty(config.TestDataFile)
        && !string.IsNullOrEmpty(config.LabelColumn)
        && !DataLoaderFactory.IsDirectoryBased(config.Task)
        && config.Task.ToLowerInvariant() is "binary-classification" or "multiclass-classification";

    /// <summary>
    /// Two-phase auto-time training: static estimate -> probe run -> reactive estimate -> main run
    /// </summary>
    private async Task<AutoMLResult> RunAutoTimeTrainingAsync(
        TrainingConfig config,
        string dataFilePath,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        // Collect data statistics for estimation
        var (rowCount, columnCount, hasTextFeatures, classCount) = CollectDataStats(dataFilePath, config.LabelColumn, config.Task);

        var staticEstimate = TimeEstimator.EstimateStatic(
            rowCount, columnCount, config.Task, classCount, hasTextFeatures);

        var probeTime = TimeEstimator.GetProbeTime(staticEstimate);

        // Phase 1: Quick Probe
        // `with` rather than a re-listed constructor: the earlier hand-written copies omitted
        // TestDataFile and the pre-featurizer fields, so auto-time (the default when --time is not
        // given) silently discarded any pre-split test set and the prep pre-featurizer.
        var probeConfig = config with { TimeLimitSeconds = probeTime, UseAutoTime = false };

        var probeTrialCount = 0;
        progress?.Report(new TrainingProgress
        {
            TrialNumber = 0, TrainerName = "", Metric = 0, MetricName = "", ElapsedSeconds = 0,
            Phase = TrainingPhase.ProbeStart,
            ProbeTimeSeconds = probeTime
        });

        // Interlocked, not ++: AutoML reports trials from its own worker threads, so the callback
        // is not serialized. (Before per-trial reporting was wired up this counter never moved at
        // all on tabular tasks, which is why the race was invisible.)
        var probeProgress = progress != null
            ? new Progress<TrainingProgress>(p =>
            {
                Interlocked.Increment(ref probeTrialCount);
                progress.Report(p);
            })
            : null;

        var probeAutoMLResult = await _autoMLRunner.RunAsync(probeConfig, probeProgress, cancellationToken);

        // Determine best metric from probe
        var bestMetric = GetPrimaryMetricValue(probeAutoMLResult.Metrics, config.Metric, config.Task);

        var probe = new ProbeResult
        {
            BestMetric = bestMetric,
            ProbeTimeSeconds = probeTime,
            TrialsCompleted = probeTrialCount
        };

        var finalTime = TimeEstimator.EstimateReactive(probe, staticEstimate);

        // Phase 2: Main Training (only if not already converged)
        if (bestMetric > 0.95)
        {
            progress?.Report(new TrainingProgress
            {
                TrialNumber = probeTrialCount, TrainerName = "", Metric = bestMetric, MetricName = "", ElapsedSeconds = 0,
                Phase = TrainingPhase.ProbeConverged,
                ProbeTimeSeconds = probeTime
            });
            return probeAutoMLResult;
        }

        progress?.Report(new TrainingProgress
        {
            TrialNumber = probeTrialCount, TrainerName = "", Metric = bestMetric, MetricName = "", ElapsedSeconds = 0,
            Phase = TrainingPhase.ProbeComplete,
            ProbeTimeSeconds = probeTime,
            FinalTimeSeconds = finalTime
        });

        var mainConfig = config with { TimeLimitSeconds = finalTime, UseAutoTime = false };

        var mainResult = await _autoMLRunner.RunAsync(mainConfig, progress, cancellationToken);

        // Pick best between probe and main
        var mainMetric = GetPrimaryMetricValue(mainResult.Metrics, config.Metric, config.Task);
        if (mainMetric >= bestMetric)
        {
            return mainResult;
        }

        return probeAutoMLResult;
    }

    /// <summary>
    /// Collects basic data statistics from CSV for time estimation
    /// </summary>
    internal static (int rowCount, int columnCount, bool hasTextFeatures, int classCount) CollectDataStats(
        string dataFilePath, string labelColumn, string task)
    {
        int rowCount = 0;
        int columnCount = 0;
        bool hasTextFeatures = false;
        var labelValues = new HashSet<string>();

        using var reader = new StreamReader(dataFilePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);

        var header = reader.ReadLine();
        if (string.IsNullOrEmpty(header))
            return (0, 0, false, 0);

        var columnNames = CsvFieldParser.ParseFields(header);
        columnCount = columnNames.Length;
        var labelIndex = Array.FindIndex(columnNames, c =>
            c.Equals(labelColumn, StringComparison.OrdinalIgnoreCase));

        // D2: class counting scans the FULL file — a sorted minority class appearing only past
        // row 1000 must not be missed, or a binary/multiclass label is misdetected as single-class
        // (Label=["OK"]), which corrupts time estimation and the promotion quality gate's 1/N
        // threshold. Text-feature detection stays sampled to the first 1000 rows: it is a heuristic
        // and the per-field scan is expensive.
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            rowCount++;

            var fields = CsvFieldParser.ParseFields(line);

            // Collect label values for class counting (full-file scan)
            if (labelIndex >= 0 && labelIndex < fields.Length)
            {
                var value = fields[labelIndex].Trim();
                if (!string.IsNullOrEmpty(value))
                    labelValues.Add(value);
            }

            // Check for text features (any field with spaces likely text) — sampled heuristic
            if (!hasTextFeatures && rowCount <= 1000)
            {
                for (int i = 0; i < fields.Length; i++)
                {
                    if (i == labelIndex) continue;
                    var val = fields[i].Trim();
                    if (val.Contains(' ') && val.Length > 20 && !double.TryParse(val, out _))
                    {
                        hasTextFeatures = true;
                        break;
                    }
                }
            }
        }

        var isClassification = task.ToLowerInvariant() switch
        {
            "binary-classification" => true,
            "multiclass-classification" => true,
            _ => false
        };

        int classCount = isClassification ? labelValues.Count : 0;

        return (rowCount, columnCount, hasTextFeatures, classCount);
    }

    /// <summary>
    /// Gets the primary metric value from a metrics dictionary based on task type. Delegates to
    /// <see cref="TaskMetadata.ResolvePrimaryMetricValue"/> (the shared metric-value resolver) so
    /// the probe/main comparison here and the experiment index's BestMetric never diverge; an empty
    /// dictionary yields 0 to preserve this method's non-nullable contract.
    /// </summary>
    internal static double GetPrimaryMetricValue(Dictionary<string, double> metrics, string metricName, string task)
        => TaskMetadata.ResolvePrimaryMetricValue(metrics, metricName, task) ?? 0;

    /// <summary>
    /// Captures input schema with enhanced type detection using actual data values
    /// </summary>
    /// <summary>
    /// Builds the input schema for a directory-based model. At predict time the model
    /// consumes an image file path (string), so the schema advertises an ImagePath feature
    /// column plus the task's target columns: image classification has a single label column;
    /// object detection has a label vector (class names) plus a bounding-box vector.
    /// </summary>
    internal static InputSchemaInfo BuildDirectoryInputSchema(string labelColumn, string? task, int? classCount = null)
    {
        var label = string.IsNullOrWhiteSpace(labelColumn)
            ? ImageDirectoryLoader.DefaultLabelColumn
            : labelColumn;

        // Use MLoop's canonical dataType vocabulary (Numeric/Categorical/Text/Boolean), not the raw
        // .NET type name "String" — predict-time consumers map these to ML.NET DataKinds and a
        // classification label MUST resolve to String so the model's MapValueToKey accepts it.
        // ImagePath is a text feature; the class label is categorical. (BUG-42)
        var columns = new List<ColumnSchema>
        {
            new ColumnSchema
            {
                Name = ImageDirectoryLoader.ImagePathColumn,
                DataType = SchemaDataTypes.Text,
                Purpose = "Feature"
            },
            new ColumnSchema
            {
                Name = label,
                DataType = SchemaDataTypes.Categorical,
                Purpose = "Label",
                // Class count (folder count) feeds the promotion quality gate's 1/N threshold
                // for image classification. Null when unknown (e.g. object detection, whose
                // classes come from annotations rather than subfolders). (BUG-46)
                UniqueValueCount = classCount
            }
        };

        // Object detection additionally carries one bounding box (x0 y0 x1 y1) per labeled
        // object, so the label is a vector of class names alongside a float-vector of boxes.
        if (string.Equals(task, "object-detection", StringComparison.OrdinalIgnoreCase))
        {
            columns.Add(new ColumnSchema
            {
                Name = CocoDataLoader.BoundingBoxColumn,
                DataType = "Single",
                Purpose = "Label"
            });
        }

        return new InputSchemaInfo
        {
            CapturedAt = DateTime.UtcNow,
            Columns = columns
        };
    }

    private InputSchemaInfo? CaptureInputSchemaEnhanced(string dataFile, string labelColumn, string? taskType = null, Dictionary<string, string>? columnOverrides = null,
        IReadOnlyList<ExcludedColumn>? featureExclusions = null)
    {
        try
        {
            // Get all columns from the file with UTF-8 encoding
            string? firstLine;
            using (var reader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine();
            }

            if (string.IsNullOrEmpty(firstLine))
            {
                return null;
            }

            var columnNames = CsvFieldParser.ParseFields(firstLine);

            // Read sample of data lines for type inference (not full file -- saves memory)
            var sampleLines = new List<string>();
            using (var sampleReader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                sampleReader.ReadLine(); // skip header
                string? sampleLine;
                while (sampleLines.Count < 1000 && (sampleLine = sampleReader.ReadLine()) != null)
                {
                    sampleLines.Add(sampleLine);
                }
            }
            var dataLines = sampleLines.ToArray();

            // Try ML.NET InferColumns first
            Microsoft.ML.AutoML.ColumnInformation? columnInfo = null;
            DataKind? labelInferredKind = null;
            try
            {
                var columnInference = _mlContext.Auto().InferColumns(
                    dataFile,
                    labelColumnName: labelColumn,
                    separatorChar: ',');
                columnInfo = columnInference?.ColumnInformation;

                // BUG-15: Track label column's inferred DataKind.
                // CsvDataLoader converts Boolean labels -> String for MapValueToKey compatibility.
                // Schema must reflect this so PredictionEngine uses the correct type.
                if (columnInference?.TextLoaderOptions?.Columns != null)
                {
                    foreach (var col in columnInference.TextLoaderOptions.Columns)
                    {
                        if (col.Name != null &&
                            col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase))
                        {
                            labelInferredKind = col.DataKind;
                            break;
                        }
                    }
                }
            }
            catch
            {
                // InferColumns may fail for complex data
            }

            var columns = new List<ColumnSchema>();

            foreach (var colName in columnNames)
            {
                var colIndex = Array.IndexOf(columnNames, colName);
                var purpose = GetColumnPurposeEnhanced(colName, labelColumn, columnInfo);
                var (dataType, categoricalValues, uniqueCount) = InferColumnTypeFromData(
                    colName, colIndex, dataLines, columnInfo);

                // BUG-15: If label column was inferred as Boolean by InferColumns,
                // CsvDataLoader converts it to String (for MapValueToKey compatibility).
                // Record as "Categorical" so PredictionEngine overrides to String type.
                // BUG-23: Skip this for regression — Boolean labels become Single (numeric),
                // so schema should remain "Numeric" for regression tasks.
                var isRegressionCapture = string.Equals(taskType, "regression", StringComparison.OrdinalIgnoreCase)
                                       || string.Equals(taskType, "Regression", StringComparison.OrdinalIgnoreCase);
                if (purpose == "Label" && labelInferredKind == DataKind.Boolean && dataType == SchemaDataTypes.Numeric
                    && !isRegressionCapture)
                {
                    dataType = SchemaDataTypes.Categorical;
                }

                // BUG-25: If InferColumns classified a text column as Ignored,
                // but InferColumnTypeFromData detected it as Text, override purpose to Feature.
                // This aligns schema metadata with AutoMLRunner's BuildColumnInformation behavior.
                if (purpose == "Ignore" && dataType == SchemaDataTypes.Text)
                {
                    purpose = "Feature";
                    Console.WriteLine($"[Info] Column '{colName}' reclassified: Ignored → Text Feature");
                }

                // Apply column type overrides from mloop.yaml
                if (columnOverrides != null &&
                    columnOverrides.TryGetValue(colName, out var overrideType))
                {
                    var normalizedType = overrideType.ToLowerInvariant();
                    var originalType = dataType;
                    var originalPurpose = purpose;

                    switch (normalizedType)
                    {
                        case "text":
                            dataType = SchemaDataTypes.Text;
                            if (purpose != "Label") purpose = "Feature";
                            break;
                        case "categorical":
                            dataType = SchemaDataTypes.Categorical;
                            if (purpose != "Label") purpose = "Feature";
                            break;
                        case "numeric":
                            dataType = SchemaDataTypes.Numeric;
                            if (purpose != "Label") purpose = "Feature";
                            break;
                        case "ignore":
                            purpose = "Exclude";
                            break;
                    }

                    if (dataType != originalType || purpose != originalPurpose)
                    {
                        Console.WriteLine($"[Info] Column '{colName}' overridden: {originalType}/{originalPurpose} → {dataType}/{purpose} (mloop.yaml)");
                    }
                }

                columns.Add(new ColumnSchema
                {
                    Name = colName,
                    DataType = dataType,
                    Purpose = purpose,
                    CategoricalValues = categoricalValues,
                    UniqueValueCount = uniqueCount
                });
            }

            // Mark the columns featurization drops (DateTime / sparse / constant) as "Exclude", so
            // the saved schema describes exactly the feature set the model was fitted on — predict
            // and evaluate replay it through CsvDataLoader.RemoveExcludedColumns.
            //
            // The set is not re-derived here. It is the same decision the loader applies, taken once
            // by CsvDataLoader.DetermineExcludedColumns: this method used to mirror the loader's
            // constant/sparse rules against its own 200-row sample, which is a second implementation
            // of a data-dependent rule and therefore a drift waiting to happen (CLAUDE.md
            // "Single-Source Authorities").
            MarkColumnsAsExcluded(columns, featureExclusions);

            // BUG-R2-06: Collect complete categorical values from the FULL file.
            // The 1000-row sample above is only for type inference heuristics.
            // For ordered data, the sample may miss categorical values that appear later.
            CollectCompleteCategoricalValues(dataFile, columns, columnNames);

            return new InputSchemaInfo
            {
                Columns = columns,
                CapturedAt = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Warning] Failed to capture input schema: {ex.Message}");
            return null;
        }
    }

    private string GetColumnPurposeEnhanced(
        string columnName,
        string labelColumn,
        Microsoft.ML.AutoML.ColumnInformation? columnInfo)
    {
        // Exact match for label
        if (columnName == labelColumn)
            return "Label";

        // Check ML.NET inference
        if (columnInfo != null)
        {
            if (columnInfo.LabelColumnName == columnName)
                return "Label";
            if (columnInfo.IgnoredColumnNames?.Contains(columnName) == true)
                return "Ignore";
        }

        // Default: treat as Feature
        return "Feature";
    }

    private (string DataType, List<string>? CategoricalValues, int? UniqueCount) InferColumnTypeFromData(
        string columnName,
        int colIndex,
        string[] dataLines,
        Microsoft.ML.AutoML.ColumnInformation? columnInfo)
    {
        // First check ML.NET inference
        if (columnInfo != null)
        {
            if (columnInfo.CategoricalColumnNames?.Contains(columnName) == true)
            {
                var (values, count) = CollectCategoricalValues(colIndex, dataLines);
                if (count > 0 && dataLines.Length > 0 && LooksLikeText(colIndex, dataLines, count))
                {
                    Console.WriteLine($"[Info] Column '{columnName}' reclassified: Categorical → Text (text-like content: {count} unique values in {dataLines.Length} rows)");
                    return (SchemaDataTypes.Text, null, count);
                }
                return (SchemaDataTypes.Categorical, values, count);
            }
            if (columnInfo.NumericColumnNames?.Contains(columnName) == true)
                return (SchemaDataTypes.Numeric, null, null);
            if (columnInfo.TextColumnNames?.Contains(columnName) == true)
                return (SchemaDataTypes.Text, null, null);
        }

        // Fallback: Infer type from actual values
        if (colIndex < 0 || dataLines.Length == 0)
            return ("Unknown", null, null);

        var uniqueValues = new HashSet<string>();
        int numericCount = 0;
        int totalCount = 0;

        foreach (var line in dataLines)
        {
            var values = CsvFieldParser.ParseFields(line);
            if (colIndex >= values.Length)
                continue;

            var value = values[colIndex].Trim();
            if (string.IsNullOrEmpty(value))
                continue;

            totalCount++;
            uniqueValues.Add(value);

            // Check if value is numeric
            if (double.TryParse(value, out _))
            {
                numericCount++;
            }
        }

        if (totalCount == 0)
            return ("Unknown", null, null);

        // Determine type based on data analysis
        double numericRatio = (double)numericCount / totalCount;
        int uniqueCount = uniqueValues.Count;

        // High numeric ratio (>90%) = Numeric column
        if (numericRatio > 0.9)
        {
            return (SchemaDataTypes.Numeric, null, null);
        }

        // Low unique count relative to total = likely Categorical
        // But text-like content should be treated as Text even with low unique ratio
        if (uniqueCount <= 100 || (numericRatio < 0.5 && uniqueCount < totalCount * 0.1))
        {
            if (totalCount > 0 && LooksLikeText(colIndex, dataLines, uniqueCount))
            {
                return (SchemaDataTypes.Text, null, uniqueCount);
            }
            var categoricalValues = uniqueValues.OrderBy(v => v).ToList();
            return (SchemaDataTypes.Categorical, categoricalValues, uniqueCount);
        }

        // High cardinality text
        return (SchemaDataTypes.Text, null, uniqueCount);
    }

    private (List<string>? Values, int Count) CollectCategoricalValues(int colIndex, string[] dataLines)
    {
        var uniqueValues = new HashSet<string>();

        foreach (var line in dataLines)
        {
            var values = CsvFieldParser.ParseFields(line);
            if (colIndex < values.Length)
            {
                var value = values[colIndex].Trim();
                if (!string.IsNullOrEmpty(value))
                {
                    uniqueValues.Add(value);
                }
            }
        }

        return (uniqueValues.OrderBy(v => v).ToList(), uniqueValues.Count);
    }

    /// <summary>
    /// Determines whether a column's values look like natural text rather than categorical codes.
    /// Uses multiple heuristics: unique ratio, average token count, and average string length.
    /// This prevents log messages, descriptions, and other free-text from being treated as
    /// categorical features (which would use OneHotEncoding instead of FeaturizeText).
    /// </summary>
    internal static bool LooksLikeText(int colIndex, string[] dataLines, int uniqueCount)
    {
        if (dataLines.Length == 0)
            return false;

        // Criterion 1: High unique ratio (original heuristic)
        double uniqueRatio = (double)uniqueCount / dataLines.Length;
        if (uniqueRatio > 0.5)
            return true;

        // Sample values for text-likeness analysis
        int sampleSize = Math.Min(dataLines.Length, 200);
        int totalTokens = 0;
        int totalLength = 0;
        int validCount = 0;

        for (int i = 0; i < sampleSize; i++)
        {
            var fields = CsvFieldParser.ParseFields(dataLines[i]);
            if (colIndex >= fields.Length)
                continue;

            var value = fields[colIndex].Trim();
            if (string.IsNullOrEmpty(value))
                continue;

            validCount++;
            totalLength += value.Length;
            totalTokens += value.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
        }

        if (validCount == 0)
            return false;

        double avgTokens = (double)totalTokens / validCount;
        double avgLength = (double)totalLength / validCount;

        // Criterion 2: Average 3+ tokens (words) per value → natural language text
        if (avgTokens >= 3)
            return true;

        // Criterion 3: Average length > 30 chars → long strings, likely text
        if (avgLength > 30)
            return true;

        // Criterion 4: High cardinality (200+) with moderate unique ratio (10%+)
        if (uniqueCount > 200 && uniqueRatio > 0.1)
            return true;

        return false;
    }

    /// <summary>
    /// Streams through the entire file to collect all unique categorical values.
    /// The initial type inference uses a 1000-row sample, but categorical values
    /// <summary>
    /// Applies the run's single featurization-exclusion decision to the captured schema.
    /// </summary>
    /// <remarks>
    /// Columns already excluded for another reason (a "ignore" column override, or the label) keep
    /// their existing purpose — the exclusion set only ever adds exclusions.
    /// </remarks>
    private static void MarkColumnsAsExcluded(
        List<ColumnSchema> columns, IReadOnlyList<ExcludedColumn>? featureExclusions)
    {
        if (featureExclusions is null || featureExclusions.Count == 0) return;

        foreach (var exclusion in featureExclusions)
        {
            var index = columns.FindIndex(c => c.Name.Equals(exclusion.Name, StringComparison.OrdinalIgnoreCase));
            if (index < 0) continue;

            var col = columns[index];
            if (col.Purpose is "Label" or "Exclude") continue;

            columns[index] = new ColumnSchema
            {
                Name = col.Name,
                DataType = exclusion.Reason,
                Purpose = "Exclude",
                CategoricalValues = null,
                UniqueValueCount = exclusion.Reason == SchemaDataTypes.ExcludedConstant ? 1 : null
            };
        }
    }

    /// <summary>
    /// Collects the complete set of categorical values for every categorical column. The set
    /// must be complete to prevent predict-time failures when unseen categories appear.
    /// Memory-efficient: only stores unique values per column, not all rows.
    /// </summary>
    internal static void CollectCompleteCategoricalValues(
        string dataFile, List<ColumnSchema> columns, string[] columnNames)
    {
        // Find categorical columns that need complete value collection — both Features and the
        // Label. D2: the label is structurally critical (the promotion quality gate derives its
        // 1/N threshold from the label's UniqueValueCount), so a head-sampled single-class label
        // (Label=["OK"]) would corrupt the gate. Regression labels are Numeric, not Categorical,
        // so they are naturally excluded by the DataType check.
        var catColumns = new Dictionary<int, ColumnSchema>();
        foreach (var col in columns)
        {
            if (col.DataType == SchemaDataTypes.Categorical && (col.Purpose == "Feature" || col.Purpose == "Label"))
            {
                var idx = Array.IndexOf(columnNames, col.Name);
                if (idx >= 0) catColumns[idx] = col;
            }
        }

        if (catColumns.Count == 0) return;

        // Stream through file collecting unique values (no full-file memory load)
        var uniqueSets = catColumns.ToDictionary(kv => kv.Key, _ => new HashSet<string>());

        using var reader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
        reader.ReadLine(); // skip header

        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            var fields = CsvFieldParser.ParseFields(line);
            foreach (var (colIndex, _) in catColumns)
            {
                if (colIndex < fields.Length)
                {
                    var value = fields[colIndex].Trim();
                    if (!string.IsNullOrEmpty(value))
                        uniqueSets[colIndex].Add(value);
                }
            }
        }

        // Update column schemas with complete categorical values
        // ColumnSchema uses init-only properties, so replace items in the list
        foreach (var (colIndex, col) in catColumns)
        {
            var unique = uniqueSets[colIndex];
            var listIndex = columns.IndexOf(col);
            if (listIndex >= 0)
            {
                columns[listIndex] = new ColumnSchema
                {
                    Name = col.Name,
                    DataType = col.DataType,
                    Purpose = col.Purpose,
                    CategoricalValues = unique.OrderBy(v => v).ToList(),
                    UniqueValueCount = unique.Count
                };
            }
        }
    }

    private class TrainingEngineLogger : ILogger
    {
        public void Debug(string message) { } // Silent during training
        public void Info(string message) => Console.WriteLine(message);
        public void Warning(string message) => Console.WriteLine($"[Warning] {message}");
        public void Error(string message) => Console.WriteLine($"[Error] {message}");
        public void Error(string message, Exception exception)
        {
            Console.WriteLine($"[Error] {message}");
            Console.WriteLine($"  {exception.Message}");
        }
    }
}
