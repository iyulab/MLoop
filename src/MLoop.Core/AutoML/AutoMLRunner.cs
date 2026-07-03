using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Scripting;
using MLoop.Extensibility;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.AutoML;

/// <summary>
/// AutoML runner implementation using ML.NET AutoML with extensibility support
/// </summary>
public class AutoMLRunner
{
    private readonly MLContext _mlContext;
    private readonly IDataProvider _dataLoader;
    private readonly string? _projectRoot;
    private readonly ScriptDiscovery _scriptDiscovery;
    private readonly ConsoleLogger _logger;

    public AutoMLRunner(MLContext mlContext, IDataProvider dataLoader, string? projectRoot = null)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _dataLoader = dataLoader ?? throw new ArgumentNullException(nameof(dataLoader));
        _projectRoot = projectRoot ?? Directory.GetCurrentDirectory();
        _scriptDiscovery = new ScriptDiscovery(_projectRoot);
        _logger = new ConsoleLogger();
    }

    /// <summary>
    /// Single source of truth for which task types consume <c>config.PreFeaturizer</c>.
    /// Only binary/multiclass/regression pass <c>preFeaturizer:</c> to their <c>experiment.Execute</c>
    /// call (see the three Execute sites in this file). All other tasks build their own pipelines
    /// and ignore the preFeaturizer, so routing statistical prep there would silently drop it.
    /// Used by prep routing (<see cref="Preprocessing.PrepRouter"/>) to decide whether statistical
    /// transforms become a leakage-safe preFeaturizer or stay CSV-baked (applied, but leaky → warn).
    /// </summary>
    public static bool SupportsPreFeaturizer(string task)
    {
        var normalized = (task ?? string.Empty).ToLowerInvariant().Replace('_', '-');
        return normalized switch
        {
            "binary-classification" => true,
            "multiclass-classification" => true,
            "regression" => true,
            _ => false
        };
    }

    /// <summary>
    /// Single source of truth for whether a task requires a label column. The unsupervised tasks
    /// (anomaly-detection, clustering, time-series-anomaly) are label-optional: when no label is
    /// configured, <see cref="Data.CsvDataLoader"/> loads a dummy label and treats every column as
    /// a feature. This set was duplicated as inline HashSets in <c>ConfigMerger</c>, <c>InitCommand</c>
    /// and <c>TrainCommand</c> — and had already drifted (TrainCommand omitted time-series-anomaly,
    /// and ValidateCommand was missing the concept entirely, so it errored "Label required" on a valid
    /// unsupervised project that <c>train</c> accepts). All four now read this (TD-06-shaped fix).
    /// Unknown tasks return <c>true</c> (require a label) — the conservative default.
    /// </summary>
    public static bool RequiresLabel(string? task)
    {
        var normalized = (task ?? string.Empty).ToLowerInvariant().Replace('_', '-');
        return normalized is not ("anomaly-detection" or "clustering" or "time-series-anomaly");
    }

    /// <summary>
    /// Single source of truth for which task types are time-series. These tasks ignore
    /// <c>config.TestSplit</c> entirely: a random train/test split would break temporal order,
    /// so <see cref="RunAsync"/> feeds the full dataset and the trainer holds out the last
    /// <c>horizon</c> rows internally (see the <c>isTimeSeriesTask</c> branch in this file).
    /// Because the split value is silently inert for these tasks, <c>ValidateCommand</c> reuses
    /// this to warn the user rather than let test_split look effective.
    /// </summary>
    public static bool IsTimeSeriesTask(string? task)
    {
        var normalized = (task ?? string.Empty).ToLowerInvariant().Replace('_', '-');
        return normalized is "forecasting" or "time-series-anomaly";
    }

    public async Task<AutoMLResult> RunAsync(
        TrainingConfig config,
        IProgress<TrainingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Load and split data
        IDataView trainSet, testSet;
        var isTimeSeriesTask = IsTimeSeriesTask(config.Task);

        // Collect columns that must be preserved as individual columns (not merged into Features vector)
        var preserveColumns = new List<string>();
        if (!string.IsNullOrEmpty(config.GroupColumn)) preserveColumns.Add(config.GroupColumn);
        if (!string.IsNullOrEmpty(config.UserColumn)) preserveColumns.Add(config.UserColumn);
        if (!string.IsNullOrEmpty(config.ItemColumn)) preserveColumns.Add(config.ItemColumn);
        // preFeaturizer columns (normalize/scale/fill-mean) must stay individually addressable;
        // otherwise InferColumns merges them into the Features vector and the preFeaturizer
        // (e.g. NormalizeMinMax("age","age")) throws "Could not find input column 'age'".
        if (config.PreFeaturizerColumns is { Count: > 0 })
            preserveColumns.AddRange(config.PreFeaturizerColumns);
        var preserve = preserveColumns.Count > 0 ? preserveColumns : null;

        // Directory-based tasks (image classification) need a different loader than the
        // injected CSV loader. The factory keeps the tabular path byte-for-byte unchanged.
        var loader = DataLoaderFactory.IsDirectoryBased(config.Task)
            ? DataLoaderFactory.Create(config.Task, _mlContext)
            : _dataLoader;

        if (!string.IsNullOrEmpty(config.TestDataFile))
        {
            // Pre-split data (e.g. balanced training with separate test set)
            trainSet = loader.LoadData(config.DataFile, config.LabelColumn, config.Task, preserve);
            testSet = loader.LoadData(config.TestDataFile, config.LabelColumn, config.Task, preserve);
        }
        else if (isTimeSeriesTask)
        {
            // Time series: use full dataset (no random split — temporal order matters)
            // Forecasting/TS-Anomaly handle holdout internally
            var dataView = loader.LoadData(config.DataFile, config.LabelColumn, config.Task, preserve);
            trainSet = dataView;
            testSet = dataView; // same data — internal holdout handles evaluation
        }
        else
        {
            var dataView = loader.LoadData(config.DataFile, config.LabelColumn, config.Task, preserve);
            (trainSet, testSet) = loader.SplitData(dataView, config.TestSplit);
        }

        // Discover hooks (zero-overhead if .mloop/scripts/hooks/ doesn't exist)
        var hooks = await _scriptDiscovery.DiscoverHooksAsync().ConfigureAwait(false);

        // Execute pre-train hooks
        if (hooks.Count > 0)
        {
            var preTrainContext = new HookContext
            {
                HookType = HookType.PreTrain,
                HookName = "pre-train",
                MLContext = _mlContext,
                DataView = trainSet,
                ProjectRoot = _projectRoot!,
                Logger = _logger,
                Metadata = new Dictionary<string, object>
                {
                    ["LabelColumn"] = config.LabelColumn,
                    ["TaskType"] = config.Task,
                    ["TimeLimitSeconds"] = config.TimeLimitSeconds
                }
            };

            foreach (var hook in hooks)
            {
                var hookResult = await hook.ExecuteAsync(preTrainContext).ConfigureAwait(false);
                if (hookResult.Action == HookAction.Abort)
                {
                    throw new InvalidOperationException(
                        $"Hook '{hook.Name}' aborted training: {hookResult.Message}");
                }
            }
        }

        // Ensure the task's on-demand native runtime is installed and loaded (no-op for tabular tasks).
        Runtime.RuntimeManager.EnsureRuntimeForTask(config.Task);

        // Run AutoML based on task type
        var result = config.Task.ToLowerInvariant() switch
        {
            "binary-classification" => await RunBinaryClassificationAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "multiclass-classification" => await RunMulticlassClassificationAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "regression" => await RunRegressionAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "anomaly-detection" => await RunAnomalyDetectionAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "clustering" => await RunClusteringAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "ranking" => await RunRankingAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "forecasting" => await RunForecastingAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "time-series-anomaly" => await RunTimeSeriesAnomalyAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "recommendation" => await RunRecommendationAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "image-classification" => await RunImageClassificationAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "text-classification" => await RunTextClassificationAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "sentence-similarity" => await RunSentenceSimilarityAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "ner" => await RunNerAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "object-detection" => await RunObjectDetectionAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "question-answering" => await RunQuestionAnsweringAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            _ => throw new NotSupportedException($"Task type '{config.Task}' is not supported")
        };

        // Execute post-train hooks
        if (hooks.Count > 0)
        {
            var postTrainContext = new HookContext
            {
                HookType = HookType.PostTrain,
                HookName = "post-train",
                MLContext = _mlContext,
                DataView = testSet,
                Model = result.Model,
                ProjectRoot = _projectRoot!,
                Logger = _logger,
                Metadata = new Dictionary<string, object>
                {
                    ["LabelColumn"] = config.LabelColumn,
                    ["TaskType"] = config.Task,
                    ["BestTrainer"] = result.BestTrainer,
                    ["Metrics"] = result.Metrics
                }
            };

            foreach (var hook in hooks)
            {
                try
                {
                    await hook.ExecuteAsync(postTrainContext).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    _logger.Warning($"Post-train hook '{hook.Name}' failed: {ex.Message}");
                }
            }
        }

        // Capture input schema from training data for prediction use
        if (result.Schema == null)
        {
            var schema = CaptureInputSchema(trainSet, config.LabelColumn, config.Task);
            result = new AutoMLResult
            {
                BestTrainer = result.BestTrainer,
                Model = result.Model,
                Metrics = result.Metrics,
                RowCount = result.RowCount,
                Schema = schema,
                ResidualModel = result.ResidualModel, // ② regression wave: preserve the σ(x) model through the schema-capture rewrap
            };
        }

        return result;
    }

    /// <summary>
    /// Captures the input schema from the training IDataView for later use in prediction.
    /// </summary>
    private InputSchemaInfo CaptureInputSchema(IDataView trainSet, string labelColumn, string taskType)
    {
        var columns = new List<ColumnSchema>();
        var schema = trainSet.Schema;

        foreach (var col in schema)
        {
            // Skip internal ML.NET columns
            if (col.IsHidden) continue;

            var purpose = col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase)
                ? "Label"
                : "Feature";

            var dataType = col.Type.RawType.Name switch
            {
                "Single" => "Single",
                "Double" => "Double",
                "Int32" => "Int32",
                "Int64" => "Int64",
                "Boolean" => "Boolean",
                "String" => "String",
                _ => col.Type.RawType.Name,
            };

            // Capture categorical values for string columns (important for prediction dimension matching)
            List<string>? categoricalValues = null;
            int? uniqueValueCount = null;

            if (dataType == "String" && purpose == "Feature")
            {
                try
                {
                    var colData = trainSet.GetColumn<string>(col.Name).Where(v => !string.IsNullOrEmpty(v)).Distinct().ToList();
                    if (colData.Count <= 10000) // reasonable limit
                    {
                        categoricalValues = colData;
                        uniqueValueCount = colData.Count;
                    }
                    else
                    {
                        uniqueValueCount = colData.Count;
                    }
                }
                catch
                {
                    // Column extraction failed — skip categorical capture
                }
            }

            columns.Add(new ColumnSchema
            {
                Name = col.Name,
                DataType = dataType,
                Purpose = purpose,
                CategoricalValues = categoricalValues,
                UniqueValueCount = uniqueValueCount,
            });
        }

        return new InputSchemaInfo
        {
            Columns = columns,
            CapturedAt = DateTime.UtcNow,
        };
    }

    private async Task<AutoMLResult> RunBinaryClassificationAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        var optimizingMetric = GetBinaryMetric(config.Metric);
        string? metricFallbackNote = null;

        try
        {
            return await RunBinaryClassificationCoreAsync(
                trainSet, testSet, config, optimizingMetric, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (IsAucUndefinedException(ex))
        {
            // BUG-22: AUC requires both positive and negative samples in the test set.
            // BUG-24: AutoML internally computes AUC regardless of the user's requested metric,
            // so this error can occur even with --metric accuracy or other non-AUC metrics.
            // Fall back to F1Score which is more robust for imbalanced data.
            if (optimizingMetric == BinaryClassificationMetric.F1Score)
            {
                // Already using F1Score — fall back to manual pipeline directly.
                _logger.Warning("AutoML failed with F1Score metric. Falling back to direct pipeline training (SDCA).");
                return await RunManualBinaryClassificationAsync(
                    trainSet, testSet, config, cancellationToken).ConfigureAwait(false);
            }

            var originalMetric = optimizingMetric.ToString();
            _logger.Warning($"AUC computation failed during {originalMetric} optimization (extreme class imbalance). Falling back to F1Score.");
            metricFallbackNote = $"{originalMetric}→F1Score (AUC imbalance)";

            try
            {
                return await RunBinaryClassificationCoreAsync(
                    trainSet, testSet, config, BinaryClassificationMetric.F1Score, cancellationToken,
                    metricFallbackNote).ConfigureAwait(false);
            }
            catch (Exception fallbackEx) when (IsAucUndefinedException(fallbackEx))
            {
                // BUG-36: AutoML failed with both metrics — fall back to manual pipeline training.
                _logger.Warning("AutoML failed with both metrics. Falling back to direct pipeline training (SDCA).");
                return await RunManualBinaryClassificationAsync(
                    trainSet, testSet, config, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private async Task<AutoMLResult> RunBinaryClassificationCoreAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        BinaryClassificationMetric optimizingMetric,
        CancellationToken cancellationToken,
        string? metricFallbackNote = null)
    {
        var settings = new BinaryExperimentSettings
        {
            MaxExperimentTimeInSeconds = (uint)config.TimeLimitSeconds,
            OptimizingMetric = optimizingMetric,
            CancellationToken = cancellationToken
        };

        var experiment = _mlContext.Auto().CreateBinaryClassificationExperiment(settings);

        // BUG-25: Build explicit ColumnInformation to ensure text columns get
        // TextFeaturizingEstimator instead of being ignored by AutoML's internal inference.
        var columnInfo = BuildColumnInformation(trainSet, config.LabelColumn, m => _logger.Info(m), config.ColumnOverrides);

        var experimentResult = await Task.Run(
            () => columnInfo != null
                ? experiment.Execute(trainSet, columnInfo, config.PreFeaturizer)
                : experiment.Execute(trainSet, labelColumnName: config.LabelColumn, preFeaturizer: config.PreFeaturizer),
            cancellationToken).ConfigureAwait(false);

        // Evaluate on test set
        var predictions = experimentResult.BestRun.Model.Transform(testSet);

        // BUG-24: Some AutoML pipelines (non-calibrated models) don't produce a Probability column.
        // Use EvaluateNonCalibrated when Probability column is missing.
        var hasProbability = predictions.Schema.GetColumnOrNull("Probability") != null;

        var metricsDict = new Dictionary<string, double>();

        if (hasProbability)
        {
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions, config.LabelColumn);
            metricsDict["accuracy"] = metrics.Accuracy;
            metricsDict["f1_score"] = metrics.F1Score;
            metricsDict["precision"] = metrics.PositivePrecision;
            metricsDict["recall"] = metrics.PositiveRecall;
            metricsDict["negative_recall"] = metrics.NegativeRecall;

            if (!double.IsNaN(metrics.AreaUnderRocCurve))
                metricsDict["auc"] = metrics.AreaUnderRocCurve;
        }
        else
        {
            var metrics = _mlContext.BinaryClassification.EvaluateNonCalibrated(predictions, config.LabelColumn);
            metricsDict["accuracy"] = metrics.Accuracy;
            metricsDict["f1_score"] = metrics.F1Score;
            metricsDict["precision"] = metrics.PositivePrecision;
            metricsDict["recall"] = metrics.PositiveRecall;
            metricsDict["negative_recall"] = metrics.NegativeRecall;

            if (!double.IsNaN(metrics.AreaUnderRocCurve))
                metricsDict["auc"] = metrics.AreaUnderRocCurve;
        }

        var trainerName = experimentResult.BestRun.TrainerName;
        if (metricFallbackNote != null)
        {
            trainerName += $" [metric fallback: {metricFallbackNote}]";
        }

        return new AutoMLResult
        {
            BestTrainer = trainerName,
            Model = EnsureCalibratedModel(_mlContext, experimentResult.BestRun.Model, predictions, config.LabelColumn, hasProbability),
            Metrics = metricsDict,
            RowCount = trainSet.GetRowCount() ?? 0
        };
    }

    /// <summary>
    /// D15 (BUG-24 follow-through): AutoML can select a binary trainer (e.g. FastForest/FastTree)
    /// whose output has no "Probability" column. Training already falls back to
    /// <see cref="BinaryClassificationCatalog.EvaluateNonCalibrated"/> so metrics still compute, but
    /// the *saved* model stayed uncalibrated — every downstream consumer (mloop predict, mloop serve
    /// /predict, HoneAI's confidence-gated dual-check) then reads a permanently-null Probability, and
    /// confidence collapses to 0 for every prediction (observed: 100% escalation in the honeai-sim
    /// live loop on KAMP SEQ089, `FastForestBinary`). Appending a Platt calibrator — fit on the
    /// already-computed test predictions, which carry the raw Score the calibrator needs — restores a
    /// real Probability column on the exact model that gets persisted and served.
    /// </summary>
    public static ITransformer EnsureCalibratedModel(
        MLContext mlContext, ITransformer model, IDataView predictions, string labelColumn, bool hasProbability)
    {
        if (hasProbability) return model;

        var calibrator = mlContext.BinaryClassification.Calibrators.Platt(labelColumnName: labelColumn, scoreColumnName: "Score");
        var calibratorTransformer = calibrator.Fit(predictions);
        return model.Append(calibratorTransformer);
    }

    /// <summary>
    /// ② regression wave (prediction-interval): computes split-conformal prediction-interval
    /// half-widths from the held-out test-set predictions. For each confidence level the half-width
    /// <c>q</c> is the finite-sample-corrected empirical quantile of the absolute residuals
    /// <c>|Label - Score|</c> over the holdout, so the interval <c>[Score - q, Score + q]</c> carries
    /// a distribution-free marginal-coverage guarantee (P(|y - ŷ| ≤ q) ≥ level) that is
    /// <b>model-agnostic</b> — it works with whatever trainer AutoML selects, unlike ML.NET's
    /// trainer-locked quantile-regression forests which would break the free trainer sweep D15/D16
    /// depend on. Returns one scalar per level (<c>interval_half_width_{pct}</c>) plus
    /// <c>residual_std</c> (RMS of residuals, diagnostic), ready to merge into the regression metrics
    /// dict — they persist flat in metrics.json with no schema change. Homoscedastic (constant-width)
    /// by design; heteroscedastic (normalized/Mondrian conformal) is deferred demand-driven.
    /// Returns an empty dictionary (graceful) when the score/label columns are absent or the holdout
    /// is empty. Note: the marginal-coverage guarantee holds only when <paramref name="predictions"/>
    /// is a genuine holdout not seen during training (MLoop's test-set split satisfies this).
    /// </summary>
    public static Dictionary<string, double> ComputeConformalIntervals(
        IDataView predictions,
        string labelColumn,
        IReadOnlyList<double>? confidenceLevels = null,
        string scoreColumn = "Score")
    {
        var levels = confidenceLevels ?? new[] { 0.80, 0.90, 0.95 };
        var result = new Dictionary<string, double>();

        var labelCol = predictions.Schema.GetColumnOrNull(labelColumn);
        var scoreCol = predictions.Schema.GetColumnOrNull(scoreColumn);
        if (labelCol is null || scoreCol is null)
            return result;

        var absResiduals = new List<double>();
        double sumSquares = 0;
        using (var cursor = predictions.GetRowCursor(new[] { labelCol.Value, scoreCol.Value }))
        {
            var labelGetter = cursor.GetGetter<float>(labelCol.Value);
            var scoreGetter = cursor.GetGetter<float>(scoreCol.Value);
            float label = 0, score = 0;
            while (cursor.MoveNext())
            {
                labelGetter(ref label);
                scoreGetter(ref score);
                double residual = Math.Abs((double)label - score);
                absResiduals.Add(residual);
                sumSquares += residual * residual;
            }
        }

        int n = absResiduals.Count;
        if (n == 0)
            return result;

        absResiduals.Sort();
        result["residual_std"] = Math.Sqrt(sumSquares / n);

        foreach (var level in levels)
        {
            // Split-conformal quantile: the smallest half-width q such that at least
            // ceil((n+1)*level) of the n calibration residuals are ≤ q (1-based rank). When that
            // rank exceeds n — the level is too high for this sample size to certify — the exact
            // guarantee needs q = ∞; we clamp to the largest observed residual and the advertised
            // coverage may be optimistic (verified live per the design doc).
            int rank = (int)Math.Ceiling((n + 1) * level);
            double q = rank > n ? absResiduals[n - 1] : absResiduals[rank - 1];
            int pct = (int)Math.Round(level * 100);
            result[$"interval_half_width_{pct}"] = q;
        }

        return result;
    }

    /// <summary>
    /// ② regression wave (heteroscedastic): normalized split-conformal prediction intervals. Where
    /// <see cref="ComputeConformalIntervals"/> gives one constant half-width for every row (valid
    /// coverage but no per-row triage signal — live-measured recall of the large-error rows equals
    /// random, cycle-134 M-13), this fits an auxiliary regressor σ(x) that predicts the *magnitude*
    /// of the residual from the features, so the band <c>[ŷ - q·σ(x), ŷ + q·σ(x)]</c> is wide exactly
    /// where the model is uncertain. The band width itself becomes the regression escalate signal.
    /// <para>
    /// Coverage validity is preserved by splitting the held-out test set into two disjoint folds: the
    /// σ-model is fit on fold C1, and the normalized nonconformity scores <c>α = |y-ŷ| / σ(x)</c> whose
    /// quantile becomes <c>q</c> are measured on fold C2 (σ never saw C2's residuals), so the marginal
    /// guarantee <c>P(|y-ŷ| ≤ q·σ(x)) ≥ level</c> still holds. σ(x) = <c>max(aux(x), 0) + β</c> with a
    /// positive floor β (a low quantile of the C1 residuals) so a near-zero aux prediction cannot
    /// collapse the band. When the errors are homoscedastic the aux model degenerates toward a
    /// constant and the band degrades gracefully to the constant-width result (backward-safe).
    /// </para>
    /// The aux trainer is SDCA linear regression: it adds no package dependency beyond core Microsoft.ML
    /// and captures the (linear) feature-driven heteroscedasticity that the offline ceiling analysis on
    /// SEQ026 confirmed (recall 0.10→0.24). A tree aux is a demand-driven future option if a live set
    /// shows nonlinear σ. Returns <c>null</c> (graceful) when score/label are absent, no numeric feature
    /// exists, or the holdout is too small to split — callers fall back to the constant-width band.
    /// </summary>
    /// <param name="ml">The ML.NET context used to fit the auxiliary σ-model and split the folds.</param>
    /// <param name="scoredPredictions">The held-out test set already transformed by the main model
    /// (carries the raw feature columns, the label column, and the <paramref name="scoreColumn"/>).</param>
    /// <param name="labelColumn">Name of the true-value column in <paramref name="scoredPredictions"/>.</param>
    /// <param name="numericFeatureColumns">Raw numeric feature columns the σ-model reads; non-existent
    /// names are ignored. Must be present in the predict-time input for the saved aux model to apply.</param>
    /// <param name="confidenceLevels">Coverage levels to certify (default 0.80/0.90/0.95).</param>
    /// <param name="scoreColumn">Name of the main model's point-prediction column (default "Score").</param>
    /// <param name="seed">Deterministic seed for the C1/C2 fold split (reproducibility).</param>
    public static NormalizedConformalResult? ComputeNormalizedConformal(
        MLContext ml,
        IDataView scoredPredictions,
        string labelColumn,
        IReadOnlyList<string> numericFeatureColumns,
        IReadOnlyList<double>? confidenceLevels = null,
        string scoreColumn = "Score",
        int seed = 42)
    {
        var levels = confidenceLevels ?? new[] { 0.80, 0.90, 0.95 };

        if (scoredPredictions.Schema.GetColumnOrNull(labelColumn) is null ||
            scoredPredictions.Schema.GetColumnOrNull(scoreColumn) is null)
            return null;

        var feats = numericFeatureColumns
            .Where(c => c != labelColumn && c != scoreColumn &&
                        scoredPredictions.Schema.GetColumnOrNull(c) is { } col && IsSingleOrVectorOfSingle(col.Type))
            .Distinct()
            .ToArray();
        if (feats.Length == 0)
            return null;

        // Normalize the label/score column names so the CustomMapping (which binds by property name)
        // can compute Target = |Label - Score| regardless of the caller's label column name.
        IDataView view = scoredPredictions;
        if (labelColumn != "Label")
            view = ml.Transforms.CopyColumns("Label", labelColumn).Fit(view).Transform(view);
        if (scoreColumn != "Score")
            view = ml.Transforms.CopyColumns("Score", scoreColumn).Fit(view).Transform(view);

        var mapping = ml.Transforms.CustomMapping<ConformalLabelScore, ConformalResidual>(
            (input, output) => output.Target = Math.Abs(input.Label - input.Score), contractName: null);
        var withTarget = mapping.Fit(view).Transform(view);

        // Disjoint folds: C1 fits σ(x), C2 calibrates the normalized quantile q (coverage validity).
        var split = ml.Data.TrainTestSplit(withTarget, testFraction: 0.5, seed: seed);
        var c1 = split.TrainSet;
        var c2 = split.TestSet;

        var c1Residuals = ReadColumn(ml, c1, "Target");
        if (c1Residuals.Count < 5 || GetRowCount(ml, c2) < 5)
            return null; // too small to fit + calibrate meaningfully

        // σ-model: raw numeric features → |residual|. Trained only on C1.
        var auxPipeline = ml.Transforms.Concatenate("Features", feats)
            .Append(ml.Transforms.NormalizeMinMax("Features"))
            .Append(ml.Regression.Trainers.Sdca(labelColumnName: "Target", featureColumnName: "Features"));
        var auxModel = auxPipeline.Fit(c1);

        // Positive floor β: a low quantile of the C1 residuals, so σ never collapses the band.
        var c1Sorted = c1Residuals.OrderBy(r => r).ToList();
        double beta = c1Sorted[(int)(c1Sorted.Count * 0.10)];
        double meanResid = c1Residuals.Average();
        if (beta <= 0) beta = Math.Max(meanResid * 0.01, 1e-6);

        // Normalized nonconformity α = |resid| / σ(x) on C2 (σ from the aux model, unseen residuals).
        var c2Scored = auxModel.Transform(c2);
        var sigmaRaw = ReadColumn(ml, c2Scored, "Score");   // aux SDCA output column
        var c2Residuals = ReadColumn(ml, c2Scored, "Target");
        if (sigmaRaw.Count == 0 || sigmaRaw.Count != c2Residuals.Count)
            return null;

        var alphas = new List<double>(sigmaRaw.Count);
        for (int i = 0; i < sigmaRaw.Count; i++)
        {
            double sigma = Math.Max(sigmaRaw[i], 0.0) + beta;
            alphas.Add(c2Residuals[i] / sigma);
        }
        alphas.Sort();

        var metrics = new Dictionary<string, double> { ["interval_beta"] = beta };
        int n = alphas.Count;
        foreach (var level in levels)
        {
            int rank = (int)Math.Ceiling((n + 1) * level);
            double q = rank > n ? alphas[n - 1] : alphas[rank - 1];
            int pct = (int)Math.Round(level * 100);
            metrics[$"norm_interval_q_{pct}"] = q;
        }

        return new NormalizedConformalResult(auxModel, metrics);
    }

    /// <summary>
    /// Picks the feature representation the σ-model reads from a scored prediction view. MLoop
    /// pre-featurizes into a single <c>Features</c> vector (the individual raw columns are gone by the
    /// time the model transforms the test set), so the aux model consumes that same featurized vector —
    /// which also lets it see categorical/text-derived features, not just raw numerics. Prefers a
    /// non-internal (<c>__…__</c> are AutoML scratch columns) vector-of-Single column named such as
    /// <c>Features</c>; falls back to scalar Single columns for callers whose view was never
    /// pre-featurized. Excludes the label and the point-prediction <c>Score</c>.
    /// </summary>
    private static string[] SelectSigmaFeatureColumns(DataViewSchema schema, string labelColumn)
    {
        var vector = schema.FirstOrDefault(c =>
            !c.IsHidden && c.Name != labelColumn && c.Name != "Score" &&
            !c.Name.StartsWith("__", StringComparison.Ordinal) &&
            c.Type is VectorDataViewType v && v.ItemType == NumberDataViewType.Single);
        if (vector.Name is not null)
            return new[] { vector.Name };

        return schema
            .Where(c => !c.IsHidden && c.Name != labelColumn && c.Name != "Score" &&
                        c.Type == NumberDataViewType.Single)
            .Select(c => c.Name)
            .ToArray();
    }

    /// <summary>True for a scalar Single column or a vector-of-Single column (both valid σ-model inputs).
    /// MLoop pre-featurizes into a single <c>Features</c> vector, so the aux model usually consumes that
    /// rather than individual scalar columns.</summary>
    private static bool IsSingleOrVectorOfSingle(DataViewType type) =>
        type == NumberDataViewType.Single ||
        (type is VectorDataViewType v && v.ItemType == NumberDataViewType.Single);

    /// <summary>Reads a Single-typed column into a list (helper for conformal calibration).</summary>
    private static List<double> ReadColumn(MLContext ml, IDataView view, string columnName)
    {
        var values = new List<double>();
        var col = view.Schema.GetColumnOrNull(columnName);
        if (col is null) return values;
        using var cursor = view.GetRowCursor(new[] { col.Value });
        var getter = cursor.GetGetter<float>(col.Value);
        float v = 0;
        while (cursor.MoveNext())
        {
            getter(ref v);
            values.Add(v);
        }
        return values;
    }

    private static long GetRowCount(MLContext ml, IDataView view)
    {
        long count = view.GetRowCount() ?? -1;
        if (count >= 0) return count;
        count = 0;
        using var cursor = view.GetRowCursor(Array.Empty<DataViewSchema.Column>());
        while (cursor.MoveNext()) count++;
        return count;
    }

    /// <summary>
    /// BUG-36: Manual pipeline fallback for small datasets where AutoML's internal AUC
    /// computation fails. Builds an explicit ML.NET pipeline with SDCA trainer, bypassing
    /// AutoML's cross-validation entirely.
    /// </summary>
    private async Task<AutoMLResult> RunManualBinaryClassificationAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        CancellationToken cancellationToken)
    {
        var columnInfo = BuildColumnInformation(trainSet, config.LabelColumn, m => _logger.Info(m), config.ColumnOverrides);

        // Build feature pipeline based on column types
        var featurePipeline = BuildFeaturePipeline(trainSet, config.LabelColumn, columnInfo);

        // Append SDCA trainer (robust general-purpose linear classifier)
        var trainingPipeline = featurePipeline
            .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: config.LabelColumn,
                featureColumnName: "Features"));

        var model = await Task.Run(
            () => trainingPipeline.Fit(trainSet),
            cancellationToken).ConfigureAwait(false);

        // Evaluate on test set
        var predictions = model.Transform(testSet);
        var hasProbability = predictions.Schema.GetColumnOrNull("Probability") != null;

        var metricsDict = new Dictionary<string, double>();

        if (hasProbability)
        {
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions, config.LabelColumn);
            metricsDict["accuracy"] = metrics.Accuracy;
            metricsDict["f1_score"] = metrics.F1Score;
            metricsDict["precision"] = metrics.PositivePrecision;
            metricsDict["recall"] = metrics.PositiveRecall;
            metricsDict["negative_recall"] = metrics.NegativeRecall;
            if (!double.IsNaN(metrics.AreaUnderRocCurve))
                metricsDict["auc"] = metrics.AreaUnderRocCurve;
        }
        else
        {
            var metrics = _mlContext.BinaryClassification.EvaluateNonCalibrated(predictions, config.LabelColumn);
            metricsDict["accuracy"] = metrics.Accuracy;
            metricsDict["f1_score"] = metrics.F1Score;
            metricsDict["precision"] = metrics.PositivePrecision;
            metricsDict["recall"] = metrics.PositiveRecall;
            metricsDict["negative_recall"] = metrics.NegativeRecall;
            if (!double.IsNaN(metrics.AreaUnderRocCurve))
                metricsDict["auc"] = metrics.AreaUnderRocCurve;
        }

        return new AutoMLResult
        {
            BestTrainer = "SdcaLogisticRegression [manual fallback: AutoML AUC failure]",
            Model = EnsureCalibratedModel(_mlContext, model, predictions, config.LabelColumn, hasProbability),
            Metrics = metricsDict,
            RowCount = trainSet.GetRowCount() ?? 0
        };
    }

    /// <summary>
    /// Builds a feature engineering pipeline based on column types.
    /// Used by manual training fallback when AutoML is unavailable.
    /// </summary>
    private IEstimator<ITransformer> BuildFeaturePipeline(
        IDataView data, string labelColumn, ColumnInformation? columnInfo)
    {
        var featureColumns = new List<string>();
        IEstimator<ITransformer>? pipeline = null;

        if (columnInfo != null)
        {
            // Text columns → FeaturizeText
            foreach (var textCol in columnInfo.TextColumnNames)
            {
                var outputCol = $"_Text_{textCol}";
                var textEstimator = _mlContext.Transforms.Text.FeaturizeText(outputCol, textCol);
                pipeline = pipeline == null
                    ? (IEstimator<ITransformer>)textEstimator
                    : pipeline.Append(textEstimator);
                featureColumns.Add(outputCol);
            }

            // Numeric columns pass through directly
            foreach (var numCol in columnInfo.NumericColumnNames)
            {
                featureColumns.Add(numCol);
            }

            // Categorical columns → OneHotEncoding
            foreach (var catCol in columnInfo.CategoricalColumnNames)
            {
                var outputCol = $"_Cat_{catCol}";
                var catEstimator = _mlContext.Transforms.Categorical.OneHotEncoding(outputCol, catCol);
                pipeline = pipeline == null
                    ? (IEstimator<ITransformer>)catEstimator
                    : pipeline.Append(catEstimator);
                featureColumns.Add(outputCol);
            }
        }
        else
        {
            // No column info — use all non-label columns as numeric features
            foreach (var col in data.Schema)
            {
                if (col.IsHidden) continue;
                if (col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase)) continue;
                featureColumns.Add(col.Name);
            }
        }

        // Concatenate all features into a single "Features" column
        var concatEstimator = _mlContext.Transforms.Concatenate("Features", featureColumns.ToArray());
        pipeline = pipeline == null
            ? (IEstimator<ITransformer>)concatEstimator
            : pipeline.Append(concatEstimator);

        return pipeline;
    }

    private async Task<AutoMLResult> RunMulticlassClassificationAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        var settings = new MulticlassExperimentSettings
        {
            MaxExperimentTimeInSeconds = (uint)config.TimeLimitSeconds,
            OptimizingMetric = GetMulticlassMetric(config.Metric),
            CancellationToken = cancellationToken
        };

        var experiment = _mlContext.Auto().CreateMulticlassClassificationExperiment(settings);

        // BUG-25: Explicit ColumnInformation for text column featurization
        var columnInfo = BuildColumnInformation(trainSet, config.LabelColumn, m => _logger.Info(m), config.ColumnOverrides);

        var experimentResult = await Task.Run(
            () => columnInfo != null
                ? experiment.Execute(trainSet, columnInfo, config.PreFeaturizer)
                : experiment.Execute(trainSet, labelColumnName: config.LabelColumn, preFeaturizer: config.PreFeaturizer),
            cancellationToken).ConfigureAwait(false);

        // Evaluate on test set
        var predictions = experimentResult.BestRun.Model.Transform(testSet);
        var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, config.LabelColumn);

        var metricsDict = new Dictionary<string, double>
        {
            ["macro_accuracy"] = metrics.MacroAccuracy,
            ["micro_accuracy"] = metrics.MicroAccuracy,
            ["log_loss"] = metrics.LogLoss
        };

        // Calculate Macro F1 from confusion matrix per-class precision/recall
        try
        {
            var cm = metrics.ConfusionMatrix;
            var classCount = cm.PerClassPrecision.Count;
            if (classCount > 0)
            {
                double f1Sum = 0;
                for (int i = 0; i < classCount; i++)
                {
                    var p = cm.PerClassPrecision[i];
                    var r = cm.PerClassRecall[i];
                    f1Sum += (p + r) > 0 ? 2 * p * r / (p + r) : 0;
                }
                metricsDict["macro_f1"] = f1Sum / classCount;
            }
        }
        catch
        {
            // Non-critical: skip if confusion matrix unavailable
        }

        return new AutoMLResult
        {
            BestTrainer = experimentResult.BestRun.TrainerName,
            Model = experimentResult.BestRun.Model,
            Metrics = metricsDict,
            RowCount = trainSet.GetRowCount() ?? 0
        };
    }

    private async Task<AutoMLResult> RunRegressionAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        var settings = new RegressionExperimentSettings
        {
            MaxExperimentTimeInSeconds = (uint)config.TimeLimitSeconds,
            OptimizingMetric = GetRegressionMetric(config.Metric),
            CancellationToken = cancellationToken
        };

        var experiment = _mlContext.Auto().CreateRegressionExperiment(settings);

        // BUG-25: Explicit ColumnInformation for text column featurization
        var columnInfo = BuildColumnInformation(trainSet, config.LabelColumn, m => _logger.Info(m), config.ColumnOverrides);

        var experimentResult = await Task.Run(
            () => columnInfo != null
                ? experiment.Execute(trainSet, columnInfo, config.PreFeaturizer)
                : experiment.Execute(trainSet, labelColumnName: config.LabelColumn, preFeaturizer: config.PreFeaturizer),
            cancellationToken).ConfigureAwait(false);

        // Evaluate on test set
        var predictions = experimentResult.BestRun.Model.Transform(testSet);
        var metrics = _mlContext.Regression.Evaluate(predictions, config.LabelColumn);

        var metricsDict = new Dictionary<string, double>
        {
            ["r_squared"] = metrics.RSquared,
            ["rmse"] = metrics.RootMeanSquaredError,
            ["mae"] = metrics.MeanAbsoluteError,
            ["mse"] = metrics.MeanSquaredError
        };

        // ② regression wave: derive split-conformal prediction-interval half-widths from the same
        // held-out test predictions used for metrics, so predict/serve can surface an uncertainty
        // band [Score ± q] (the regression analogue of the confidence signal binary/multiclass expose).
        foreach (var interval in ComputeConformalIntervals(predictions, config.LabelColumn))
            metricsDict[interval.Key] = interval.Value;

        // ② regression wave (heteroscedastic): fit the per-row σ(x) band on top of the constant-width
        // one. The constant-width metrics stay as the backward-safe fallback; when the aux model fits,
        // its normalized-quantile metrics + the residual model let predict widen/narrow per row.
        ITransformer? residualModel = null;
        var featureColumns = SelectSigmaFeatureColumns(predictions.Schema, config.LabelColumn);
        var normalized = ComputeNormalizedConformal(_mlContext, predictions, config.LabelColumn, featureColumns);
        if (normalized is not null)
        {
            residualModel = normalized.AuxModel;
            foreach (var kv in normalized.Metrics)
                metricsDict[kv.Key] = kv.Value;
        }

        return new AutoMLResult
        {
            BestTrainer = experimentResult.BestRun.TrainerName,
            Model = experimentResult.BestRun.Model,
            Metrics = metricsDict,
            ResidualModel = residualModel,
            RowCount = trainSet.GetRowCount() ?? 0
        };
    }

    private async Task<AutoMLResult> RunAnomalyDetectionAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            // Identify numeric feature columns (exclude label, hidden, and non-numeric)
            var featureColumns = new List<string>();
            foreach (var col in trainSet.Schema)
            {
                if (col.IsHidden) continue;
                if (col.Name.Equals(config.LabelColumn, StringComparison.OrdinalIgnoreCase)) continue;
                if (col.Type is NumberDataViewType || col.Type is BooleanDataViewType
                    || (col.Type is VectorDataViewType vt && vt.ItemType is NumberDataViewType))
                    featureColumns.Add(col.Name);
            }

            if (featureColumns.Count == 0)
                throw new InvalidOperationException("No numeric feature columns found for anomaly detection.");

            _logger.Info($"Anomaly detection: {featureColumns.Count} feature columns, using RandomizedPca");

            // Determine PCA rank (auto: min of feature count and 20)
            var rank = Math.Min(featureColumns.Count, 20);

            // Build pipeline: Concatenate features → RandomizedPca
            var pipeline = _mlContext.Transforms.Concatenate("Features", featureColumns.ToArray())
                .Append(_mlContext.AnomalyDetection.Trainers.RandomizedPca(
                    featureColumnName: "Features",
                    rank: rank,
                    oversampling: 20));

            // Train
            progress?.Report(new TrainingProgress
            {
                TrialNumber = 1,
                TrainerName = "RandomizedPca",
                MetricName = "detection_rate",
                Metric = 0,
                ElapsedSeconds = 0
            });

            var model = pipeline.Fit(trainSet);

            // Evaluate on test set
            var predictions = model.Transform(testSet);

            var metricsDict = new Dictionary<string, double>();

            // Try ML.NET's built-in anomaly evaluation (requires label column)
            if (!string.IsNullOrEmpty(config.LabelColumn))
            {
                try
                {
                    var metrics = _mlContext.AnomalyDetection.Evaluate(predictions, config.LabelColumn);
                    metricsDict["auc"] = double.IsNaN(metrics.AreaUnderRocCurve) ? 0 : metrics.AreaUnderRocCurve;
                    metricsDict["detection_rate_at_fp5"] = metrics.DetectionRateAtFalsePositiveCount;
                }
                catch
                {
                    // Label column may not have correct format for AnomalyDetection.Evaluate
                    // Fall back to manual counting
                }
            }

            // Manual anomaly counting from predictions
            var predictedLabelCol = predictions.Schema.GetColumnOrNull("PredictedLabel");
            if (predictedLabelCol.HasValue)
            {
                long totalCount = 0;
                long anomalyCount = 0;

                using (var cursor = predictions.GetRowCursor(new[] { predictedLabelCol.Value }))
                {
                    var getter = cursor.GetGetter<bool>(predictedLabelCol.Value);
                    while (cursor.MoveNext())
                    {
                        bool isAnomaly = false;
                        getter(ref isAnomaly);
                        totalCount++;
                        if (isAnomaly) anomalyCount++;
                    }
                }

                metricsDict["anomaly_count"] = anomalyCount;
                metricsDict["total_count"] = totalCount;
                metricsDict["detection_rate"] = totalCount > 0 ? (double)anomalyCount / totalCount : 0;
            }

            return new AutoMLResult
            {
                BestTrainer = $"RandomizedPca (rank={rank})",
                Model = model,
                Metrics = metricsDict,
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunClusteringAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            // Identify numeric feature columns (exclude label, hidden, and non-numeric)
            var featureColumns = new List<string>();
            foreach (var col in trainSet.Schema)
            {
                if (col.IsHidden) continue;
                if (!string.IsNullOrEmpty(config.LabelColumn) &&
                    col.Name.Equals(config.LabelColumn, StringComparison.OrdinalIgnoreCase)) continue;
                if (col.Type is NumberDataViewType || col.Type is BooleanDataViewType
                    || (col.Type is VectorDataViewType vt && vt.ItemType is NumberDataViewType))
                    featureColumns.Add(col.Name);
            }

            if (featureColumns.Count == 0)
                throw new InvalidOperationException("No numeric feature columns found for clustering.");

            var concatenate = _mlContext.Transforms.Concatenate("Features", featureColumns.ToArray());

            // Determine K values to try
            int[] kValues;
            if (config.NumClusters > 0)
            {
                kValues = [config.NumClusters];
                _logger.Info($"Clustering: {featureColumns.Count} features, fixed k={config.NumClusters}");
            }
            else
            {
                // Auto-search: k=2..min(10, sqrt(rowCount))
                var rowCount = CountRows(trainSet);
                var maxK = Math.Min(10, Math.Max(2, (int)Math.Sqrt(rowCount)));
                kValues = Enumerable.Range(2, maxK - 1).ToArray();
                _logger.Info($"Clustering: {featureColumns.Count} features, searching k=2..{maxK}");
            }

            ITransformer? bestModel = null;
            double bestDbi = double.MaxValue; // DBI: lower is better (considers both separation and cohesion)
            int bestK = kValues[0];
            Dictionary<string, double>? bestMetrics = null;

            for (int i = 0; i < kValues.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var k = kValues[i];

                var pipeline = concatenate
                    .Append(_mlContext.Clustering.Trainers.KMeans(
                        featureColumnName: "Features",
                        numberOfClusters: k));

                var model = pipeline.Fit(trainSet);
                var predictions = model.Transform(testSet);

                // Evaluate. featureColumnName is REQUIRED for ML.NET to compute the Davies-Bouldin
                // Index — without it DBI comes back 0, which made `useDbi` below always false so the
                // K-search silently fell back to average_distance (monotonically decreasing with K)
                // and always picked the largest K. Passing "Features" restores real DBI-based K-search.
                var clusterMetrics = _mlContext.Clustering.Evaluate(
                    predictions, scoreColumnName: "Score", featureColumnName: "Features");
                var avgDistance = double.IsNaN(clusterMetrics.AverageDistance) ? double.MaxValue : clusterMetrics.AverageDistance;
                var dbi = double.IsNaN(clusterMetrics.DaviesBouldinIndex) ? double.MaxValue : clusterMetrics.DaviesBouldinIndex;
                var nmi = double.IsNaN(clusterMetrics.NormalizedMutualInformation) ? 0 : clusterMetrics.NormalizedMutualInformation;

                progress?.Report(new TrainingProgress
                {
                    TrialNumber = i + 1,
                    TrainerName = $"KMeans (k={k})",
                    MetricName = "davies_bouldin_index",
                    Metric = dbi,
                    ElapsedSeconds = 0
                });

                // Select K with lowest DBI (Davies-Bouldin Index).
                // Unlike average_distance which monotonically decreases with K,
                // DBI considers both cluster separation and cohesion, finding the natural K.
                // Fallback to average_distance if DBI is unavailable (0 or MaxValue).
                var useDbi = dbi > 0 && dbi < double.MaxValue;
                var isBetter = useDbi
                    ? dbi < bestDbi
                    : (bestDbi == double.MaxValue && avgDistance < (bestMetrics?.GetValueOrDefault("average_distance") ?? double.MaxValue));

                if (isBetter || bestModel == null)
                {
                    bestDbi = useDbi ? dbi : bestDbi;
                    bestModel = model;
                    bestK = k;
                    bestMetrics = new Dictionary<string, double>
                    {
                        ["average_distance"] = avgDistance,
                        ["davies_bouldin_index"] = dbi,
                        ["normalized_mutual_information"] = nmi,
                        ["num_clusters"] = k
                    };
                }
            }

            // Add cluster distribution info from best model
            if (bestModel != null && bestMetrics != null)
            {
                var bestPredictions = bestModel.Transform(testSet);
                var clusterCounts = new Dictionary<uint, long>();
                var predictedLabelCol = bestPredictions.Schema.GetColumnOrNull("PredictedLabel");

                if (predictedLabelCol.HasValue)
                {
                    using var cursor = bestPredictions.GetRowCursor(new[] { predictedLabelCol.Value });
                    var getter = cursor.GetGetter<uint>(predictedLabelCol.Value);
                    while (cursor.MoveNext())
                    {
                        uint clusterId = 0;
                        getter(ref clusterId);
                        clusterCounts[clusterId] = clusterCounts.GetValueOrDefault(clusterId) + 1;
                    }
                }

                if (clusterCounts.Count > 0)
                {
                    var totalCount = clusterCounts.Values.Sum();
                    var largestCluster = clusterCounts.Values.Max();
                    bestMetrics["cluster_count"] = clusterCounts.Count;
                    bestMetrics["largest_cluster_ratio"] = totalCount > 0 ? (double)largestCluster / totalCount : 0;
                }
            }

            return new AutoMLResult
            {
                BestTrainer = $"KMeans (k={bestK})",
                Model = bestModel!,
                Metrics = bestMetrics ?? new Dictionary<string, double>(),
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunRankingAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            if (string.IsNullOrEmpty(config.GroupColumn))
                throw new InvalidOperationException("Ranking task requires a group column. Set 'group_column' in mloop.yaml or use --group-column.");

            // Validate group column exists
            var groupCol = trainSet.Schema.GetColumnOrNull(config.GroupColumn);
            if (!groupCol.HasValue)
                throw new InvalidOperationException($"Group column '{config.GroupColumn}' not found in data.");

            // Identify numeric feature columns (exclude label, group, hidden, non-numeric)
            var featureColumns = new List<string>();
            foreach (var col in trainSet.Schema)
            {
                if (col.IsHidden) continue;
                if (col.Name.Equals(config.LabelColumn, StringComparison.OrdinalIgnoreCase)) continue;
                if (col.Name.Equals(config.GroupColumn, StringComparison.OrdinalIgnoreCase)) continue;
                if (col.Type is NumberDataViewType || col.Type is BooleanDataViewType
                    || (col.Type is VectorDataViewType vt && vt.ItemType is NumberDataViewType))
                    featureColumns.Add(col.Name);
            }

            if (featureColumns.Count == 0)
                throw new InvalidOperationException("No numeric feature columns found for ranking.");

            _logger.Info($"Ranking: {featureColumns.Count} features, group='{config.GroupColumn}', label='{config.LabelColumn}'");

            // Build pipeline: convert label to Single, hash group to Key, concatenate features
            var pipeline = _mlContext.Transforms.Conversion.ConvertType(config.LabelColumn, outputKind: Microsoft.ML.Data.DataKind.Single)
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("GroupId", config.GroupColumn))
                .Append(_mlContext.Transforms.Concatenate("Features", featureColumns.ToArray()));

            // Try both trainers: LightGbm and FastTree
            var trainers = new (string Name, Microsoft.ML.IEstimator<Microsoft.ML.ITransformer> Trainer)[]
            {
                ("LightGbmRanking", _mlContext.Ranking.Trainers.LightGbm(
                    labelColumnName: config.LabelColumn,
                    featureColumnName: "Features",
                    rowGroupColumnName: "GroupId")),
                ("FastTreeRanking", _mlContext.Ranking.Trainers.FastTree(
                    labelColumnName: config.LabelColumn,
                    featureColumnName: "Features",
                    rowGroupColumnName: "GroupId"))
            };

            ITransformer? bestModel = null;
            double bestNdcg = double.MinValue;
            string bestTrainerName = trainers[0].Name;
            Dictionary<string, double>? bestMetrics = null;

            for (int i = 0; i < trainers.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var (trainerName, trainer) = trainers[i];

                try
                {
                    var fullPipeline = pipeline.Append(trainer);
                    var model = fullPipeline.Fit(trainSet);
                    var predictions = model.Transform(testSet);

                    var rankMetrics = _mlContext.Ranking.Evaluate(predictions,
                        labelColumnName: config.LabelColumn,
                        rowGroupColumnName: "GroupId");

                    var ndcg = rankMetrics.NormalizedDiscountedCumulativeGains;
                    var dcg = rankMetrics.DiscountedCumulativeGains;

                    var metricsDict = new Dictionary<string, double>();

                    // Store per-level NDCG (NDCG@1, NDCG@2, ..., NDCG@10)
                    for (int level = 0; level < ndcg.Count; level++)
                    {
                        metricsDict[$"ndcg_at_{level + 1}"] = double.IsNaN(ndcg[level]) ? 0 : ndcg[level];
                    }
                    for (int level = 0; level < dcg.Count; level++)
                    {
                        metricsDict[$"dcg_at_{level + 1}"] = double.IsNaN(dcg[level]) ? 0 : dcg[level];
                    }

                    // Primary metric: NDCG@10 (or last available level)
                    var primaryNdcg = ndcg.Count > 0 ? ndcg[ndcg.Count - 1] : 0;
                    if (double.IsNaN(primaryNdcg)) primaryNdcg = 0;
                    metricsDict["ndcg"] = primaryNdcg;

                    progress?.Report(new TrainingProgress
                    {
                        TrialNumber = i + 1,
                        TrainerName = trainerName,
                        MetricName = "ndcg",
                        Metric = primaryNdcg,
                        ElapsedSeconds = 0
                    });

                    if (primaryNdcg > bestNdcg)
                    {
                        bestNdcg = primaryNdcg;
                        bestModel = model;
                        bestTrainerName = trainerName;
                        bestMetrics = metricsDict;
                    }
                }
                catch (Exception ex)
                {
                    var hint = ex.Message.Contains("invalid label")
                        ? " (FastTree requires labels in range 0-4)"
                        : "";
                    _logger.Info($"Ranking trainer '{trainerName}' failed: {ex.Message}{hint}");
                }
            }

            if (bestModel == null)
                throw new InvalidOperationException(
                    "All ranking trainers failed. Check data format and group column. " +
                    "Note: FastTree requires label values 0-4; LightGbm accepts any numeric range.");

            return new AutoMLResult
            {
                BestTrainer = bestTrainerName,
                Model = bestModel,
                Metrics = bestMetrics ?? new Dictionary<string, double>(),
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunForecastingAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            if (config.Horizon <= 0)
                throw new InvalidOperationException("Forecasting task requires horizon > 0.");

            // Find the value column (label column = the series to forecast)
            var valueColumn = config.LabelColumn;
            var valueCol = trainSet.Schema.GetColumnOrNull(valueColumn);
            if (!valueCol.HasValue)
                throw new InvalidOperationException($"Value column '{valueColumn}' not found in data.");

            if (valueCol.Value.Type is not NumberDataViewType and not (VectorDataViewType { ItemType: NumberDataViewType }))
                throw new InvalidOperationException($"Value column '{valueColumn}' must be numeric for forecasting.");

            // Determine parameters — use CountRows for lazy IDataView
            var totalRows = (int)CountRows(trainSet);
            if (totalRows == 0)
                throw new InvalidOperationException("Training data is empty.");

            var horizon = config.Horizon;
            var seriesLength = config.SeriesLength > 0 ? config.SeriesLength : totalRows;
            var windowSize = config.WindowSize > 0 ? config.WindowSize : Math.Max(2, seriesLength / 4);
            var trainSize = totalRows - horizon; // Hold out last `horizon` points for evaluation

            if (trainSize <= windowSize)
                throw new InvalidOperationException(
                    $"Insufficient data: {totalRows} rows with horizon={horizon}, need at least {windowSize + horizon + 1} rows.");

            _logger.Info($"Forecasting: series='{valueColumn}', {totalRows} rows, horizon={horizon}, window={windowSize}, seriesLength={seriesLength}");

            // Build SSA forecasting pipeline
            var pipeline = _mlContext.Forecasting.ForecastBySsa(
                outputColumnName: "ForecastedValues",
                inputColumnName: valueColumn,
                windowSize: windowSize,
                seriesLength: seriesLength,
                trainSize: trainSize,
                horizon: horizon,
                confidenceLowerBoundColumn: "LowerBound",
                confidenceUpperBoundColumn: "UpperBound",
                confidenceLevel: 0.95f);

            progress?.Report(new TrainingProgress
            {
                TrialNumber = 1,
                TrainerName = "SsaForecasting",
                MetricName = "mae",
                Metric = 0,
                ElapsedSeconds = 0
            });

            var model = pipeline.Fit(trainSet);

            // Evaluate: extract actual holdout values and compare with forecasts
            var metricsDict = new Dictionary<string, double>();

            // Get the actual values from the series (last `horizon` values are holdout)
            var actualValues = new List<float>();
            using (var cursor = trainSet.GetRowCursor(new[] { valueCol.Value }))
            {
                var getter = cursor.GetGetter<float>(valueCol.Value);
                while (cursor.MoveNext())
                {
                    float val = 0;
                    getter(ref val);
                    actualValues.Add(val);
                }
            }

            if (actualValues.Count >= trainSize + horizon)
            {
                // Get forecasted values via Transform
                var predictions = model.Transform(trainSet);
                var forecastCol = predictions.Schema.GetColumnOrNull("ForecastedValues");

                if (forecastCol.HasValue)
                {
                    // ForecastedValues is a VBuffer<float> containing the horizon-length forecast
                    using var forecastCursor = predictions.GetRowCursor(new[] { forecastCol.Value });
                    var forecastGetter = forecastCursor.GetGetter<VBuffer<float>>(forecastCol.Value);

                    // Move to last row to get the final forecast
                    VBuffer<float> forecastBuffer = default;
                    while (forecastCursor.MoveNext())
                    {
                        forecastGetter(ref forecastBuffer);
                    }

                    var forecastedValues = forecastBuffer.DenseValues().ToArray();
                    var holdoutActual = actualValues.Skip(trainSize).Take(horizon).ToArray();

                    if (forecastedValues.Length >= horizon && holdoutActual.Length == horizon)
                    {
                        double sumAbsError = 0, sumSqError = 0, sumAbsPercentError = 0;
                        int validMapeCount = 0;

                        for (int i = 0; i < horizon; i++)
                        {
                            var actual = (double)holdoutActual[i];
                            var predicted = (double)forecastedValues[i];
                            var error = actual - predicted;

                            sumAbsError += Math.Abs(error);
                            sumSqError += error * error;

                            if (Math.Abs(actual) > 1e-10)
                            {
                                sumAbsPercentError += Math.Abs(error / actual);
                                validMapeCount++;
                            }
                        }

                        metricsDict["mae"] = sumAbsError / horizon;
                        metricsDict["rmse"] = Math.Sqrt(sumSqError / horizon);
                        metricsDict["mape"] = validMapeCount > 0 ? sumAbsPercentError / validMapeCount : 0;
                    }
                }
            }

            metricsDict["horizon"] = horizon;
            metricsDict["window_size"] = windowSize;
            metricsDict["series_length"] = seriesLength;
            metricsDict["train_size"] = trainSize;

            return new AutoMLResult
            {
                BestTrainer = $"SsaForecasting (window={windowSize}, horizon={horizon})",
                Model = model,
                Metrics = metricsDict,
                RowCount = totalRows
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunTimeSeriesAnomalyAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            // Find the value column (label = the series to monitor)
            var valueColumn = config.LabelColumn;
            var valueCol = trainSet.Schema.GetColumnOrNull(valueColumn);
            if (!valueCol.HasValue)
                throw new InvalidOperationException($"Value column '{valueColumn}' not found in data.");

            if (valueCol.Value.Type is not NumberDataViewType and not (VectorDataViewType { ItemType: NumberDataViewType }))
                throw new InvalidOperationException($"Value column '{valueColumn}' must be numeric for time series anomaly detection.");

            var totalRows = (int)CountRows(trainSet);
            if (totalRows < 12)
                throw new InvalidOperationException($"Time series anomaly detection requires at least 12 data points (got {totalRows}).");

            _logger.Info($"Time series anomaly: series='{valueColumn}', {totalRows} rows");

            // Try multiple detectors: SrCnn (best), then SSA Spike, then SSA ChangePoint
            var detectors = new List<(string Name, Func<IEstimator<ITransformer>> Factory)>();

            // SR-CNN: Spectral Residual based — best general-purpose detector
            var srCnnWindowSize = Math.Min(64, Math.Max(8, totalRows / 4));
            detectors.Add(("SrCnnAnomaly", () =>
                _mlContext.Transforms.DetectAnomalyBySrCnn(
                    outputColumnName: "Prediction",
                    inputColumnName: valueColumn,
                    windowSize: srCnnWindowSize,
                    backAddWindowSize: 5,
                    lookaheadWindowSize: 5,
                    averagingWindowSize: 3,
                    judgementWindowSize: 21,
                    threshold: 0.3)));

            // SSA Spike Detector — trainingWindowSize must be > 2 * seasonalityWindowSize
            var ssaWindowSize = Math.Max(2, Math.Min(totalRows / 8, 50));
            var ssaPValueSize = Math.Max(2, ssaWindowSize / 2);
            var ssaTrainingSize = Math.Max(ssaWindowSize * 2 + 1, Math.Min(totalRows, 100));
            detectors.Add(("SsaSpikeDetector", () =>
                _mlContext.Transforms.DetectSpikeBySsa(
                    outputColumnName: "Prediction",
                    inputColumnName: valueColumn,
                    confidence: 95.0,
                    pvalueHistoryLength: ssaPValueSize,
                    trainingWindowSize: ssaTrainingSize,
                    seasonalityWindowSize: ssaWindowSize)));

            ITransformer? bestModel = null;
            string bestDetectorName = "";
            Dictionary<string, double>? bestMetrics = null;
            long bestAnomalyCount = -1;

            for (int i = 0; i < detectors.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var (name, factory) = detectors[i];

                try
                {
                    var estimator = factory();
                    var model = estimator.Fit(trainSet);
                    var predictions = model.Transform(trainSet);

                    // Count anomalies from Prediction column (VBuffer<double>: [alert, score, p-value])
                    var predCol = predictions.Schema.GetColumnOrNull("Prediction");
                    long anomalyCount = 0;
                    long rowCount = 0;

                    if (predCol.HasValue)
                    {
                        using var cursor = predictions.GetRowCursor(new[] { predCol.Value });
                        var getter = cursor.GetGetter<VBuffer<double>>(predCol.Value);

                        while (cursor.MoveNext())
                        {
                            VBuffer<double> pred = default;
                            getter(ref pred);
                            var values = pred.DenseValues().ToArray();
                            rowCount++;
                            // values[0] = alert (1 = anomaly), values[1] = score, values[2] = p-value
                            if (values.Length > 0 && values[0] != 0)
                                anomalyCount++;
                        }
                    }

                    var detectionRate = rowCount > 0 ? (double)anomalyCount / rowCount : 0;
                    var metricsDict = new Dictionary<string, double>
                    {
                        ["anomaly_count"] = anomalyCount,
                        ["total_count"] = rowCount,
                        ["detection_rate"] = detectionRate
                    };

                    progress?.Report(new TrainingProgress
                    {
                        TrialNumber = i + 1,
                        TrainerName = name,
                        MetricName = "detection_rate",
                        Metric = detectionRate,
                        ElapsedSeconds = 0
                    });

                    // Pick the first detector that produces reasonable results
                    if (bestModel == null)
                    {
                        bestModel = model;
                        bestDetectorName = name;
                        bestMetrics = metricsDict;
                        bestAnomalyCount = anomalyCount;
                    }
                }
                catch (Exception ex)
                {
                    _logger.Info($"Time series anomaly detector '{name}' failed: {ex.Message}");
                }
            }

            if (bestModel == null)
                throw new InvalidOperationException("All time series anomaly detectors failed.");

            return new AutoMLResult
            {
                BestTrainer = bestDetectorName,
                Model = bestModel,
                Metrics = bestMetrics ?? new Dictionary<string, double>(),
                RowCount = totalRows
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunRecommendationAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            if (string.IsNullOrEmpty(config.UserColumn))
                throw new InvalidOperationException("Recommendation task requires user_column.");
            if (string.IsNullOrEmpty(config.ItemColumn))
                throw new InvalidOperationException("Recommendation task requires item_column.");

            // Validate columns exist
            var userCol = trainSet.Schema.GetColumnOrNull(config.UserColumn);
            var itemCol = trainSet.Schema.GetColumnOrNull(config.ItemColumn);
            var labelCol = trainSet.Schema.GetColumnOrNull(config.LabelColumn);

            if (!userCol.HasValue)
                throw new InvalidOperationException($"User column '{config.UserColumn}' not found.");
            if (!itemCol.HasValue)
                throw new InvalidOperationException($"Item column '{config.ItemColumn}' not found.");
            if (!labelCol.HasValue)
                throw new InvalidOperationException($"Rating/label column '{config.LabelColumn}' not found.");

            _logger.Info($"Recommendation: user='{config.UserColumn}', item='{config.ItemColumn}', rating='{config.LabelColumn}'");

            // Build pipeline: MapValueToKey for user & item, then MatrixFactorization
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("UserKey", config.UserColumn)
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("ItemKey", config.ItemColumn))
                .Append(_mlContext.Recommendation().Trainers.MatrixFactorization(
                    labelColumnName: config.LabelColumn,
                    matrixColumnIndexColumnName: "UserKey",
                    matrixRowIndexColumnName: "ItemKey",
                    numberOfIterations: 20,
                    approximationRank: 32));

            progress?.Report(new TrainingProgress
            {
                TrialNumber = 1,
                TrainerName = "MatrixFactorization",
                MetricName = "rmse",
                Metric = 0,
                ElapsedSeconds = 0
            });

            var model = pipeline.Fit(trainSet);

            // Evaluate on test set
            var predictions = model.Transform(testSet);
            var regressionMetrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: config.LabelColumn);

            var metricsDict = new Dictionary<string, double>
            {
                ["rmse"] = double.IsNaN(regressionMetrics.RootMeanSquaredError) ? 0 : regressionMetrics.RootMeanSquaredError,
                ["mae"] = double.IsNaN(regressionMetrics.MeanAbsoluteError) ? 0 : regressionMetrics.MeanAbsoluteError,
                ["r_squared"] = double.IsNaN(regressionMetrics.RSquared) ? 0 : regressionMetrics.RSquared,
                ["loss_function"] = double.IsNaN(regressionMetrics.LossFunction) ? 0 : regressionMetrics.LossFunction
            };

            return new AutoMLResult
            {
                BestTrainer = "MatrixFactorization (rank=32, iter=20)",
                Model = model,
                Metrics = metricsDict,
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunImageClassificationAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            _logger.Info($"Image classification: label='{config.LabelColumn}', TensorFlow transfer learning");

            // The ImageClassification trainer requires raw image bytes as its feature
            // column. ImageDirectoryLoader produces an "ImagePath" string column, so
            // LoadRawImageBytes reads each file into a VarVector<byte> before fitting.
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", config.LabelColumn)
                .Append(_mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "ImageBytes", imageFolder: null, inputColumnName: "ImagePath"))
                .Append(_mlContext.MulticlassClassification.Trainers.ImageClassification(
                    featureColumnName: "ImageBytes", labelColumnName: "Label"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "ImageClassification (TF)", MetricName = "accuracy", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            return new AutoMLResult
            {
                BestTrainer = "ImageClassification (TensorFlow)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["accuracy"] = NanSafe(metrics.MacroAccuracy),
                    ["micro_accuracy"] = NanSafe(metrics.MicroAccuracy),
                    ["log_loss"] = NanSafe(metrics.LogLoss)
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunTextClassificationAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCol = FindFirstTextColumn(trainSet.Schema, config.LabelColumn)
                ?? throw new InvalidOperationException("No text column found for text classification.");

            _logger.Info($"Text classification: text='{textCol}', label='{config.LabelColumn}'");

            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", config.LabelColumn)
                .Append(_mlContext.MulticlassClassification.Trainers.TextClassification(
                    labelColumnName: "Label", sentence1ColumnName: textCol))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "TextClassification (NAS-BERT)", MetricName = "accuracy", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            return new AutoMLResult
            {
                BestTrainer = "TextClassification (NAS-BERT)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["accuracy"] = NanSafe(metrics.MacroAccuracy),
                    ["micro_accuracy"] = NanSafe(metrics.MicroAccuracy),
                    ["log_loss"] = NanSafe(metrics.LogLoss)
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunSentenceSimilarityAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCols = FindTextColumns(trainSet.Schema, config.LabelColumn, 2);
            if (textCols.Count < 2)
                throw new InvalidOperationException("Sentence similarity requires at least two text columns.");

            _logger.Info($"Sentence similarity: s1='{textCols[0]}', s2='{textCols[1]}', label='{config.LabelColumn}'");

            var pipeline = _mlContext.Regression.Trainers.SentenceSimilarity(
                labelColumnName: config.LabelColumn,
                sentence1ColumnName: textCols[0],
                sentence2ColumnName: textCols[1]);

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "SentenceSimilarity (NAS-BERT)", MetricName = "r_squared", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: config.LabelColumn);

            return new AutoMLResult
            {
                BestTrainer = "SentenceSimilarity (NAS-BERT)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["r_squared"] = NanSafe(metrics.RSquared),
                    ["rmse"] = NanSafe(metrics.RootMeanSquaredError),
                    ["mae"] = NanSafe(metrics.MeanAbsoluteError)
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunNerAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCol = FindFirstTextColumn(trainSet.Schema, config.LabelColumn)
                ?? throw new InvalidOperationException("No text column found for NER.");

            _logger.Info($"NER: text='{textCol}', label='{config.LabelColumn}'");

            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", config.LabelColumn)
                .Append(_mlContext.MulticlassClassification.Trainers.NamedEntityRecognition(
                    labelColumnName: "Label", sentence1ColumnName: textCol))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "NER (NAS-BERT)", MetricName = "accuracy", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            return new AutoMLResult
            {
                BestTrainer = "NER (NAS-BERT)",
                Model = model,
                Metrics = new Dictionary<string, double>
                {
                    ["accuracy"] = NanSafe(metrics.MacroAccuracy),
                    ["micro_accuracy"] = NanSafe(metrics.MicroAccuracy)
                },
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunObjectDetectionAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var labelColumn = string.IsNullOrWhiteSpace(config.LabelColumn)
                ? CocoDataLoader.DefaultLabelColumn
                : config.LabelColumn;

            _logger.Info($"Object detection: label='{labelColumn}', AutoFormerV2 transfer learning");

            // CocoDataLoader produces three columns: ImagePath (string), the label vector
            // (VBuffer<string>, one class name per object), and BoundingBoxes (VBuffer<float>,
            // four values per object in x0 y0 x1 y1 order). The AutoFormerV2 ObjectDetection
            // trainer requires the image as an MLImage, the label as a vector of keys, and the
            // bounding-box float vector as-is — so LoadImages converts the path and
            // MapValueToKey converts the label vector before fitting.
            var pipeline = _mlContext.Transforms.LoadImages(
                    outputColumnName: "Image", imageFolder: string.Empty, inputColumnName: CocoDataLoader.ImagePathColumn)
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "LabelKey", inputColumnName: labelColumn))
                .Append(_mlContext.MulticlassClassification.Trainers.ObjectDetection(
                    labelColumnName: "LabelKey",
                    boundingBoxColumnName: CocoDataLoader.BoundingBoxColumn,
                    imageColumnName: "Image"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "ObjectDetection (AutoFormerV2)", MetricName = "accuracy", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);
            var predictions = model.Transform(testSet);

            return new AutoMLResult
            {
                BestTrainer = "ObjectDetection (AutoFormerV2)",
                Model = model,
                Metrics = new Dictionary<string, double>(),
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<AutoMLResult> RunQuestionAnsweringAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        return await Task.Run(() =>
        {
            var textCols = FindTextColumns(trainSet.Schema, config.LabelColumn, 2);
            var contextCol = textCols.Count > 0 ? textCols[0] : throw new InvalidOperationException("No context column found.");
            var questionCol = textCols.Count > 1 ? textCols[1] : contextCol;

            _logger.Info($"Question answering: context='{contextCol}', question='{questionCol}', answer='{config.LabelColumn}'");

            var pipeline = _mlContext.MulticlassClassification.Trainers.QuestionAnswer(
                contextColumnName: contextCol,
                questionColumnName: questionCol);

            progress?.Report(new TrainingProgress { TrialNumber = 1, TrainerName = "QA (NAS-BERT)", MetricName = "accuracy", Metric = 0, ElapsedSeconds = 0 });

            var model = pipeline.Fit(trainSet);

            return new AutoMLResult
            {
                BestTrainer = "QA (NAS-BERT)",
                Model = model,
                Metrics = new Dictionary<string, double>(),
                RowCount = trainSet.GetRowCount() ?? 0
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private static double NanSafe(double value) => double.IsNaN(value) ? 0 : value;

    private static string? FindFirstTextColumn(DataViewSchema schema, string labelColumn)
    {
        foreach (var col in schema)
        {
            if (col.IsHidden) continue;
            if (col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase)) continue;
            if (col.Type is TextDataViewType)
                return col.Name;
        }
        return null;
    }

    private static List<string> FindTextColumns(DataViewSchema schema, string labelColumn, int maxCount)
    {
        var result = new List<string>();
        foreach (var col in schema)
        {
            if (col.IsHidden) continue;
            if (col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase)) continue;
            if (col.Type is TextDataViewType)
            {
                result.Add(col.Name);
                if (result.Count >= maxCount) break;
            }
        }
        return result;
    }

    private BinaryClassificationMetric GetBinaryMetric(string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "accuracy" => BinaryClassificationMetric.Accuracy,
            "auc" => BinaryClassificationMetric.AreaUnderRocCurve,
            "f1" or "f1_score" => BinaryClassificationMetric.F1Score,
            "auprc" => BinaryClassificationMetric.AreaUnderPrecisionRecallCurve,
            _ => BinaryClassificationMetric.Accuracy
        };
    }

    private MulticlassClassificationMetric GetMulticlassMetric(string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "accuracy" or "macro_accuracy" => MulticlassClassificationMetric.MacroAccuracy,
            "micro_accuracy" => MulticlassClassificationMetric.MicroAccuracy,
            "log_loss" => MulticlassClassificationMetric.LogLoss,
            _ => MulticlassClassificationMetric.MacroAccuracy
        };
    }

    private RegressionMetric GetRegressionMetric(string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "r_squared" or "r2" => RegressionMetric.RSquared,
            "rmse" => RegressionMetric.RootMeanSquaredError,
            "mae" => RegressionMetric.MeanAbsoluteError,
            "mse" => RegressionMetric.MeanSquaredError,
            _ => RegressionMetric.RSquared
        };
    }

    private double GetMetricValue(BinaryClassificationMetrics metrics, string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "accuracy" => metrics.Accuracy,
            "auc" => metrics.AreaUnderRocCurve,
            "f1" or "f1_score" => metrics.F1Score,
            _ => metrics.Accuracy
        };
    }

    private double GetMetricValue(MulticlassClassificationMetrics metrics, string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "accuracy" or "macro_accuracy" => metrics.MacroAccuracy,
            "micro_accuracy" => metrics.MicroAccuracy,
            "log_loss" => metrics.LogLoss,
            _ => metrics.MacroAccuracy
        };
    }

    private double GetMetricValue(RegressionMetrics metrics, string metricName)
    {
        return metricName.ToLowerInvariant() switch
        {
            "r_squared" or "r2" => metrics.RSquared,
            "rmse" => metrics.RootMeanSquaredError,
            "mae" => metrics.MeanAbsoluteError,
            "mse" => metrics.MeanSquaredError,
            _ => metrics.RSquared
        };
    }

    /// <summary>
    /// BUG-22/24: Check if exception is an AUC undefined error, handling both direct
    /// InvalidOperationException and AggregateException wrappers from AutoML's internal threading.
    /// Matches errors like "AUC is not defined when there is no positive class" and
    /// "AUC is not defined when there is no negative class".
    /// </summary>
    private static bool IsAucUndefinedException(Exception ex)
    {
        // Direct exception
        if (IsAucMessage(ex.Message))
            return true;

        // AggregateException from Task.Run or AutoML internal threading
        if (ex is AggregateException aggEx)
        {
            return aggEx.InnerExceptions.Any(inner => IsAucMessage(inner.Message));
        }

        // Nested inner exception
        if (ex.InnerException != null)
        {
            return IsAucMessage(ex.InnerException.Message);
        }

        return false;
    }

    /// <summary>
    /// BUG-25: Builds explicit ColumnInformation when text columns are present.
    /// ML.NET's InferColumns may classify text columns as Ignored (especially in text-only datasets),
    /// causing AutoML to generate an empty __Features__ pipeline and crash.
    /// Returns null if no text columns are found (existing behavior sufficient).
    /// </summary>
    public static ColumnInformation? BuildColumnInformation(
        IDataView data, string labelColumn,
        Action<string>? log = null,
        Dictionary<string, string>? columnOverrides = null)
    {
        var textColumns = new List<string>();
        var numericColumns = new List<string>();
        var categoricalColumns = new List<string>();
        var ignoredColumns = new List<string>();


        foreach (var col in data.Schema)
        {
            if (col.IsHidden) continue;
            if (col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase)) continue;

            // Check for column override (case-insensitive + trimmed for robustness)
            string? overrideType = null;
            if (columnOverrides != null)
            {
                // Try exact match first
                if (!columnOverrides.TryGetValue(col.Name, out overrideType))
                {
                    // Try trimmed and case-insensitive match
                    var colNameTrimmed = col.Name.Trim();
                    var match = columnOverrides.Keys.FirstOrDefault(k =>
                        k.Trim().Equals(colNameTrimmed, StringComparison.OrdinalIgnoreCase));
                    if (match != null)
                        overrideType = columnOverrides[match];
                }
            }
            if (overrideType != null)
            {
                switch (overrideType.ToLowerInvariant())
                {
                    case "text":
                        textColumns.Add(col.Name);
                        continue;
                    case "categorical":
                        categoricalColumns.Add(col.Name);
                        continue;
                    case "numeric":
                        numericColumns.Add(col.Name);
                        continue;
                    case "ignore":
                        ignoredColumns.Add(col.Name);
                        continue;
                }
            }

            // Default: infer from IDataView type
            if (col.Type is TextDataViewType)
            {
                textColumns.Add(col.Name);
            }
            else if (col.Type is NumberDataViewType || col.Type is BooleanDataViewType
                || (col.Type is VectorDataViewType vt2 && vt2.ItemType is NumberDataViewType))
            {
                numericColumns.Add(col.Name);
            }
        }

        // If no text columns and no overrides applied, return null (default behavior)
        if (textColumns.Count == 0 && categoricalColumns.Count == 0 && ignoredColumns.Count == 0)
            return null;

        var columnInfo = new ColumnInformation { LabelColumnName = labelColumn };

        foreach (var tc in textColumns)
            columnInfo.TextColumnNames.Add(tc);

        foreach (var nc in numericColumns)
            columnInfo.NumericColumnNames.Add(nc);

        foreach (var cc in categoricalColumns)
            columnInfo.CategoricalColumnNames.Add(cc);

        foreach (var ic in ignoredColumns)
            columnInfo.IgnoredColumnNames.Add(ic);

        var write = log ?? Console.WriteLine;
        if (textColumns.Count > 0)
            write($"ℹ️  [Info] Text column(s): {string.Join(", ", textColumns)} — applying text featurization (TF-IDF, n-gram)");
        if (categoricalColumns.Count > 0)
            write($"ℹ️  [Info] Categorical column(s) (override): {string.Join(", ", categoricalColumns)}");
        if (ignoredColumns.Count > 0)
            write($"[Info] Ignored column(s) (override): {string.Join(", ", ignoredColumns)}");

        return columnInfo;
    }

    /// <summary>
    /// Gets row count from IDataView, counting manually if lazy view returns null.
    /// </summary>
    private static long CountRows(IDataView data)
    {
        var count = data.GetRowCount();
        if (count.HasValue) return count.Value;

        // Manual count for lazy IDataView (TextLoader)
        long rows = 0;
        using var cursor = data.GetRowCursor(Array.Empty<DataViewSchema.Column>());
        while (cursor.MoveNext()) rows++;
        return rows;
    }

    private static bool IsAucMessage(string message)
    {
        return message.Contains("AUC") ||
               message.Contains("positive class") ||
               message.Contains("negative class") ||
               message.Contains("PosSample") ||
               message.Contains("NegSample");
    }
}

/// <summary>
/// Input schema for time series prediction engine
/// </summary>
public class ForecastInput
{
    public float Value { get; set; }
}

/// <summary>
/// Output schema for time series prediction engine
/// </summary>
public class ForecastOutput
{
    public float[] ForecastedValues { get; set; } = [];
    public float[] LowerBound { get; set; } = [];
    public float[] UpperBound { get; set; } = [];
}

/// <summary>
/// Result from AutoML execution
/// </summary>
public class AutoMLResult
{
    public required string BestTrainer { get; init; }
    public required ITransformer Model { get; init; }
    public required Dictionary<string, double> Metrics { get; init; }

    /// <summary>
    /// Number of rows in the training dataset (for memory collection)
    /// </summary>
    public long RowCount { get; init; }

    /// <summary>
    /// Input schema captured during training — required for prediction.
    /// Contains column names, data types, purposes, and categorical value lists.
    /// </summary>
    public InputSchemaInfo? Schema { get; init; }

    /// <summary>
    /// ② regression wave (heteroscedastic): the auxiliary σ(x) model that predicts residual magnitude
    /// from the features, used to widen/narrow the conformal band per row. Null for non-regression
    /// tasks, homoscedastic fallbacks, or when the holdout was too small to fit one — callers then use
    /// the constant-width band. Persisted alongside the main model (see <c>ResidualModelFileName</c>).
    /// </summary>
    public ITransformer? ResidualModel { get; init; }
}

/// <summary>
/// ② regression wave (heteroscedastic): output of
/// <see cref="AutoMLRunner.ComputeNormalizedConformal"/> — the fitted auxiliary σ(x) model plus the
/// normalized-conformal metrics (<c>norm_interval_q_{pct}</c>, <c>interval_beta</c>) to merge into the
/// regression metrics dict. Predict-time width for a row = <c>q · (max(σ(x),0) + β)</c>.
/// </summary>
public sealed record NormalizedConformalResult(ITransformer AuxModel, Dictionary<string, double> Metrics);

/// <summary>CustomMapping input for computing the conformal residual target (binds by field name).</summary>
internal sealed class ConformalLabelScore
{
    public float Label { get; set; }
    public float Score { get; set; }
}

/// <summary>CustomMapping output carrying the absolute residual as the σ-model's regression target.</summary>
internal sealed class ConformalResidual
{
    public float Target { get; set; }
}

/// <summary>
/// Simple console logger implementation for preprocessing scripts
/// </summary>
internal class ConsoleLogger : ILogger
{
    public void Info(string message) => Console.WriteLine($"ℹ️  {message}");
    public void Warning(string message) => Console.WriteLine($"⚠️  {message}");
    public void Error(string message) => Console.WriteLine($"❌ {message}");
    public void Error(string message, Exception exception) => Console.WriteLine($"❌ {message}{Environment.NewLine}{exception}");
    public void Debug(string message) => Console.WriteLine($"🔍 {message}");
}
