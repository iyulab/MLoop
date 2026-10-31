using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Detection;
using MLoop.Core.Diagnostics;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;
using MLoop.Core.Scripting;
using MLoop.Extensibility;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Preprocessing;
using MLoop.Search;
using MLoop.Search.Candidates;
using MLoop.Search.Validation;

namespace MLoop.Core.Training;

/// <summary>
/// Training runner: the task pipelines — MLoop's trainer search, the deep-learning and time-series paths — with extensibility support
/// </summary>
public partial class TrainingRunner
{
    private readonly MLContext _mlContext;
    private readonly IDataProvider _dataLoader;
    private readonly SinkLogger _logger;

    // What a run has already said. A run can try more than one search (another metric, then a
    // direct pipeline) and each one states the same column roles; saying it once is enough.
    private readonly HashSet<string> _saidThisRun = new(StringComparer.Ordinal);

    /// <param name="mlContext">ML.NET context shared with the loader and the fitted pipelines.</param>
    /// <param name="dataLoader">Format-specific loader for the task's data.</param>
    /// <param name="warningSink">
    /// Where this runner's warnings go. Supplied by a host that has a warning channel of its own —
    /// the CLI routes them through its warning seam, which renders a <c>Warning:</c> line and, in
    /// machine mode, emits a <c>warning</c> event.
    /// </param>
    /// <param name="narrationSink">
    /// Where this runner's info and debug lines go. The CLI points this at its own console, so the
    /// scope a <c>--json</c> command installs governs them like every other line it writes.
    /// </param>
    /// <remarks>
    /// Both channels are injected and neither has a console default. An earlier version injected only
    /// the warning channel, on the stated ground that narration "has no host channel to reach because
    /// machine mode discards it" — which was true of <c>train</c>'s event stream and of nothing else:
    /// the scope the <c>--json</c> commands install redirects the console it owns and forwards
    /// stdout, so a library writing to <see cref="Console"/> lands <i>ahead of the document</i>
    /// rather than being discarded. Measured: <c>mloop info --json</c> emitting
    /// <c>[Info] Removed index column(s): …</c> on stdout, exit 0, unparseable.
    /// </remarks>
    public TrainingRunner(
        MLContext mlContext,
        IDataProvider dataLoader,
        Action<string>? warningSink = null,
        Action<string>? narrationSink = null)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _dataLoader = dataLoader ?? throw new ArgumentNullException(nameof(dataLoader));
        _logger = new SinkLogger(warningSink, narrationSink);
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
        return Models.TaskTypes.Canonical(task) switch
        {
            "binary-classification" => true,
            "multiclass-classification" => true,
            "regression" => true,
            _ => false
        };
    }

    /// <summary>
    /// Single source of truth for whether a task predicts a discrete class.
    /// </summary>
    /// <remarks>
    /// <para>This question had seven inline answers and they fell into two groups, because two
    /// different questions were being asked under one name. One group meant "does this model
    /// predict a class" — that is what decides whether a text label column is legitimate, whether
    /// a dummy label has to be injected for the saved transform, and which extractor reads the
    /// prediction rows. The other meant "can I count and stratify on a class column in a CSV",
    /// which is what the stratified split, the class-count time estimate, the per-class minimum
    /// sample check and <c>--balance</c> need.</para>
    /// <para>The second question is this one composed with
    /// <c>DataLoaderFactory.IsDirectoryBased</c>, which already owns "where does the label come
    /// from": a classification task whose labels are directory names has no column to count. So
    /// there is one new predicate here rather than two, and each call site says which question it
    /// is asking by whether it composes.</para>
    /// <para>The groups had drifted apart on <c>text-classification</c>, which reads a CSV with a
    /// label column like any other: it was in the first group everywhere and in the second group
    /// nowhere, so a text classifier got an unstratified split, no class-count estimate and no
    /// per-class minimum check.</para>
    /// </remarks>
    /// <summary>
    /// The rows a view holds. A view read from a file does not know its length
    /// (<see cref="IDataView.GetRowCount"/> is null), so it is counted with a cursor that reads no column.
    /// </summary>
    internal static long RowsIn(IDataView data)
    {
        if (data.GetRowCount() is { } known)
            return known;
        long rows = 0;
        using var cursor = data.GetRowCursor(Array.Empty<DataViewSchema.Column>());
        while (cursor.MoveNext())
            rows++;
        return rows;
    }

    public static bool IsClassification(string? task) =>
        Models.TaskTypes.Canonical(task)
            is "binary-classification"
            or "multiclass-classification"
            or "text-classification"
            or "image-classification";

    /// <summary>
    /// Single source of truth for whether a task requires a label column. The unsupervised tasks
    /// (anomaly-detection, clustering, time-series-anomaly) are label-optional: when no label is
    /// configured, <see cref="Data.CsvDataLoader"/> loads the file without one and treats every
    /// column as a feature. This set was duplicated as inline HashSets in <c>ConfigMerger</c>, <c>InitCommand</c>
    /// and <c>TrainCommand</c> — and had already drifted (TrainCommand omitted time-series-anomaly,
    /// and ValidateCommand was missing the concept entirely, so it errored "Label required" on a valid
    /// unsupervised project that <c>train</c> accepts). All four now read this.
    /// Unknown tasks return <c>true</c> (require a label) — the conservative default.
    /// </summary>
    public static bool RequiresLabel(string? task)
    {
        // An unrecognized task canonicalizes to null, which is not in the unsupervised set, so it
        // still requires a label — the conservative default this has always had.
        return Models.TaskTypes.Canonical(task) is not ("anomaly-detection" or "clustering" or "time-series-anomaly" or "image-anomaly");
    }

    /// <summary>
    /// Whether a task's label is text that is neither a class nor a number: NER labels every word
    /// of a sentence (one tag per word, space-separated), question answering labels with the text
    /// of the answer. Checks written for the other two kinds — a numeric label's variance, a class
    /// column's balance — do not apply, and "not a classification task, so the label must be a
    /// number" refused every NER set before its trainer was reached.
    /// </summary>
    public static bool HasTextLabel(string? task) =>
        Models.TaskTypes.Canonical(task) is "ner" or "question-answering";

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
        return Models.TaskTypes.Canonical(task) is "forecasting" or "time-series-anomaly";
    }

    /// <summary>
    /// Whether <paramref name="task"/>'s model reads the table's other columns as features. A series model
    /// reads its one series column, matrix factorization its user and item, and a deep-learning trainer the
    /// text or images it is pointed at — for those, advice drawn from the column count (samples per
    /// feature, which columns look like an index) counts columns the model never sees. Measured: a series
    /// of 300 readings was warned it had "300 samples for 35 features".
    /// </summary>
    public static bool ReadsFeatureColumns(string? task) =>
        !DeepLearningRegistry.Handles(task)
        && !IsTimeSeriesTask(task)
        && Models.TaskTypes.Canonical(task) is not ("recommendation" or "image-anomaly");

    /// <summary>
    /// Why <paramref name="task"/>'s fit takes no time budget, or <c>null</c> when it does. Only the trainer search
    /// (binary, multiclass, regression) spends a budget; a deep-learning fit runs its epochs (a set number, or up to a
    /// cap with early stopping), a time-series fit runs its algorithm over the series once, matrix factorization
    /// (recommendation) runs a fixed number of iterations, and anomaly detection, clustering and ranking fit one
    /// pipeline. For those the auto-time probe would be a whole second training whose result is thrown away, its
    /// estimate would describe time nothing uses, and a "Time Limit" row would promise a bound nothing enforces —
    /// measured: a 2-second time-series fit was reported as a 19-second probe and a 114-second main pass, its
    /// warnings printed twice.
    /// </summary>
    public static string? TimeBudgetUnused(string? task) =>
        DeepLearningRegistry.Handles(task) ? "trains by epochs, not against a clock"
        : IsTimeSeriesTask(task) ? "fits the series once, without a search"
        : Models.TaskTypes.Canonical(task) switch
        {
            "recommendation" => "trains a fixed number of iterations",
            "image-anomaly" => "keeps the patches of its images, without a search",
            "anomaly-detection" or "clustering" or "ranking" => "fits one pipeline, without a search",
            _ => null,
        };

    /// <summary>
    /// Why <paramref name="task"/> has no epoch count to set, or <c>null</c> when it has one. The tasks that
    /// do are the ones whose fit runs a fixed number of epochs (<c>EpochProgress</c>) — image classification
    /// stops on its own once accuracy stops improving, and a tabular task searches trainers instead.
    /// </summary>
    public static string? EpochsUnused(string? task) =>
        Models.TaskTypes.Canonical(task) switch
        {
            "text-classification" or "sentence-similarity" or "ner" or "question-answering" or "object-detection" => null,
            "image-classification" => "stops early on its own once accuracy stops improving",
            _ => "does not train by epochs",
        };

    public async Task<TrainedModel> RunAsync(
        TrainingConfig config,
        IProgress<TrainingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        lock (_saidThisRun)
            _saidThisRun.Clear();

        // A folder of normal images, remembered as patch vectors: no table to load, split or search.
        if (Models.TaskTypes.Canonical(config.Task) == "image-anomaly")
            return await RunImageAnomalyAsync(config, cancellationToken).ConfigureAwait(false);

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
        // The weight is read by name to resample the rows; merged into the feature vector it is gone.
        if (!string.IsNullOrEmpty(config.WeightColumn)) preserveColumns.Add(config.WeightColumn);
        // A column declared `ignore` is kept out of the features by its role, and a role names a column of the
        // loaded view. A numeric one is inside the merged Features vector — no column of its name — so the
        // model read it, while the saved schema (which marks it Exclude) had evaluate and predict drop it
        // from every file: a vector one slot narrower than the model's.
        if (config.ColumnOverrides is { Count: > 0 } overrides)
            preserveColumns.AddRange(overrides
                .Where(kv => string.Equals(kv.Value?.Trim(), "ignore", StringComparison.OrdinalIgnoreCase))
                .Select(kv => kv.Key));
        var preserve = preserveColumns.Count > 0 ? preserveColumns : null;

        // Directory-based tasks (image classification) need a different loader than the
        // injected CSV loader. The factory keeps the tabular path byte-for-byte unchanged.
        var loader = DataLoaderFactory.IsDirectoryBased(config.Task)
            ? DataLoaderFactory.Create(config.Task, _mlContext, _logger.Info)
            : _dataLoader;

        IReadOnlyDictionary<string, string[]>? mergedColumnGroups;
        if (!string.IsNullOrEmpty(config.TestDataFile))
        {
            // Pre-split data (e.g. stratified classification split, or balanced training with a
            // separate test set). Both files must be featurized identically, so the exclusion set is
            // decided once — normally upstream by the caller, and from the training file here when
            // the caller made no decision. Letting each file decide for itself is what produced the
            // "expected Vector<Single, N>, got Vector<Single, N-1>" schema mismatch: a column that is
            // constant only inside one partition silently narrows that partition alone.
            var featureExclusions = config.FeatureExclusions;
            if (featureExclusions is null && !DataLoaderFactory.IsDirectoryBased(config.Task))
            {
                featureExclusions = CsvDataLoader
                    .DetermineExcludedColumns(config.DataFile, config.LabelColumn, _logger.Info, config.ClaimedColumns)
                    .Select(c => c.Name).ToList();
            }

            trainSet = loader.LoadData(config.DataFile, config.LabelColumn, config.Task, preserve, featureExclusions, config.LabelKind);
            // Grab the training file's merge mapping before the test-file load overwrites it
            mergedColumnGroups = loader.GetMergedColumnGroups();
            testSet = loader.LoadData(config.TestDataFile, config.LabelColumn, config.Task, preserve, featureExclusions, config.LabelKind);
        }
        else if (isTimeSeriesTask)
        {
            // Time series: use full dataset (no random split — temporal order matters)
            // Forecasting/TS-Anomaly handle holdout internally
            var dataView = loader.LoadData(config.DataFile, config.LabelColumn, config.Task, preserve, config.FeatureExclusions, config.LabelKind);
            trainSet = dataView;
            testSet = dataView; // same data — internal holdout handles evaluation
            mergedColumnGroups = loader.GetMergedColumnGroups();
        }
        else
        {
            // Single file split in-memory: both partitions come from one load, so they share one
            // featurization by construction. The exclusion set still flows through so the saved
            // schema predict replays describes exactly what was fitted.
            var dataView = loader.LoadData(config.DataFile, config.LabelColumn, config.Task, preserve, config.FeatureExclusions, config.LabelKind);
            (trainSet, testSet) = loader.SplitData(dataView, config.TestSplit);
            mergedColumnGroups = loader.GetMergedColumnGroups();
        }

        // Weighted training: the search gives the trainers the weights and scores its folds by them.
        if (!string.IsNullOrEmpty(config.WeightColumn))
            CheckWeightedTask(testSet, config);

        // Text read through an encoder: each such column gets its sentence vector beside it, embedded once for both
        // partitions and kept in memory — a numeric vector column the trainers read with the text's own featurization.
        // The saved schema is the data's, since that is what a prediction is given.
        var schemaSource = trainSet;
        TextEncodings? textEncodings = null;
        if (config.TextEncoders is { Count: > 0 } encoders)
        {
            if (TextEncodings.Unsupported(config.Task) is { } unsupported)
                throw new InvalidOperationException(unsupported);
            var opened = new List<(string, Embedding.TextEmbedder)>();
            foreach (var (column, id) in encoders)
            {
                if (trainSet.Schema.GetColumnOrNull(column) is not { Type: TextDataViewType })
                    throw new InvalidOperationException($"Column '{column}' is declared to be read through text encoder '{id}', but the data has no text column of that name.");
                var (embedder, refusal) = Embedding.EmbedderPool.Shared.GetText(id);
                opened.Add((column, embedder ?? throw new InvalidOperationException($"Column '{column}': {refusal}")));
            }
            var layer = new TextEncodingLayer(_mlContext, opened);
            _logger.Info($"Embedding {string.Join(", ", encoders.Select(e => $"'{e.Key}' with {e.Value}"))}...");
            trainSet = _mlContext.Data.Cache(layer.Transform(trainSet));
            testSet = ReferenceEquals(testSet, schemaSource) ? trainSet : _mlContext.Data.Cache(layer.Transform(testSet));
            textEncodings = layer.Encodings;
        }

        // Ensure the task's on-demand native runtime is installed and loaded (no-op for tabular tasks).
        Runtime.RuntimeManager.EnsureRuntimeForTask(config.Task);

        // Run the task's training path
        var result = config.Task.ToLowerInvariant() switch
        {
            "binary-classification" => await RunBinaryClassificationAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "multiclass-classification" => await RunMulticlassClassificationAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "regression" => await RunRegressionAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "anomaly-detection" => await RunAnomalyDetectionAsync(
                trainSet, testSet, config, progress, cancellationToken, mergedColumnGroups).ConfigureAwait(false),
            "clustering" => await RunClusteringAsync(
                trainSet, testSet, config, progress, cancellationToken, mergedColumnGroups).ConfigureAwait(false),
            "ranking" => await RunRankingAsync(
                trainSet, testSet, config, progress, cancellationToken, mergedColumnGroups).ConfigureAwait(false),
            "forecasting" => await RunForecastingAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "time-series-anomaly" => await RunTimeSeriesAnomalyAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            "recommendation" => await RunRecommendationAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false),
            _ => await RunDeepLearningOrThrowAsync(
                trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false)
        };

        // Single-authority guard: replace any non-finite metric (NaN/±∞ — e.g. R² on a
        // near-zero-variance test split, or log-loss when a class probability underflows to 0)
        // with the direction-aware worst value, so a broken measurement can never win selection,
        // pass the quality gate, or crash metrics.json serialization. Runs before hooks/persistence
        // so every downstream consumer of result.Metrics sees only finite values.
        if (!string.IsNullOrEmpty(config.WeightColumn))
            ReplaceWithWeightedMetrics(result, testSet, config);

        var undefinedMetrics = Evaluation.MetricSanitizer.SanitizeInPlace(result.Metrics);
        if (undefinedMetrics.Count > 0)
            _logger.Warning(
                $"{undefinedMetrics.Count} metric(s) were undefined (NaN/Infinity), most likely because the " +
                $"test split has too few samples (near-zero variance). Recorded as worst-case so they cannot be " +
                $"mistaken for a good model: {string.Join(", ", undefinedMetrics)}. Provide more data for reliable metrics.");

        // Capture input schema from training data for prediction use.
        // `with` rather than a hand-written copy: this rewrap used to list every property, so each
        // field added to TrainedModel was silently dropped here until someone noticed — the σ(x)
        // residual model needed an explicit line, and the trial history was lost the same way until
        // the empty leaderboard gave it away. A copy that names nothing cannot forget anything.
        if (result.Schema == null)
        {
            result = result with
            {
                Schema = CaptureInputSchema(schemaSource, config.LabelColumn, mergedColumnGroups)
            };
        }

        return result with { EvaluationData = result.EvaluationData ?? testSet, TextEncodings = textEncodings };
    }

    /// <summary>
    /// The image-anomaly task: the patch vectors of the normal images in <c>config.DataFile</c> as a bank, and how far held-out
    /// normal images score against it. The saved <c>model.zip</c> is an empty transformer chain — the experiment's header;
    /// the model is the bank beside it (<see cref="ImageAnomaly.ImageAnomalyArtifacts.WriteBeside"/>).
    /// </summary>
    private Task<TrainedModel> RunImageAnomalyAsync(TrainingConfig config, CancellationToken cancellationToken) =>
        Task.Run(() =>
        {
            var (artifacts, metrics) = ImageAnomaly.ImageAnomalyTrainer.Train(
                config.DataFile, config.ImageEncoder, config.BankSize, _logger.Info, cancellationToken: cancellationToken);
            return new TrainedModel
            {
                Trainer = TrainerDescriptor.Of("PatchBank",
                    ("encoder", artifacts.Model.Encoder.Id), ("patches", artifacts.Bank.Count.ToString(System.Globalization.CultureInfo.InvariantCulture))),
                Model = new TransformerChain<ITransformer>(),
                Metrics = metrics,
                RowCount = metrics["images"] is var images ? (long)images : 0,
                Schema = new InputSchemaInfo { Columns = [], CapturedAt = DateTime.UtcNow },
                ImageAnomaly = artifacts,
            };
        }, cancellationToken);

    /// <summary>
    /// Dispatches deep-learning tasks (image-classification, text-classification,
    /// sentence-similarity, ner, object-detection, question-answering) to the optional
    /// <see cref="DeepLearningRegistry.Current"/> module. MLoop.Core no longer references the
    /// TorchSharp/Vision-backed handlers directly — they live in
    /// MLoop.Core.DeepLearning, and consumers register the module at startup. If no module is
    /// registered (or it doesn't recognize the task), this throws an actionable error rather
    /// than silently failing.
    /// </summary>
    private async Task<TrainedModel> RunDeepLearningOrThrowAsync(
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken)
    {
        var module = DeepLearningRegistry.Current;
        if (module is null || !module.CanHandleTask(config.Task))
            throw new NotSupportedException(
                $"Task type '{config.Task}' is not supported. " +
                "Deep-learning tasks (image-classification, text-classification, sentence-similarity, " +
                "ner, object-detection, question-answering) require the MLoop.Core.DeepLearning package.");
        return await module.TrainAsync(
            _mlContext, msg => _logger.Info(msg), config.Task,
            trainSet, testSet, config, progress, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Captures the input schema from the training IDataView for later use in prediction.
    /// Emits the <see cref="SchemaDataTypes"/> semantic vocabulary — the same names the CLI
    /// schema writer produces and PredictionService/CsvDataLoader consume
    /// (raw .NET type names made every numeric feature fall through to DataKind.String).
    /// </summary>
    private InputSchemaInfo CaptureInputSchema(
        IDataView trainSet,
        string labelColumn,
        IReadOnlyDictionary<string, string[]>? mergedColumnGroups)
    {
        var columns = new List<ColumnSchema>();
        var schema = trainSet.Schema;

        foreach (var col in schema)
        {
            // Skip internal ML.NET columns
            if (col.IsHidden) continue;

            // InferColumns merges adjacent same-kind CSV columns into one ranged vector
            // (typically "Features") and the loaded view carries no slot names for it.
            // Capture the constituent named columns via the loader's merge mapping —
            // row-based predict reconstructs the vector from named columns
            // (EnsureFeaturesColumn) but cannot do the reverse.
            if (col.Type is VectorDataViewType vectorType)
            {
                var slotNames = mergedColumnGroups is not null
                        && mergedColumnGroups.TryGetValue(col.Name, out var groupNames)
                    ? groupNames
                    : GetSlotNames(col);
                if (slotNames.Length == vectorType.Size && slotNames.Length > 0)
                {
                    var slotDataType = SchemaDataTypes.FromRawType(vectorType.ItemType.RawType);
                    foreach (var slot in slotNames)
                    {
                        columns.Add(new ColumnSchema
                        {
                            Name = slot,
                            DataType = slotDataType,
                            Purpose = slot.Equals(labelColumn, StringComparison.OrdinalIgnoreCase)
                                ? "Label"
                                : "Feature",
                        });
                    }
                }
                else
                {
                    // No usable slot names (e.g. in-memory vector property) — record the vector
                    // column by its item type so the vocabulary stays semantic, even though
                    // row-based predict cannot reconstruct it per-column.
                    columns.Add(new ColumnSchema
                    {
                        Name = col.Name,
                        DataType = SchemaDataTypes.FromRawType(vectorType.ItemType.RawType),
                        Purpose = "Feature",
                    });
                }
                continue;
            }

            var purpose = col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase)
                ? "Label"
                : "Feature";

            var dataType = SchemaDataTypes.FromRawType(col.Type.RawType);

            // Capture categorical values for string columns (important for prediction dimension matching)
            List<string>? categoricalValues = null;
            int? uniqueValueCount = null;

            if (dataType == SchemaDataTypes.Text && purpose == "Feature")
            {
                try
                {
                    var colData = trainSet.GetColumn<string>(col.Name).Where(v => !string.IsNullOrEmpty(v)).Distinct().ToList();
                    uniqueValueCount = colData.Count;
                    if (colData.Count <= 10000) // reasonable limit
                    {
                        categoricalValues = colData;
                        dataType = SchemaDataTypes.Categorical;
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

    private static string[] GetSlotNames(DataViewSchema.Column col)
    {
        if (col.Annotations.Schema.GetColumnOrNull("SlotNames") is null)
            return [];

        var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
        col.Annotations.GetValue("SlotNames", ref slotNames);
        return slotNames.DenseValues().Select(v => v.ToString()).ToArray();
    }

    private static void CheckWeightedTask(IDataView testSet, TrainingConfig config)
    {
        if (!RowWeights.Supports(config.Task))
            throw new NotSupportedException(
                $"Weighted training (weight column '{config.WeightColumn}') applies to binary-classification, " +
                $"multiclass-classification and regression; '{config.Task}' would ignore the weights. Remove the weight to train it.");

        if (testSet.Schema.GetColumnOrNull(config.WeightColumn!) is null)
            throw new InvalidOperationException(
                $"The test data has no weight column '{config.WeightColumn}'; a weighted model is judged on weighted test rows.");
    }

    /// <summary>
    /// Re-scores the test partition with the final model and replaces the task's evaluator metrics
    /// with their weighted counterparts — the whole family, so a weighted value is never reported
    /// next to an unweighted one under the same name. Descriptive extras (prediction-interval widths)
    /// stay as computed.
    /// </summary>
    private void ReplaceWithWeightedMetrics(TrainedModel result, IDataView testSet, TrainingConfig config)
    {
        OverlayWeighted(result.Metrics, result.ScoringModel(_mlContext).Transform(testSet), config.Task, config.LabelColumn, config.WeightColumn!);
    }

    /// <summary>
    /// Replaces the task's evaluator metrics in <paramref name="metrics"/> with their counterparts
    /// weighted by <paramref name="weight"/> — the test partition's and, on MLoop's search, each fold's.
    /// </summary>
    private static void OverlayWeighted(Dictionary<string, double> metrics, IDataView predictions, string task, string label, string weight)
    {
        var (family, weighted) = task.ToLowerInvariant() switch
        {
            "binary-classification" => (WeightedBinaryKeys, WeightedMetrics.Binary(predictions, label, weight)),
            "multiclass-classification" => (WeightedMulticlassKeys, WeightedMetrics.Multiclass(predictions, label, weight)),
            _ => (WeightedRegressionKeys, WeightedMetrics.Regression(predictions, label, weight)),
        };
        if (weighted is null)
            throw new InvalidOperationException(
                $"The predictions could not be scored by weight '{weight}' — the columns the {task} metrics need are missing.");

        foreach (var key in family)
            metrics.Remove(key);
        foreach (var (key, value) in weighted)
            metrics[key] = value;
    }

    private static readonly string[] WeightedBinaryKeys =
        ["accuracy", "precision", "recall", "negative_precision", "negative_recall", "f1_score", "auc", "auprc", "ece"];
    private static readonly string[] WeightedMulticlassKeys =
        ["micro_accuracy", "macro_accuracy", "log_loss", "log_loss_reduction", "macro_f1", "ece"];
    private static readonly string[] WeightedRegressionKeys = ["r_squared", "rmse", "mae", "mse"];

    private Task<TrainedModel> RunBinaryClassificationAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
        => RunWithMLoopSearchAsync(trainSet, testSet, config,
            CandidateCatalog.Binary(name => name != "LightGbm" || LightGbmLoads.Value),
            OptimizingMetric(config), ScoreBinaryWithCalibrationError, progress, cancellationToken,
            (model, predictions, outOfFold) => FinishBinary(model, trainSet, testSet, predictions, outOfFold, config));

    /// <summary>What a calibration fitted on the search's out-of-fold answers records as the rows it was fitted on.</summary>
    public const string OutOfFoldSource = "out-of-fold (training partition)";

    /// <summary>
    /// A binary model's saved form and test results. Its probabilities are recalibrated on the kept trial's
    /// out-of-fold answers (<see cref="OutOfFoldScores"/>): a trainer's own probability is not fitted to the data —
    /// FastTree's is a fixed sigmoid of twice its learning rate times the score, so a few shallow trees that rank
    /// every test row right say 0.66 where they should say 0.99 — and a confidence a caller routes on has to mean
    /// how often the answer is right. A Platt fit on the logit (<see cref="ProbabilityCalibration"/>, beside the
    /// model) maps one to the other, keeps the order of the rows (AUC unchanged), and answers at its own 0.5 — so
    /// the answer-based metrics are counted from the answers it gives, as <c>evaluate</c> counts them. A trainer
    /// with no probability gets a Platt calibrator on its score, fitted on the same out-of-fold rows. The test
    /// partition is the run's report: nothing is fitted on it — when the kept trial left no out-of-fold answers
    /// (a part that could not be read), a trainer with no probability is calibrated on the training rows instead,
    /// with a warning, rather than on the rows the run reports on.
    /// </summary>
    private (ITransformer Model, Dictionary<string, double> Metrics, ITransformer? ResidualModel, ProbabilityCalibration? Calibration) FinishBinary(
        ITransformer model, IDataView trainSet, IDataView testSet, IDataView predictions, OutOfFoldScores? outOfFold, TrainingConfig config)
    {
        if (predictions.Schema.GetColumnOrNull("Probability") is null)
        {
            model = outOfFold is null
                ? EnsureCalibratedModel(_mlContext, model, model.Transform(trainSet), config.LabelColumn, hasProbability: false)
                : model.Append(_mlContext.BinaryClassification.Calibrators.Platt(labelColumnName: "Label", scoreColumnName: "Score")
                    .Fit(_mlContext.Data.LoadFromEnumerable(Enumerable.Range(0, outOfFold.Count)
                        .Select(i => new OutOfFoldScore { Label = outOfFold.Label[i], Score = outOfFold.Score[i] }))));
            if (outOfFold is null)
                _logger.Warning("No out-of-fold answers were kept for the chosen trial; its probability calibrator was fitted on the training rows, which the model has seen — its probabilities are likelier to be over-confident than a calibrator fitted out of fold.");
            return (model, ScoreBinaryWithCalibrationError(model.Transform(testSet), config.LabelColumn), null, null);
        }

        var calibration = outOfFold?.Probability is { } probability
            ? ProbabilityCalibration.Fit(
                [.. Enumerable.Range(0, outOfFold.Count).Select(i => ((double)probability[i], outOfFold.Label[i], (double)outOfFold.Weight[i]))],
                "platt", OutOfFoldSource, string.IsNullOrEmpty(config.WeightColumn) ? null : config.WeightColumn, DateTimeOffset.UtcNow)
            : null;
        if (calibration is null)
            return (model, ScoreBinaryWithCalibrationError(predictions, config.LabelColumn), null, null);

        var calibrated = calibration.Apply(_mlContext, model).Transform(testSet);
        var metrics = ScoreBinaryWithCalibrationError(calibrated, config.LabelColumn);
        // ML.NET's binary evaluator re-derives each answer from the raw score at 0; the calibrated model answers at
        // its own 0.5, so the metrics that depend on the answer are counted from the answers it gives.
        if (BinaryCounts.Metrics(calibrated, config.LabelColumn) is { } answered)
            BinaryCounts.ReplaceAnswerMetrics(metrics, answered);
        return (model, metrics, null, calibration);
    }

    private sealed class OutOfFoldScore
    {
        public bool Label { get; set; }
        public float Score { get; set; }
    }
    /// <summary>
    /// A binary, multiclass or regression run on MLoop's trainer search (<see cref="TrialRunner"/>): column roles,
    /// portable-text step, the search — candidates, tuner, folds, refit — the test evaluation and the trial
    /// records. <paramref name="score"/> gives the metrics of scored rows, for each fold (weighted when the run
    /// is) and for the test partition; <paramref name="finish"/>, when given, turns the refit model and its test
    /// predictions into what is saved (a binary calibrator, a regression's intervals and σ-model).
    /// </summary>
    private async Task<TrainedModel> RunWithMLoopSearchAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IReadOnlyList<TrainerCandidate> candidates,
        string metric,
        Func<IDataView, string, Dictionary<string, double>> score,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken,
        Func<ITransformer, IDataView, OutOfFoldScores?, (ITransformer Model, Dictionary<string, double> Metrics, ITransformer? ResidualModel, ProbabilityCalibration? Calibration)>? finish = null)
    {
        var roles = BuildColumnRoles(trainSet, config.LabelColumn, InfoOnce, config.ColumnOverrides, config.CategoryColumns);
        ITransformer? portableText = null;
        if (config.Portable)
            (roles, portableText, trainSet, _) = PortableTextFeaturizer.FitAndApply(_mlContext, roles, trainSet, null);

        var channel = CreateTrialChannel(progress);
        var request = new SearchRequest(
            candidates,
            RankingPolicy.For(metric),
            scored => ScoreFold(scored, config, score),
            TimeSpan.FromSeconds(config.TimeLimitSeconds),
            config.SearchSettings ?? Settings,
            Weight: string.IsNullOrEmpty(config.WeightColumn) ? null : config.WeightColumn,
            Stratify: !IsRegression(config),
            MaxTrials: config.MaxTrials,
            Preprocess: config.PreFeaturizer,
            KeepOutOfFold: IsBinary(config) || config.Task.Equals("multiclass-classification", StringComparison.OrdinalIgnoreCase));

        SearchResult search;
        try
        {
            search = await Task.Run(() => new TrialRunner(request, channel).Run(trainSet, roles, cancellationToken), cancellationToken)
                .ConfigureAwait(false);
        }
        catch (SearchFailedException ex)
        {
            throw new NoSuccessfulTrialException(config.TimeLimitSeconds, ex);
        }

        // The model is the smallest within one standard error of the top score (TrialRunner); when that is not the
        // top-scoring trial, say so here and mark it in the trial records, where the leaderboard ranks by score.
        var records = channel.Records;
        if (!ReferenceEquals(search.Best, search.TopScoring))
        {
            var selected = IndexOf(search.Trials, search.Best);
            if (selected >= 0 && selected < records.Count)
                records = [.. records.Select((r, i) => i == selected ? r.AsSelected() : r)];
            search.TopScoring.Metrics!.TryGetValue(metric, out var topValue);
            search.Best.Metrics!.TryGetValue(metric, out var keptValue);
            _logger.Info($"Kept {search.Best.Trainer} ({metric} {keptValue:0.####}, model size {TrainerSize(search.Best)}) over " +
                $"{search.TopScoring.Trainer} ({topValue:0.####}, size {TrainerSize(search.TopScoring)}): within one standard error " +
                "of the top score, the smaller model.");
        }

        var bestModel = PortableTextFeaturizer.Prepend(portableText, search.Model);
        var predictions = bestModel.Transform(testSet);
        var (model, metrics, residualModel, calibration) = finish is null
            ? (bestModel, score(predictions, config.LabelColumn), null, null)
            : finish(bestModel, predictions, search.OutOfFold);

        return new TrainedModel
        {
            Trainer = new TrainerDescriptor
            {
                Name = search.Best.Trainer,
                FallbackReason = search.FallbackReason,
                FallbackKind = search.FallbackReason is null ? TrainerFallbackKind.None : TrainerFallbackKind.RefitFailed,
            },
            Model = model,
            Metrics = metrics,
            ResidualModel = residualModel,
            Calibration = calibration,
            RowCount = RowsIn(trainSet),
            Trials = records,
            RankingMetric = metric,
            ValidationScore = search.BestValidation.Mean.TryGetValue(metric, out var validation) ? validation : null,
            Search = request.Settings
        };
    }

    private static bool IsBinary(TrainingConfig config)
        => config.Task.Equals("binary-classification", StringComparison.OrdinalIgnoreCase);

    private static bool IsRegression(TrainingConfig config)
        => config.Task.Equals("regression", StringComparison.OrdinalIgnoreCase);

    // The seed and threads the MLoop search runs with unless the run sets its own (TrainingConfig.SearchSettings);
    // recorded with the experiment (TrainedModel.Search).
    private static readonly SearchSettings Settings = new(Seed: 1, Threads: 1);

    // A fold's metrics: weighted by the run's weight column when it has one, as the test partition's are.
    private static Dictionary<string, double> ScoreFold(
        IDataView scored, TrainingConfig config, Func<IDataView, string, Dictionary<string, double>> score)
    {
        var metrics = score(scored, config.LabelColumn);
        if (!string.IsNullOrEmpty(config.WeightColumn))
            OverlayWeighted(metrics, scored, config.Task, config.LabelColumn, config.WeightColumn);
        return metrics;
    }

    private static int IndexOf(IReadOnlyList<TrialResult> trials, TrialResult trial)
    {
        for (var i = 0; i < trials.Count; i++)
            if (ReferenceEquals(trials[i], trial))
                return i;
        return -1;
    }

    // Trees × leaves from a trial's settings, as TrainerCandidate.ModelSize reads them from its point.
    private static string TrainerSize(TrialResult trial)
    {
        double size = 1;
        foreach (var (name, value) in trial.Parameters ?? [])
        {
            if (name is "NumberOfTrees" or "NumberOfLeaves" && double.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var n))
                size *= n;
        }
        return size.ToString("N0", System.Globalization.CultureInfo.InvariantCulture);
    }

    // A binary model's metrics on scored rows and, where it gives probabilities, their calibration error.
    private Dictionary<string, double> ScoreBinaryWithCalibrationError(IDataView scored, string labelColumn)
    {
        var metrics = ScoreBinary(scored, labelColumn);
        if (CalibrationError.Binary(scored, labelColumn) is { } ece)
            metrics[CalibrationError.MetricName] = ece;
        return metrics;
    }

    // A multiclass model's metrics on scored rows: the evaluator's, the calibration error, and macro F1 from
    // the per-class precision and recall — for the test partition and, on MLoop's search, each fold.
    private Dictionary<string, double> ScoreMulticlass(IDataView scored, string labelColumn)
    {
        var metrics = _mlContext.MulticlassClassification.Evaluate(scored, labelColumn);
        var described = MetricReport.Describe(metrics);
        if (CalibrationError.Multiclass(scored, labelColumn) is { } ece)
            described[CalibrationError.MetricName] = ece;

        var cm = metrics.ConfusionMatrix;
        var classCount = cm.PerClassPrecision.Count;
        if (classCount > 0)
        {
            double f1Sum = 0;
            for (var i = 0; i < classCount; i++)
            {
                var p = cm.PerClassPrecision[i];
                var r = cm.PerClassRecall[i];
                f1Sum += (p + r) > 0 ? 2 * p * r / (p + r) : 0;
            }
            described["macro_f1"] = f1Sum / classCount;
        }
        return described;
    }

    private Dictionary<string, double> ScoreBinary(IDataView scored, string labelColumn)
    {
        try
        {
            return WithoutUndefinedRanking(MetricReport.Describe(scored.Schema.GetColumnOrNull("Probability") != null
                ? _mlContext.BinaryClassification.Evaluate(scored, labelColumn)
                : _mlContext.BinaryClassification.EvaluateNonCalibrated(scored, labelColumn)));
        }
        catch (Exception ex) when (IsAucUndefinedException(ex))
        {
            // Rows of one class — a fold or a test partition of rare positives. ML.NET's evaluator throws
            // rather than leave AUC undefined; the metrics the rows do define come from their counts.
            if (WeightedMetrics.Binary(scored, labelColumn, weightColumn: null) is { } counted)
                return counted;
            throw;
        }
    }

    // Whether LightGBM's native library loads here — it has none for osx-arm64. A search leaves the
    // trainer out rather than spend trials failing on it.
    private static readonly Lazy<bool> LightGbmLoads = new(() =>
        System.Runtime.InteropServices.NativeLibrary.TryLoad(
            "lib_lightgbm", typeof(Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer).Assembly, null, out _));

    /// <summary>
    /// The search can select a binary trainer (e.g. FastForest/FastTree)
    /// whose output has no "Probability" column. Training already falls back to
    /// <see cref="BinaryClassificationCatalog.EvaluateNonCalibrated"/> so metrics still compute, but
    /// the *saved* model stayed uncalibrated — every downstream consumer (mloop predict, mloop serve
    /// /predict, any confidence-gated caller) then reads a permanently-null Probability, and
    /// confidence collapses to 0 for every prediction: measured live, a forest binary trainer sent
    /// 100% of rows to escalation. Appending a Platt calibrator — fit on the
    /// scored rows <paramref name="predictions"/>, which carry the raw Score the calibrator needs — restores a
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
    /// <b>model-agnostic</b> — it works with whatever trainer the search selects, unlike ML.NET's
    /// trainer-locked quantile-regression forests which would break the free trainer sweep
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
        // Descriptive uncertainty stats are graceful (see summary): when the residuals are
        // non-finite — e.g. a degenerate model scored NaN on a near-empty test split — omit the
        // key rather than emit a meaningless value. These are not optimization metrics, so the
        // MetricSanitizer's direction-aware worst-sentinel would wrongly stamp a nonsensical
        // -double.MaxValue "residual std"; absence is the honest "interval unavailable" signal
        // (predict/serve already fall back when the keys are missing).
        var residualStd = Math.Sqrt(sumSquares / n);
        if (double.IsFinite(residualStd))
            result["residual_std"] = residualStd;

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
            if (double.IsFinite(q))
                result[$"interval_half_width_{pct}"] = q;
        }

        return result;
    }

    /// <summary>
    /// True when a scored view has at least one row and EVERY row's <paramref name="scoreColumn"/> is
    /// non-finite (NaN/±∞) — the signature of a degenerate model that predicts nothing usable (e.g.
    /// FastForest fitted on a handful of rows). Callers must treat the evaluator's metrics for such a
    /// view as fabricated: ML.NET evaluators silently filter non-finite-scored rows, so an all-non-finite
    /// holdout yields all-zero metrics that look boundary-passing. Returns false when the column is
    /// absent, not scalar-Single, or any row scores a finite value.
    /// </summary>
    public static bool AllScoresNonFinite(IDataView predictions, string scoreColumn = "Score")
    {
        var col = predictions.Schema.GetColumnOrNull(scoreColumn);
        if (col is null || col.Value.Type != NumberDataViewType.Single)
            return false;

        bool anyRow = false;
        using var cursor = predictions.GetRowCursor(new[] { col.Value });
        var getter = cursor.GetGetter<float>(col.Value);
        float value = 0;
        while (cursor.MoveNext())
        {
            anyRow = true;
            getter(ref value);
            if (float.IsFinite(value))
                return false;
        }
        return anyRow;
    }

    /// <summary>
    /// ② regression wave (heteroscedastic): normalized split-conformal prediction intervals. Where
    /// <see cref="ComputeConformalIntervals"/> gives one constant half-width for every row (valid
    /// coverage but no per-row triage signal — live-measured recall of the large-error rows equals
    /// random), this fits an auxiliary regressor σ(x) that predicts the *magnitude*
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
    /// a real manufacturing set confirmed (recall 0.10→0.24). A tree aux is a demand-driven future option if a live set
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

        // σ-model: raw numeric features → |residual|. Trained only on C1. FastForest, not SDCA:
        // SDCA's fit is platform-/thread-trajectory-sensitive and can silently DIVERGE on this tiny
        // aux problem (observed: σ ≈ -8.7e7 single-threaded on win-x64, and the macOS-arm64 CI's
        // multi-threaded fit degenerated the same way), and the max(σ,0)+β floor then masks the
        // divergence as a constant-width band while the metadata still claims heteroscedastic.
        // A forest predicts leaf averages of the |residual| target, so σ is structurally bounded to
        // the training residual range (never negative, cannot diverge); NumberOfThreads = 1 keeps
        // the conformal calibration reproducible for the same data.
        var auxPipeline = ml.Transforms.Concatenate("Features", feats)
            .Append(ml.Transforms.NormalizeMinMax("Features"))
            .Append(ml.Regression.Trainers.FastForest(new Microsoft.ML.Trainers.FastTree.FastForestRegressionTrainer.Options
            {
                LabelColumnName = "Target",
                FeatureColumnName = "Features",
                NumberOfThreads = 1,
            }));
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

        // Omit any non-finite descriptive stat (same graceful contract as ComputeConformalIntervals):
        // these interval widths are not optimization metrics, so absence — not a worst-sentinel — is
        // the correct "interval unavailable" signal.
        foreach (var key in metrics.Keys.ToArray())
            if (!double.IsFinite(metrics[key]))
                metrics.Remove(key);

        return new NormalizedConformalResult(auxModel, metrics);
    }

    /// <summary>
    /// Picks the feature representation the σ-model reads from a scored prediction view. MLoop
    /// pre-featurizes into a single <c>Features</c> vector (the individual raw columns are gone by the
    /// time the model transforms the test set), so the aux model consumes that same featurized vector —
    /// which also lets it see categorical/text-derived features, not just raw numerics. Prefers a
    /// non-internal (<c>__…__</c> are internal scratch columns) vector-of-Single column named such as
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

    private Task<TrainedModel> RunMulticlassClassificationAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
        => RunWithMLoopSearchAsync(trainSet, testSet, config,
            CandidateCatalog.Multiclass(name => name != "LightGbm" || LightGbmLoads.Value),
            OptimizingMetric(config), ScoreMulticlass, progress, cancellationToken,
            (model, predictions, outOfFold) => FinishMulticlass(model, testSet, predictions, outOfFold, config));

    /// <summary>
    /// A multiclass model's test results, its class probabilities recalibrated by a temperature fitted on the kept
    /// trial's out-of-fold answers — the multiclass counterpart of <see cref="FinishBinary"/>. One-versus-all trainers
    /// normalize fixed per-class sigmoids, and their probabilities were measured far from how often they are right
    /// (top-label ECE 0.32 to 0.43, where a maximum-entropy model on the same rows was below 0.001). A temperature
    /// scales every class's log-probability alike, so each row's answer and order of classes stay: accuracy does not
    /// move, the log loss and the ECE do.
    /// </summary>
    private (ITransformer Model, Dictionary<string, double> Metrics, ITransformer? ResidualModel, ProbabilityCalibration? Calibration) FinishMulticlass(
        ITransformer model, IDataView testSet, IDataView predictions, OutOfFoldScores? outOfFold, TrainingConfig config)
    {
        var calibration = outOfFold is { IsMulticlass: true, Count: > 0 }
            ? ProbabilityCalibration.FitTemperature(
                [.. Enumerable.Range(0, outOfFold.Count).Select(i => (outOfFold.ClassProbabilities[i], outOfFold.Class[i], (double)outOfFold.Weight[i]))],
                OutOfFoldSource, string.IsNullOrEmpty(config.WeightColumn) ? null : config.WeightColumn, DateTimeOffset.UtcNow)
            : null;
        return calibration is null
            ? (model, ScoreMulticlass(predictions, config.LabelColumn), null, null)
            : (model, ScoreMulticlass(calibration.Apply(_mlContext, model).Transform(testSet), config.LabelColumn), null, calibration);
    }

    private Task<TrainedModel> RunRegressionAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken)
        => RunWithMLoopSearchAsync(trainSet, testSet, config,
            CandidateCatalog.Regression(name => name != "LightGbm" || LightGbmLoads.Value),
            OptimizingMetric(config), ScoreRegression, progress, cancellationToken,
            (model, predictions, _) =>
            {
                var (metrics, residualModel) = FinishRegression(predictions, config.LabelColumn);
                return (model, metrics, residualModel, null);
            });
    /// <summary>
    /// A regression model's test results: its metrics on the test predictions, the prediction-interval
    /// widths, and the σ-model that widens or narrows the band per row when one fits — on either search.
    /// </summary>
    private (Dictionary<string, double> Metrics, ITransformer? ResidualModel) FinishRegression(IDataView predictions, string labelColumn)
    {
        var metricsDict = ScoreRegression(predictions, labelColumn);

        // Every score non-finite: the metrics are already NaN (ScoreRegression), and intervals over NaN
        // residuals mean nothing.
        ITransformer? residualModel = null;
        if (AllScoresNonFinite(predictions))
        {
            _logger.Warning(
                "The selected model scored a non-finite value (NaN/Infinity) for every holdout row — a " +
                "degenerate model, typically the result of training on too little data. The evaluator " +
                "metrics are fabricated (every row was filtered out) and will be recorded as worst-case " +
                "so this model cannot pass the quality gate. Provide more training data.");
            // Conformal calibration over all-NaN residuals is meaningless — skip it (its graceful
            // key-omission contract would drop everything anyway).
        }
        else
        {
            // ② regression wave: derive split-conformal prediction-interval half-widths from the same
            // held-out test predictions used for metrics, so predict/serve can surface an uncertainty
            // band [Score ± q] (the regression analogue of the confidence signal binary/multiclass expose).
            foreach (var interval in ComputeConformalIntervals(predictions, labelColumn))
                metricsDict[interval.Key] = interval.Value;

            // ② regression wave (heteroscedastic): fit the per-row σ(x) band on top of the constant-width
            // one. The constant-width metrics stay as the backward-safe fallback; when the aux model fits,
            // its normalized-quantile metrics + the residual model let predict widen/narrow per row.
            var featureColumns = SelectSigmaFeatureColumns(predictions.Schema, labelColumn);
            var normalized = ComputeNormalizedConformal(_mlContext, predictions, labelColumn, featureColumns);
            if (normalized is not null)
            {
                residualModel = normalized.AuxModel;
                foreach (var kv in normalized.Metrics)
                    metricsDict[kv.Key] = kv.Value;
            }
        }

        return (metricsDict, residualModel);
    }

    // A regression model's metrics on scored rows, for the test partition and, on MLoop's search, each fold.
    private Dictionary<string, double> ScoreRegression(IDataView scored, string labelColumn)
    {
        // ML.NET's Regression.Evaluate silently drops rows whose Score is non-finite. When a
        // model is degenerate (e.g. FastForest fitted on a handful of rows scores NaN for EVERY input)
        // the evaluator is left with zero usable rows and fabricates all-zero metrics — r²=0 AND rmse=0,
        // a mathematically contradictory pair that lands exactly on the quality gate's r²≥0 boundary and
        // gets promoted; predict then emits literal '?' CSVs. Detect the all-non-finite holdout directly
        // and mark the core metrics NaN so the MetricSanitizer funnel (RunAsync) replaces them with
        // direction-aware worst-sentinels — the existing r²<0 gate then blocks promotion, with no new
        // gate logic and no way for the fabricated zeros to masquerade as a boundary-passing model.
        // A fold scored this way counts as a failed trial's loss, so such a model never wins the search either.
        var metrics = MetricReport.Describe(_mlContext.Regression.Evaluate(scored, labelColumn));
        if (AllScoresNonFinite(scored))
        {
            foreach (var key in metrics.Keys.ToArray())
                metrics[key] = double.NaN;
        }
        return metrics;
    }

    private async Task<TrainedModel> RunAnomalyDetectionAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string[]>? mergedColumnGroups)
    {
        return await Task.Run(() =>
        {
            // The numeric feature columns (not the label)
            var featureColumns = NumericVectorColumns(trainSet.Schema,
                name => name.Equals(config.LabelColumn, StringComparison.OrdinalIgnoreCase));

            if (featureColumns.Count == 0)
                throw new InvalidOperationException("No numeric feature columns found for anomaly detection.");

            _logger.Info($"Anomaly detection: {featureColumns.Count} feature columns, using RandomizedPca");

            // Determine PCA rank (auto: min of feature count and 20)
            var rank = Math.Min(featureColumns.Count, 20);

            // Build pipeline: Concatenate features → missing values to the mean (MissingNumbers) → RandomizedPca
            var pipeline = FeatureVector(trainSet.Schema, featureColumns)
                .Append(_mlContext.Transforms.ReplaceMissingValues("Features", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(_mlContext.AnomalyDetection.Trainers.RandomizedPca(
                    featureColumnName: "Features",
                    rank: rank,
                    oversampling: 20));

            // The trial and the result describe the same single fit — one descriptor, named by the
            // rank that identifies this configuration.
            var trainer = TrainerDescriptor.Of("RandomizedPca", ("rank", rank));
            var trialChannel = CreateTrialChannel(progress);

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

            // No detection rate means the prediction produced nothing to report — stay silent
            // rather than send a zero.
            if (metricsDict.TryGetValue("detection_rate", out var detectionRate))
                trialChannel.ReportCompleted(trainer, "detection_rate", detectionRate, metricsDict);

            return new TrainedModel
            {
                Trainer = trainer,
                Model = model,
                Metrics = metricsDict,
                RowCount = RowsIn(trainSet),
                Trials = trialChannel.Records,
                RankingMetric = trialChannel.RankingMetric
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<TrainedModel> RunClusteringAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string[]>? mergedColumnGroups)
    {
        return await Task.Run(() =>
        {
            // The numeric feature columns (not a label, when there is one)
            var featureColumns = NumericVectorColumns(trainSet.Schema,
                name => !string.IsNullOrEmpty(config.LabelColumn) && name.Equals(config.LabelColumn, StringComparison.OrdinalIgnoreCase));

            if (featureColumns.Count == 0)
                throw new InvalidOperationException("No numeric feature columns found for clustering.");

            // Featurize once, before the k search, and fit only the trainer on the result; the saved model is
            // the two together, so it reads the loaded columns as every other task's model does — yes/no
            // columns included, which no prediction path folds into "Features" on its own.
            var featurizer = FeatureVector(trainSet.Schema, featureColumns).Fit(trainSet);
            var featurizedTrainSet = featurizer.Transform(trainSet);
            var featurizedTestSet = featurizer.Transform(testSet);

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
            var trialChannel = CreateTrialChannel(progress);

            for (int i = 0; i < kValues.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var k = kValues[i];

                // Missing values to the mean inside the model, in place — its input stays "Features" (MissingNumbers).
                var pipeline = _mlContext.Transforms.ReplaceMissingValues("Features", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
                    .Append(_mlContext.Clustering.Trainers.KMeans(
                        featureColumnName: "Features",
                        numberOfClusters: k));

                var model = pipeline.Fit(featurizedTrainSet);
                var predictions = model.Transform(featurizedTestSet);

                // Evaluate. featureColumnName is REQUIRED for ML.NET to compute the Davies-Bouldin
                // Index — without it DBI comes back 0, which made `useDbi` below always false so the
                // K-search silently fell back to average_distance (monotonically decreasing with K)
                // and always picked the largest K. Passing "Features" restores real DBI-based K-search.
                var clusterMetrics = _mlContext.Clustering.Evaluate(
                    predictions, scoreColumnName: "Score", featureColumnName: "Features");
                var avgDistance = double.IsNaN(clusterMetrics.AverageDistance) ? double.MaxValue : clusterMetrics.AverageDistance;
                var dbi = double.IsNaN(clusterMetrics.DaviesBouldinIndex) ? double.MaxValue : clusterMetrics.DaviesBouldinIndex;
                var nmi = double.IsNaN(clusterMetrics.NormalizedMutualInformation) ? 0 : clusterMetrics.NormalizedMutualInformation;

                trialChannel.ReportCompleted(TrainerDescriptor.Of("KMeans", ("k", k)), "davies_bouldin_index", dbi,
                    new Dictionary<string, double>
                    {
                        ["davies_bouldin_index"] = dbi,
                        ["average_distance"] = avgDistance,
                        ["normalized_mutual_information"] = nmi,
                        ["k"] = k
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
                var bestPredictions = bestModel.Transform(featurizedTestSet);
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

            return new TrainedModel
            {
                Trainer = TrainerDescriptor.Of("KMeans", ("k", bestK)),
                Model = featurizer.Append(bestModel!),
                Metrics = bestMetrics ?? new Dictionary<string, double>(),
                RowCount = RowsIn(trainSet),
                Trials = trialChannel.Records,
                RankingMetric = trialChannel.RankingMetric
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<TrainedModel> RunRankingAsync(
        IDataView trainSet,
        IDataView testSet,
        TrainingConfig config,
        IProgress<TrainingProgress>? progress,
        CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string[]>? mergedColumnGroups)
    {
        return await Task.Run(() =>
        {
            if (string.IsNullOrEmpty(config.GroupColumn))
                throw new InvalidOperationException("Ranking task requires a group column. Set 'group_column' in mloop.yaml or use --group-column.");

            // Validate group column exists
            var groupCol = trainSet.Schema.GetColumnOrNull(config.GroupColumn);
            if (!groupCol.HasValue)
                throw new InvalidOperationException($"Group column '{config.GroupColumn}' not found in data.");

            // The numeric feature columns (not the label or the group)
            var featureColumns = NumericVectorColumns(trainSet.Schema,
                name => name.Equals(config.LabelColumn, StringComparison.OrdinalIgnoreCase)
                     || name.Equals(config.GroupColumn, StringComparison.OrdinalIgnoreCase));

            if (featureColumns.Count == 0)
                throw new InvalidOperationException("No numeric feature columns found for ranking.");

            // Count values, not columns: the loader may have merged many numeric columns into one vector.
            var featureWidth = featureColumns.Sum(name =>
                trainSet.Schema[name].Type is VectorDataViewType vector ? vector.Size : 1);
            _logger.Info($"Ranking: {featureWidth} features, group='{config.GroupColumn}', label='{config.LabelColumn}'");

            // Build pipeline: convert label to Single, hash group to Key, concatenate features
            var pipeline = _mlContext.Transforms.Conversion.ConvertType(config.LabelColumn, outputKind: Microsoft.ML.Data.DataKind.Single)
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("GroupId", config.GroupColumn))
                .Append(FeatureVector(trainSet.Schema, featureColumns))
                .Append(_mlContext.Transforms.ReplaceMissingValues("Features", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean));

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
            var trialChannel = CreateTrialChannel(progress);

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

                    trialChannel.ReportCompleted(
                        TrainerDescriptor.Of(trainerName), "ndcg", primaryNdcg, metricsDict);

                    if (primaryNdcg > bestNdcg)
                    {
                        bestNdcg = primaryNdcg;
                        bestModel = model;
                        bestTrainerName = trainerName;
                        bestMetrics = metricsDict;
                    }
                }
                catch (Exception ex) when (MLoop.Core.Runtime.NativeLoad.IsMissing(ex))
                {
                    _logger.Info(MLoop.Core.Runtime.NativeLoad.TrainerSkipped(trainerName));
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

            return new TrainedModel
            {
                Trainer = TrainerDescriptor.Of(bestTrainerName),
                Model = bestModel,
                Metrics = bestMetrics ?? new Dictionary<string, double>(),
                RowCount = RowsIn(trainSet),
                Trials = trialChannel.Records,
                RankingMetric = trialChannel.RankingMetric
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<TrainedModel> RunForecastingAsync(
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
            MissingNumbers.RefuseGapsInSeries(trainSet, valueColumn);

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
                outputColumnName: ForecastOutput.ForecastColumnName,
                inputColumnName: valueColumn,
                windowSize: windowSize,
                seriesLength: seriesLength,
                trainSize: trainSize,
                horizon: horizon,
                confidenceLowerBoundColumn: ForecastOutput.LowerBoundColumnName,
                confidenceUpperBoundColumn: ForecastOutput.UpperBoundColumnName,
                confidenceLevel: (float)ForecastOutput.ConfidenceLevel);

            // One descriptor for the trial and the result: the same fit, named by the window and
            // horizon that identify it.
            var forecastTrainer = TrainerDescriptor.Of(
                "SsaForecasting", ("window", windowSize), ("horizon", horizon));
            var trialChannel = CreateTrialChannel(progress);

            // SSA is an FFT algorithm, so this is where a machine without ML.NET's MKL native fails.
            var model = TimeSeriesNativeSupport.Guard(() => pipeline.Fit(trainSet));

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

            // The holdout comparison above is skipped when the series is too short to leave one, so
            // there may be no error metric to report — say nothing rather than report mae=0, which
            // reads as a perfect forecast.
            if (metricsDict.TryGetValue("mae", out var mae))
                trialChannel.ReportCompleted(forecastTrainer, "mae", mae, metricsDict);

            metricsDict["horizon"] = horizon;
            metricsDict["window_size"] = windowSize;
            metricsDict["series_length"] = seriesLength;
            metricsDict["train_size"] = trainSize;

            return new TrainedModel
            {
                Trainer = forecastTrainer,
                Model = model,
                Metrics = metricsDict,
                RowCount = totalRows,
                Trials = trialChannel.Records,
                RankingMetric = trialChannel.RankingMetric
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<TrainedModel> RunTimeSeriesAnomalyAsync(
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
            MissingNumbers.RefuseGapsInSeries(trainSet, valueColumn);

            var totalRows = (int)CountRows(trainSet);
            if (totalRows < 12)
                throw new InvalidOperationException($"Time series anomaly detection requires at least 12 data points (got {totalRows}).");

            _logger.Info($"Time series anomaly: series='{valueColumn}', {totalRows} rows");

            // Try multiple detectors: SrCnn (best), then SSA Spike, then SSA ChangePoint
            var detectors = new List<(string Name, Func<IEstimator<ITransformer>> Factory)>();

            // SR-CNN: Spectral Residual based — best general-purpose detector
            // Every window derives from the row count together — see
            // SrCnnWindowSizes for why the judgement window cannot be a literal.
            var srCnn = SrCnnWindowSizes.ForSeries(totalRows);
            detectors.Add(("SrCnnAnomaly", () =>
                _mlContext.Transforms.DetectAnomalyBySrCnn(
                    outputColumnName: TimeSeriesAnomalyOutput.PredictionColumnName,
                    inputColumnName: valueColumn,
                    windowSize: srCnn.WindowSize,
                    backAddWindowSize: srCnn.BackAddWindowSize,
                    lookaheadWindowSize: srCnn.LookaheadWindowSize,
                    averagingWindowSize: srCnn.AveragingWindowSize,
                    judgementWindowSize: srCnn.JudgementWindowSize,
                    threshold: 0.3)));

            // SSA Spike Detector — trainingWindowSize must be > 2 * seasonalityWindowSize
            var ssaWindowSize = Math.Max(2, Math.Min(totalRows / 8, 50));
            var ssaPValueSize = Math.Max(2, ssaWindowSize / 2);
            var ssaTrainingSize = Math.Max(ssaWindowSize * 2 + 1, Math.Min(totalRows, 100));
            detectors.Add(("SsaSpikeDetector", () =>
                _mlContext.Transforms.DetectSpikeBySsa(
                    outputColumnName: TimeSeriesAnomalyOutput.PredictionColumnName,
                    inputColumnName: valueColumn,
                    confidence: 95.0,
                    pvalueHistoryLength: ssaPValueSize,
                    trainingWindowSize: ssaTrainingSize,
                    seasonalityWindowSize: ssaWindowSize)));

            ITransformer? bestModel = null;
            string bestDetectorName = "";
            Dictionary<string, double>? bestMetrics = null;
            long bestAnomalyCount = -1;
            Exception? missingNative = null;
            var trialChannel = CreateTrialChannel(progress);

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
                    var predCol = predictions.Schema.GetColumnOrNull(TimeSeriesAnomalyOutput.PredictionColumnName);
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

                    trialChannel.ReportCompleted(
                        TrainerDescriptor.Of(name), "detection_rate", detectionRate, metricsDict);

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
                    // Every detector here is FFT-based, so a machine without ML.NET's MKL native
                    // fails all of them for one reason. Remember that, or the loop reports only
                    // "all detectors failed" and the actionable cause is lost.
                    if (TimeSeriesNativeSupport.IsMissingNative(ex))
                        missingNative = ex;

                    _logger.Info($"Time series anomaly detector '{name}' failed: {ex.Message}");
                }
            }

            if (bestModel == null)
            {
                if (missingNative is not null)
                    throw new PlatformNotSupportedException(
                        TimeSeriesNativeSupport.UnavailableMessage, missingNative);

                throw new InvalidOperationException("All time series anomaly detectors failed.");
            }

            return new TrainedModel
            {
                Trainer = TrainerDescriptor.Of(bestDetectorName),
                Model = bestModel,
                Metrics = bestMetrics ?? new Dictionary<string, double>(),
                RowCount = totalRows,
                Trials = trialChannel.Records,
                RankingMetric = trialChannel.RankingMetric
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<TrainedModel> RunRecommendationAsync(
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

            var trialChannel = CreateTrialChannel(progress);

            var model = pipeline.Fit(trainSet);

            // Evaluate on test set
            var predictions = model.Transform(testSet);
            var regressionMetrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: config.LabelColumn);

            // Non-finite values are normalized centrally by MetricSanitizer in RunAsync
            // (direction-aware worst) — a local NaN→0 guard here would wrongly launder a broken
            // error metric into a perfect-looking 0.
            var metricsDict = new Dictionary<string, double>
            {
                ["rmse"] = regressionMetrics.RootMeanSquaredError,
                ["mae"] = regressionMetrics.MeanAbsoluteError,
                ["r_squared"] = regressionMetrics.RSquared,
                ["loss_function"] = regressionMetrics.LossFunction
            };

            trialChannel.ReportCompleted(
                TrainerDescriptor.Of("MatrixFactorization", ("rank", 32), ("iter", 20)),
                "rmse", metricsDict["rmse"], metricsDict);

            return new TrainedModel
            {
                Trainer = TrainerDescriptor.Of("MatrixFactorization", ("rank", 32), ("iter", 20)),
                Model = model,
                Metrics = metricsDict,
                RowCount = RowsIn(trainSet),
                Trials = trialChannel.Records,
                RankingMetric = trialChannel.RankingMetric
            };
        }, cancellationToken).ConfigureAwait(false);
    }

    // The optimizing metric for a run. The name arrives already canonical
    // (TrainingConfig.CanonicalMetric, via MetricNames) — these switches map MLoop's vocabulary to
    // ML.NET's enum and hold no spellings of their own. A name the switch cannot take is refused
    // (MetricNames.Rejection), never replaced by the task default: a substitute would be optimized
    // while every record kept the name that was given, which is how `metric: F1Score` optimized
    // accuracy for a long time. Producers (CLI train, API /train) refuse the same names before any
    // work starts; this is the backstop for a caller that skipped them.
    //
    // The *Metric switches below are the inverse of the Describe*Metric switches further down
    // (enum → name), and a round-trip test pins that they agree.

    // The metric a binary, multiclass or regression search optimizes. The name arrives already canonical
    // (TrainingConfig.CanonicalMetric, via MetricNames), and a name the task cannot optimize
    // (MetricNames.OptimizableFor) is refused, never replaced by the task default: a substitute would be
    // optimized while every record kept the name that was given, which is how `metric: F1Score` optimized
    // accuracy for a long time. Producers (CLI train, API /train) refuse the same names before any work
    // starts; this is the backstop for a caller that skipped them.
    private static string OptimizingMetric(TrainingConfig config) =>
        MetricNames.SearchObjective(config.Task, config.Metric)
        ?? (MetricNames.OptimizableFor(config.Task) is { } optimizable && optimizable.Contains(config.CanonicalMetric)
            ? config.CanonicalMetric
            : throw Refuse(config));
    private static ArgumentException Refuse(TrainingConfig config) =>
        new(MetricNames.Rejection(config.Task, config.Metric)
            ?? $"Metric '{config.Metric}' cannot be optimized for {config.Task}.", nameof(config));

    /// <summary>
    /// Trial channel for the task paths that do not run the trainer search: they fit one explicit pipeline (or
    /// loop over a few candidates) and report each one as it completes.
    /// </summary>
    /// <remarks>
    /// Always created, even with nobody listening. It used to be null in that case — which was
    /// harmless while it only forwarded progress, but became a defect the moment it also kept the
    /// trial records: the leaderboard would have existed only for runs someone happened to be
    /// watching, and these paths have no other record to fall back on.
    /// </remarks>
    private static TrialProgressChannel CreateTrialChannel(IProgress<TrainingProgress>? progress)
        => new(progress);

    // A test partition of one class leaves AUC and AUPRC undefined (NaN); the result records them as
    // absent rather than as a number, as it always has for AUC.
    private static Dictionary<string, double> WithoutUndefinedRanking(Dictionary<string, double> metrics)
    {
        foreach (var key in new[] { "auc", "auprc" })
            if (metrics.TryGetValue(key, out var value) && double.IsNaN(value))
                metrics.Remove(key);
        return metrics;
    }

    /// <summary>
    /// Check if exception is an AUC undefined error, handling both direct
    /// InvalidOperationException and AggregateException wrappers a thread-pool task adds.
    /// Matches errors like "AUC is not defined when there is no positive class" and
    /// "AUC is not defined when there is no negative class".
    /// </summary>
    private static bool IsAucUndefinedException(Exception ex)
    {
        // Direct exception
        if (IsAucMessage(ex.Message))
            return true;

        // AggregateException from Task.Run
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

    private void InfoOnce(string message)
    {
        bool first;
        lock (_saidThisRun)
            first = _saidThisRun.Add(message);
        if (first)
            _logger.Info(message);
    }

    /// <summary>
    /// Decides the role of every non-label column the search may read: an override in the
    /// configuration wins, then a string column the run decided holds categories is categorical,
    /// any other string column is text, and numbers, booleans and numeric vectors are numeric.
    /// Columns of any other type get no role.
    /// </summary>
    public static ColumnRoles BuildColumnRoles(
        IDataView data, string labelColumn,
        Action<string>? log = null,
        Dictionary<string, string>? columnOverrides = null,
        IReadOnlyCollection<string>? categoryColumns = null)
    {
        var categories = new HashSet<string>(categoryColumns ?? [], StringComparer.OrdinalIgnoreCase);
        var inferredCategories = new List<string>();
        var textColumns = new List<string>();
        var numericColumns = new List<string>();
        var categoricalColumns = new List<string>();
        var ignoredColumns = new List<string>();
        var unreadColumns = new List<string>();

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

            // Default: infer from IDataView type. A string column the run decided holds categories
            // is one-hot encoded (CategoryColumns); any other is featurized as text.
            if (col.Type is TextDataViewType)
            {
                if (categories.Contains(col.Name))
                {
                    categoricalColumns.Add(col.Name);
                    inferredCategories.Add(col.Name);
                }
                else
                {
                    textColumns.Add(col.Name);
                }
            }
            // The loader folds every yes/no column into one boolean vector (BooleanFeatures) once there are
            // two or more of them; leaving vectors of booleans out dropped all of those columns from every
            // model without a word.
            else if (col.Type is NumberDataViewType || col.Type is BooleanDataViewType
                || (col.Type is VectorDataViewType vt2 && vt2.ItemType is NumberDataViewType or BooleanDataViewType))
            {
                numericColumns.Add(col.Name);
            }
            else
            {
                unreadColumns.Add($"{col.Name} ({col.Type})");
            }
        }

        // One role per column; the order within a role is the order the columns were met in.
        var roles = new ColumnRoles(labelColumn,
            textColumns.Select(c => KeyValuePair.Create(c, ColumnRole.Text))
                .Concat(numericColumns.Select(c => KeyValuePair.Create(c, ColumnRole.Numeric)))
                .Concat(categoricalColumns.Select(c => KeyValuePair.Create(c, ColumnRole.Categorical)))
                .Concat(ignoredColumns.Select(c => KeyValuePair.Create(c, ColumnRole.Ignored))));

        var write = log ?? Data.CsvDataLoader.NoLog;
        if (textColumns.Count > 0)
            write($"[Info] Text column(s): {string.Join(", ", textColumns)} — applying text featurization (TF-IDF, n-gram)");
        var declaredCategories = categoricalColumns.Except(inferredCategories).ToList();
        if (inferredCategories.Count > 0)
            write($"[Info] Categorical column(s): {string.Join(", ", inferredCategories)} — one-hot encoding");
        if (declaredCategories.Count > 0)
            write($"[Info] Categorical column(s) (override): {string.Join(", ", declaredCategories)}");
        if (ignoredColumns.Count > 0)
            write($"[Info] Ignored column(s) (override): {string.Join(", ", ignoredColumns)}");
        // A column no rule above reads is not in the model. Said here, where it is decided — a vector of yes/no
        // values once fell through without a word, and every model left those columns out.
        if (unreadColumns.Count > 0)
            write(CoreMessage.Warning($"Column(s) of a type the search does not read, not used: {string.Join(", ", unreadColumns)}"));

        return roles;
    }

    /// <summary>
    /// The columns a trainer that reads one numeric vector (anomaly detection, clustering, ranking) puts into
    /// <c>Features</c>: the numbers and the yes/no columns — scalar columns and the loader's vectors — in
    /// schema order. <see cref="FeatureVector"/> turns them into the vector.
    /// </summary>
    private static List<string> NumericVectorColumns(DataViewSchema schema, Func<string, bool> skip)
    {
        var columns = new List<string>();
        foreach (var col in schema)
        {
            if (col.IsHidden || skip(col.Name))
                continue;

            var item = col.Type is VectorDataViewType vector ? vector.ItemType : col.Type;
            if (item is NumberDataViewType or BooleanDataViewType)
                columns.Add(col.Name);
        }
        return columns;
    }

    /// <summary>
    /// <c>Features</c> from <paramref name="columns"/>: yes/no columns as 0/1 (<c>Concatenate</c> joins one item
    /// type only), then every column in order. Part of the saved model, so it reads the columns as the loader
    /// names them — and as serving folds them (<see cref="Data.ColumnLayoutInference.FoldedName"/>).
    /// </summary>
    private IEstimator<ITransformer> FeatureVector(DataViewSchema schema, IReadOnlyList<string> columns)
    {
        var concatenate = _mlContext.Transforms.Concatenate("Features", [.. columns]);
        var yesNo = columns
            .Where(c => (schema[c].Type is VectorDataViewType v ? v.ItemType : schema[c].Type) is BooleanDataViewType)
            .Select(c => new InputOutputColumnPair(c, c))
            .ToArray();
        return yesNo.Length == 0
            ? concatenate
            : _mlContext.Transforms.Conversion.ConvertType(yesNo, DataKind.Single).Append(concatenate);
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
/// Single authority for the time-series-anomaly detector output contract. Every detector in the
/// RunTimeSeriesAnomalyAsync chain (SrCnn, SsaSpike) emits one vector column whose first slot is
/// the alert flag (1 = anomaly) and second the detector's raw score. The third slot's meaning
/// differs per detector (SrCnn → magnitude, SSA → p-value), so it is not surfaced generically.
/// Consumed by the trainer chain and PredictionService's extraction so the two never drift.
/// </summary>
public static class TimeSeriesAnomalyOutput
{
    public const string PredictionColumnName = "Prediction";
    public const int AlertSlot = 0;
    public const int RawScoreSlot = 1;
}

/// <summary>
/// Output schema for time series prediction engine. Also the single authority for the SSA
/// forecast output contract — column names and confidence level — consumed by the trainer
/// pipeline (RunForecastingAsync) and the CLI forecast predict path, so the two never drift.
/// </summary>
public class ForecastOutput
{
    public const string ForecastColumnName = "ForecastedValues";
    public const string LowerBoundColumnName = "LowerBound";
    public const string UpperBoundColumnName = "UpperBound";
    /// <summary>Coverage level of the SSA native prediction interval (confidenceLevel at train time).</summary>
    public const double ConfidenceLevel = 0.95;

    public float[] ForecastedValues { get; set; } = [];
    public float[] LowerBound { get; set; } = [];
    public float[] UpperBound { get; set; } = [];
}

/// <summary>
/// Result of a training run
/// </summary>
/// <remarks>
/// A record, not a plain class, so <c>RunAsync</c>'s schema-capture step can rewrap it with
/// <c>with</c>. The hand-written copy it replaced had to name every property, which made adding a
/// property a silent-data-loss hazard rather than a compile error.
/// </remarks>
public record TrainedModel
{
    /// <summary>What was trained, in parts — name, hyperparameters, fallback reason.</summary>
    public required TrainerDescriptor Trainer { get; init; }

    /// <summary>
    /// The human-facing rendering of <see cref="Trainer"/>. Derived, not stored: it used to be
    /// assembled by hand at each task path, which is how hyperparameters and fallback notes ended
    /// up inside what consumers read as a trainer identifier.
    /// </summary>
    public string BestTrainer => Trainer.Display;

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
    /// Columns the trainer read that a prediction does not supply, besides the label — for question
    /// answering, the answer's start position. The saved schema records them as not required.
    /// </summary>
    public IReadOnlyList<string>? TrainingOnlyColumns { get; init; }

    /// <summary>
    /// ② regression wave (heteroscedastic): the auxiliary σ(x) model that predicts residual magnitude
    /// from the features, used to widen/narrow the conformal band per row. Null for non-regression
    /// tasks, homoscedastic fallbacks, or when the holdout was too small to fit one — callers then use
    /// the constant-width band. Persisted alongside the main model (see <c>ResidualModelFileName</c>).
    /// </summary>
    public ITransformer? ResidualModel { get; init; }

    /// <summary>
    /// The probability calibration fitted with the model — a binary model's, on the search's out-of-fold answers —
    /// saved beside it (<see cref="ProbabilityCalibration.WriteBeside"/>), where every scoring path applies it. Null
    /// where none was fitted. <see cref="Metrics"/> are the calibrated model's.
    /// </summary>
    public ProbabilityCalibration? Calibration { get; init; }

    /// <summary>
    /// The text columns the model also reads through a text encoder, with the encoders' identities — saved beside it
    /// (<see cref="TextEncodings.WriteBeside"/>), where every loading path puts the same encoding in front of the model.
    /// Null when the model reads no encoded column. <see cref="Model"/> itself takes the vectors beside the text
    /// (<see cref="TextEncodings.VectorColumnOf"/>), and <see cref="EvaluationData"/> carries them.
    /// </summary>
    public TextEncodings? TextEncodings { get; init; }

    /// <summary>
    /// An image-anomaly model's bank and its record — the model itself, saved beside <see cref="Model"/> (which is only that
    /// experiment's header). Null for every other task.
    /// </summary>
    public ImageAnomaly.ImageAnomalyArtifacts? ImageAnomaly { get; init; }

    /// <summary><see cref="Model"/> as it answers once saved — with <see cref="Calibration"/> applied when there is one.</summary>
    public ITransformer ScoringModel(MLContext mlContext) => Calibration?.Apply(mlContext, Model) ?? Model;

    /// <summary>
    /// Every trial the run completed and reported, in completion order — the leaderboard that used
    /// to be discarded with <c>RunDetails</c> once <c>BestRun</c> had been taken off it. Every
    /// reported trial has a record here, including paths that fit a single pipeline: exempting
    /// those is what let the reported count and the recorded count mean different things. Empty
    /// only where nothing was reported (the paths with no metric to report).
    /// </summary>
    public IReadOnlyList<TrialRecord> Trials { get; init; } = [];

    /// <summary>
    /// Canonical name of the metric the search ranked trials by — the key to sort
    /// <see cref="Trials"/> on. Kept separately from the requested metric because the two can
    /// diverge: binary classification switches to <c>f1_score</c> mid-run when AUC turns out to be
    /// undefined. Null when nothing was ranked.
    /// </summary>
    public string? RankingMetric { get; init; }

    /// <summary>
    /// The seed and thread count MLoop's own search ran with — what repeats it: one seed on the same
    /// data proposes the same trials in the same order. Null when the run did not use the trainer search.
    /// </summary>
    public SearchSettings? Search { get; init; }

    /// <summary>
    /// The held-out rows <see cref="Metrics"/> were measured on, unscored — what the project's own
    /// metric scripts are computed over after training. Null where a task has no such partition.
    /// </summary>
    public IDataView? EvaluationData { get; init; }

    /// <summary>
    /// The kept trial's score on <see cref="RankingMetric"/> over its validation folds (or holdout) — what the search
    /// chose the model by, measured on the training partition. Null where no search ran. Unlike <see cref="Metrics"/>
    /// (the test partition, which the run reports), it may be compared to choose between two runs.
    /// </summary>
    public double? ValidationScore { get; init; }
}

/// <summary>
/// ② regression wave (heteroscedastic): output of
/// <see cref="TrainingRunner.ComputeNormalizedConformal"/> — the fitted auxiliary σ(x) model plus the
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
/// Routes the runner's diagnostics to the sinks its host supplied, and discards what has no sink.
/// </summary>
/// <remarks>
/// Each level is a separate channel because a host distinguishes them: the CLI renders a warning as
/// a <c>Warning:</c> line <i>and</i> reports it as a machine event, while narration is a line it may
/// route elsewhere or suppress entirely. Nothing here writes to <see cref="Console"/> — a library
/// that does reaches around whatever the host has established, which is how narration ended up on
/// the stdout that <c>--json</c> reserves for its document.
/// </remarks>
internal sealed class SinkLogger(Action<string>? warningSink = null, Action<string>? narrationSink = null) : ILogger
{
    private readonly Action<string> _warning = warningSink ?? Data.CsvDataLoader.NoLog;
    private readonly Action<string> _narration = narrationSink ?? Data.CsvDataLoader.NoLog;

    public void Info(string message) => _narration(message);
    public void Warning(string message) => _warning(message);
    public void Debug(string message) => _narration(message);

    // An error a host does not take is still an error: it reaches the caller as a thrown exception
    // from the operation that raised it, so dropping the line here loses nothing but a duplicate.
    public void Error(string message) => _narration(message);
    public void Error(string message, Exception exception) => _narration($"{message}{Environment.NewLine}{exception}");
}
