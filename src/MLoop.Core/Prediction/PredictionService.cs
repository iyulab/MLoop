using System.Globalization;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.AutoML;
using MLoop.Core.Models;
using MLoop.Core.Runtime;
using MLoop.Core.Storage;

namespace MLoop.Core.Prediction;

/// <summary>
/// Shared prediction engine used by both CLI and API.
/// Handles the full pipeline: categorical mapping → CSV → TextLoader → model.Transform → result extraction.
/// </summary>
public class PredictionService
{
    private readonly MLContext _mlContext;

    public PredictionService(MLContext? mlContext = null)
    {
        _mlContext = mlContext ?? new MLContext(seed: 42);
    }

    /// <summary>
    /// Runs prediction on in-memory row data using a saved ML.NET model file.
    /// Loads the model from disk on every call — use the <see cref="Predict(Dictionary{string, object}[], InputSchemaInfo, ITransformer, string, string?, RegressionInterval?, ITransformer?)"/>
    /// overload with a pre-loaded model for warm-path scenarios (e.g. API serve with model caching).
    /// </summary>
    public PredictionResult Predict(
        Dictionary<string, object>[] rows,
        InputSchemaInfo schema,
        string modelPath,
        string taskType,
        string? labelColumn = null,
        RegressionInterval? interval = null)
    {
        RejectRowBasedForecasting(taskType);

        if (rows.Length == 0)
        {
            return new PredictionResult
            {
                TaskType = taskType,
                Rows = new List<PredictionRow>(),
                Warnings = new List<string> { "No input rows provided." }
            };
        }

        // DL tasks need their native runtime loaded before deserializing the model (BUG-40).
        RuntimeManager.EnsureRuntimeForTask(taskType);

        var model = _mlContext.Model.Load(modelPath, out _);

        // ② regression wave (heteroscedastic): load the sibling σ-model for per-row band widths when the
        // interval carries the normalized-conformal parameters and the file was promoted alongside.
        ITransformer? residualModel = null;
        if (interval?.IsHeteroscedastic == true)
        {
            var residualPath = Path.Combine(Path.GetDirectoryName(modelPath) ?? ".", ExperimentLayout.ResidualModelFileName);
            if (File.Exists(residualPath))
                residualModel = _mlContext.Model.Load(residualPath, out _);
        }

        return Predict(rows, schema, model, taskType, labelColumn, interval, residualModel);
    }

    /// <summary>
    /// Runs prediction using a pre-loaded <see cref="ITransformer"/>. Enables the caller to cache
    /// model instances across requests (e.g. mloop serve model cache).
    /// </summary>
    /// <remarks>
    /// <see cref="ITransformer.Transform"/> is thread-safe, but <see cref="MLContext"/> is not —
    /// callers sharing an ITransformer across threads must still use a per-request MLContext.
    /// </remarks>
    public PredictionResult Predict(
        Dictionary<string, object>[] rows,
        InputSchemaInfo schema,
        ITransformer model,
        string taskType,
        string? labelColumn = null,
        RegressionInterval? interval = null,
        ITransformer? residualModel = null)
    {
        ArgumentNullException.ThrowIfNull(model);

        RejectRowBasedForecasting(taskType);

        if (rows.Length == 0)
        {
            return new PredictionResult
            {
                TaskType = taskType,
                Rows = new List<PredictionRow>(),
                Warnings = new List<string> { "No input rows provided." }
            };
        }

        var warnings = new List<string>();

        // Silent-GIGO guard: if the input rows share NO column with the trained schema (e.g. the
        // caller wrapped rows in an envelope like {"rows":[...]} or posted an unrelated payload),
        // every schema column falls back to its default value and the model emits an all-zero score
        // whose argmax is a fabricated label — returned with 200 and no warning. Fail fast instead:
        // partial column overlap stays legitimate (missing values are defaulted below), but zero
        // overlap is unambiguous caller error.
        RequireSchemaColumnOverlap(rows, schema);

        MapCategoricalValues(rows, schema, warnings);

        bool needsDummyLabel = IsLabelRequiredForTransform(taskType);
        using var csvStream = CsvStreamBuilder.Build(rows, schema, injectDummyLabel: needsDummyLabel);
        var dataSource = new StreamDataSource(csvStream);

        var loaderOptions = BuildTextLoaderOptions(schema, taskType, labelColumn);
        var textLoader = _mlContext.Data.CreateTextLoader(loaderOptions);
        var dataView = textLoader.Load(dataSource);

        // The manual-pipeline tasks (anomaly/clustering/time-series-anomaly) train on data loaded as a
        // single "Features" vector, so their saved model expects a "Features" input column (D8). The CLI
        // predict path gets this for free from InferColumns, but here we load the schema's individual
        // named columns — so concatenate them into "Features" before Transform, or the model throws
        // "Could not find input column 'Features'". Tasks whose model featurizes named columns internally
        // (AutoML binary/multiclass/regression) are left untouched.
        var transformInput = EnsureFeaturesColumn(dataView, schema, taskType, labelColumn);

        IDataView predictions;
        try
        {
            predictions = model.Transform(transformInput);
        }
        catch (Exception ex) when (ex.Message.Contains("Schema mismatch") ||
                                   ex.Message.Contains("Vector<Single"))
        {
            throw new InvalidOperationException(
                $"Feature vector dimension mismatch during prediction. " +
                $"The prediction data's columns don't match the schema the model was trained on " +
                $"(a feature column may be missing, renamed, or have a different type). The saved model " +
                $"embeds its fitted featurizers, so this is a column-structure mismatch, not a text/value " +
                $"distribution issue. Ensure the prediction data has the same columns (names and types) as " +
                $"the training data. Original error: {ex.Message}", ex);
        }

        // ② regression wave (heteroscedastic): the σ-model reads the "Features" column the main model
        // just produced, so score it here (before RestoreOriginalLabels touches label columns) and
        // materialize the per-row σ in the same row order the result cursor will iterate. A failure
        // (e.g. a missing Features column) degrades gracefully to the constant-width band.
        IReadOnlyList<double>? perRowSigma = null;
        if (residualModel != null && interval?.IsHeteroscedastic == true)
        {
            try { perRowSigma = ComputeResidualSigma(residualModel, predictions); }
            catch { perRowSigma = null; }
        }

        predictions = RestoreOriginalLabels(predictions);

        var result = ExtractResults(predictions, taskType, interval, perRowSigma);

        if (warnings.Count > 0)
        {
            result = new PredictionResult
            {
                TaskType = result.TaskType,
                Rows = result.Rows,
                Metadata = result.Metadata,
                Warnings = (result.Warnings ?? new List<string>()).Concat(warnings).ToList()
            };
        }

        return result;
    }

    /// <summary>
    /// Maps unknown categorical values to known training values in-place.
    /// </summary>
    public static void MapCategoricalValues(
        Dictionary<string, object>[] rows,
        InputSchemaInfo schema,
        List<string> warnings)
    {
        var categoricalColumns = schema.Columns
            .Where(c => c.CategoricalValues != null && c.CategoricalValues.Count > 0)
            .ToList();

        if (categoricalColumns.Count == 0) return;

        foreach (var col in categoricalColumns)
        {
            var knownValues = new HashSet<string>(col.CategoricalValues!);
            var fallbackValue = col.CategoricalValues![0]; // Most frequent value

            foreach (var row in rows)
            {
                if (!row.TryGetValue(col.Name, out var rawValue) || rawValue == null)
                    continue;

                var strValue = rawValue.ToString() ?? "";
                if (!string.IsNullOrEmpty(strValue) && !knownValues.Contains(strValue))
                {
                    row[col.Name] = fallbackValue;
                    warnings.Add($"Column '{col.Name}': unknown value '{strValue}' replaced with '{fallbackValue}'");
                }
            }
        }
    }

    /// <summary>
    /// Builds TextLoader.Options from the trained schema, handling task-specific label type requirements.
    /// </summary>
    public TextLoader.Options BuildTextLoaderOptions(
        InputSchemaInfo schema,
        string taskType,
        string? labelColumn)
    {
        var activeColumns = schema.Columns
            .Where(c => !c.Purpose.Equals("Exclude", StringComparison.OrdinalIgnoreCase))
            .ToList();

        var textColumns = new List<TextLoader.Column>();

        for (int i = 0; i < activeColumns.Count; i++)
        {
            var col = activeColumns[i];
            var dataKind = ResolveDataKind(col, taskType, labelColumn);

            textColumns.Add(new TextLoader.Column(col.Name, dataKind, i));
        }

        return new TextLoader.Options
        {
            Columns = textColumns.ToArray(),
            HasHeader = true,
            AllowQuoting = true,
            Separators = new[] { ',' },
        };
    }

    /// <summary>
    /// Resolves the ML.NET DataKind for a column, applying task-specific label type rules.
    /// BUG-11/12: Use schema dataType, NOT InferColumns.
    /// BUG-15/17/23: Label type depends on task type.
    /// </summary>
    /// <summary>
    /// Throws when the input rows contain none of the trained schema's non-label input columns —
    /// with zero overlap every column defaults and the model returns a fabricated all-zero-score
    /// prediction with 200/no-warning (silent garbage-in-garbage-out, D20). Partial overlap is
    /// legitimate (missing values default) and stays untouched.
    /// </summary>
    /// <summary>
    /// Throws for forecasting models: the SSA forecaster is stateful (it forecasts a fixed horizon
    /// ahead of its training series), so a stateless per-row Transform extracts nothing — every
    /// output field comes back null while the response still reports success (silent no-op, D21 —
    /// same silent-failure family as the D20 zero-overlap guard). Fail fast with the working path
    /// instead of fabricating an all-null 200.
    /// </summary>
    private static void RejectRowBasedForecasting(string taskType)
    {
        if (string.Equals(taskType, "forecasting", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                "Forecasting models do not support row-based prediction: the SSA forecaster is stateful and " +
                "forecasts a fixed horizon ahead of its training series, so transforming posted rows yields no output. " +
                "Use 'mloop predict' (optionally with --json) to generate the horizon forecast with confidence bounds.");
        }
    }

    private static void RequireSchemaColumnOverlap(Dictionary<string, object>[] rows, InputSchemaInfo schema)
    {
        var inputColumns = schema.Columns
            .Where(c => !string.Equals(c.Purpose, "Label", StringComparison.OrdinalIgnoreCase))
            .Select(c => c.Name)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        if (inputColumns.Count == 0)
            return;

        if (rows.Any(r => r.Keys.Any(inputColumns.Contains)))
            return;

        var provided = rows.SelectMany(r => r.Keys).Distinct(StringComparer.OrdinalIgnoreCase).Take(8).ToList();
        throw new ArgumentException(
            $"Input rows contain none of the model's input columns. " +
            $"Expected columns like [{string.Join(", ", inputColumns.Take(8))}] but got [{string.Join(", ", provided)}]. " +
            $"If calling the API, POST a bare JSON array of row objects (e.g. [{{\"col\":1}}]) — not an envelope like {{\"rows\":[...]}}.");
    }

    private static DataKind ResolveDataKind(ColumnSchema col, string taskType, string? labelColumn)
    {
        bool isLabel = labelColumn != null &&
                       col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase);

        if (isLabel)
        {
            return taskType switch
            {
                "binary-classification" => col.DataType switch
                {
                    "Boolean" => DataKind.Boolean,
                    _ => DataKind.String
                },
                // D14: a multiclass label can be numeric-looking (e.g. KAMP class ids "0"/"1"/"2"), in
                // which case train-time schema inference records DataType=Numeric and AutoML's fitted
                // pipeline embeds MapValueToKey over a Single-typed column, not String. Forcing String
                // unconditionally mismatched that trained schema and made model.Transform throw
                // "Could not apply a map over type 'Single' to column '...' since it has type 'String'".
                "multiclass-classification" or "text-classification" or "image-classification"
                    => MapDataTypeToDataKind(col.DataType),
                "regression" or "forecasting" => col.DataType switch
                {
                    "Boolean" => DataKind.Single, // BUG-23
                    _ => DataKind.Single
                },
                _ => MapDataTypeToDataKind(col.DataType)
            };
        }

        return MapDataTypeToDataKind(col.DataType);
    }

    private static DataKind MapDataTypeToDataKind(string dataType) =>
        SchemaDataTypes.ToDataKind(dataType, fallback: DataKind.String);

    /// <summary>
    /// For classification tasks, restores original label values from Key type using MapKeyToValue.
    /// </summary>
    internal IDataView RestoreOriginalLabels(IDataView predictions)
    {
        var predictedLabelCol = predictions.Schema.GetColumnOrNull("PredictedLabel");
        // Only classification keys carry a KeyValues mapping back to the original label strings.
        // Clustering's PredictedLabel is also a key (the cluster id 1..k) but has NO KeyValues
        // annotation — applying MapKeyToValue to it throws "Metadata KeyValues does not exist", which
        // broke `mloop predict` for clustering. Guard on the annotation actually being present.
        if (predictedLabelCol.HasValue && predictedLabelCol.Value.Type is KeyDataViewType
            && HasKeyValues(predictedLabelCol.Value))
        {
            var keyToValue = _mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel");
            predictions = keyToValue.Fit(predictions).Transform(predictions);
        }
        return predictions;
    }

    /// <summary>
    /// Whether a key-typed column carries a <c>KeyValues</c> annotation (the map from key id back to
    /// the original value). Classification's PredictedLabel has it; clustering's cluster-id key does
    /// not, so a MapKeyToValue caller must skip it for the latter (else ML.NET throws
    /// "Metadata KeyValues does not exist"). Shared by <see cref="RestoreOriginalLabels"/> and the
    /// CLI's PredictionEngine so the guard cannot drift between the two predict paths.
    /// </summary>
    public static bool HasKeyValues(DataViewSchema.Column column)
        => column.Annotations.Schema.GetColumnOrNull("KeyValues") is not null;

    /// <summary>
    /// Eagerly materializes all prediction rows from the IDataView using cursor iteration.
    /// </summary>
    internal static PredictionResult ExtractResults(IDataView predictions, string taskType, RegressionInterval? interval = null, IReadOnlyList<double>? perRowSigma = null)
    {
        var rows = new List<PredictionRow>();
        var schema = predictions.Schema;

        var predictedLabelCol = schema.GetColumnOrNull("PredictedLabel");
        var scoreCol = schema.GetColumnOrNull("Score");
        var probabilityCol = schema.GetColumnOrNull("Probability");

        using var cursor = predictions.GetRowCursor(schema);

        if (IsClassificationTask(taskType))
        {
            rows = ExtractClassificationRows(cursor, predictedLabelCol, scoreCol, probabilityCol);
        }
        else if (taskType is "regression" or "forecasting")
        {
            rows = ExtractRegressionRows(cursor, scoreCol, interval, perRowSigma);
        }
        else if (taskType == "clustering")
        {
            rows = ExtractClusteringRows(cursor, predictedLabelCol, scoreCol);
        }
        else if (taskType == "anomaly-detection")
        {
            rows = ExtractAnomalyRows(cursor, predictedLabelCol, scoreCol);
        }
        else if (taskType == "time-series-anomaly")
        {
            rows = ExtractTimeSeriesAnomalyRows(cursor, schema);
        }
        else
        {
            // ranking, recommendation, etc. — just score (no conformal band; interval is regression-only)
            rows = ExtractRegressionRows(cursor, scoreCol);
        }

        // Single authority for the normalized per-row confidence — computed once here so every consumer
        // of PredictionService (serve, CLI) gets the same value and none re-derives it (ConfidencePolicy).
        double? residualStd = interval?.ResidualStd;
        for (int r = 0; r < rows.Count; r++)
            rows[r] = rows[r] with { Confidence = ConfidencePolicy.Compute(rows[r], taskType, residualStd) };

        return new PredictionResult
        {
            TaskType = taskType,
            Rows = rows
        };
    }

    private static List<PredictionRow> ExtractClassificationRows(
        DataViewRowCursor cursor,
        DataViewSchema.Column? predictedLabelCol,
        DataViewSchema.Column? scoreCol,
        DataViewSchema.Column? probabilityCol)
    {
        var rows = new List<PredictionRow>();

        ValueGetter<ReadOnlyMemory<char>>? labelGetter = null;
        ValueGetter<bool>? boolLabelGetter = null;
        ValueGetter<float>? singleLabelGetter = null;
        ValueGetter<VBuffer<float>>? scoreGetter = null;
        ValueGetter<float>? probGetter = null;

        // PredictedLabel type varies by classification family: multiclass/text/image map their Key back
        // to the type MapValueToKey was originally fit on (String for categorical labels, but Single for
        // a numeric-looking label like KAMP class ids "0"/"1"/"2" — D14), while binary-classification
        // outputs a raw Boolean (True=positive). Reading a column with the wrong getter throws
        // "Invalid TValue: <actual>, expected <requested>" — the crash that made serve /predict fail for
        // every binary model (D13) and every numeric-labeled multiclass model (D14); the CLI predict path
        // renders the label itself and so never hit this. Pick the getter by the actual column type.
        if (predictedLabelCol.HasValue)
        {
            if (predictedLabelCol.Value.Type is BooleanDataViewType)
                boolLabelGetter = cursor.GetGetter<bool>(predictedLabelCol.Value);
            else if (predictedLabelCol.Value.Type == NumberDataViewType.Single)
                singleLabelGetter = cursor.GetGetter<float>(predictedLabelCol.Value);
            else
                labelGetter = cursor.GetGetter<ReadOnlyMemory<char>>(predictedLabelCol.Value);
        }
        if (scoreCol.HasValue && scoreCol.Value.Type is VectorDataViewType)
            scoreGetter = cursor.GetGetter<VBuffer<float>>(scoreCol.Value);
        if (probabilityCol.HasValue)
            probGetter = cursor.GetGetter<float>(probabilityCol.Value);

        while (cursor.MoveNext())
        {
            string? label = null;
            Dictionary<string, double>? probabilities = null;
            double? probability = null;

            if (labelGetter != null)
            {
                ReadOnlyMemory<char> labelValue = default;
                labelGetter(ref labelValue);
                label = labelValue.ToString();
            }
            else if (boolLabelGetter != null)
            {
                bool boolValue = false;
                boolLabelGetter(ref boolValue);
                label = boolValue ? "True" : "False";
            }
            else if (singleLabelGetter != null)
            {
                float singleValue = 0f;
                singleLabelGetter(ref singleValue);
                label = singleValue.ToString(CultureInfo.InvariantCulture);
            }

            if (scoreGetter != null)
            {
                VBuffer<float> scoreBuffer = default;
                scoreGetter(ref scoreBuffer);
                var scores = scoreBuffer.DenseValues().ToArray();
                probabilities = new Dictionary<string, double>();
                for (int i = 0; i < scores.Length; i++)
                {
                    probabilities[$"class_{i}"] = scores[i];
                }
            }

            if (probGetter != null)
            {
                float prob = 0;
                probGetter(ref prob);
                probability = prob;
            }

            // binary-classification returns a single scalar Probability = P(positive/True), not a
            // per-class Score vector (that path only fires for multiclass, where Score is a vector).
            // Expose both class probabilities so consumers reading a probability distribution
            // (e.g. confidence = max class probability) compute it correctly for the negative class too —
            // otherwise a confident "False" (P(True)≈0.02) would read as low confidence.
            if (boolLabelGetter != null && probabilities is null && probability.HasValue)
            {
                probabilities = new Dictionary<string, double>
                {
                    ["True"] = probability.Value,
                    ["False"] = 1.0 - probability.Value,
                };
            }

            rows.Add(new PredictionRow
            {
                PredictedLabel = label,
                Probabilities = probabilities,
                Score = probability
            });
        }

        return rows;
    }

    private static List<PredictionRow> ExtractRegressionRows(
        DataViewRowCursor cursor,
        DataViewSchema.Column? scoreCol,
        RegressionInterval? interval = null,
        IReadOnlyList<double>? perRowSigma = null)
    {
        var rows = new List<PredictionRow>();

        ValueGetter<float>? scoreGetter = null;
        if (scoreCol.HasValue)
            scoreGetter = cursor.GetGetter<float>(scoreCol.Value);

        int i = 0;
        while (cursor.MoveNext())
        {
            double? score = null;
            if (scoreGetter != null)
            {
                float scoreValue = 0;
                scoreGetter(ref scoreValue);
                score = scoreValue;
            }

            // ② regression wave: wrap Score in the conformal band when the model carries one. The width
            // is per-row (q·σ(x)) when the heteroscedastic σ-model scored this row, else constant.
            double? lower = null, upper = null, confidence = null;
            if (score.HasValue && interval != null)
            {
                double half = perRowSigma != null && i < perRowSigma.Count
                    ? interval.WidthFor(perRowSigma[i])
                    : interval.HalfWidth;
                lower = score.Value - half;
                upper = score.Value + half;
                confidence = interval.Confidence;
            }

            rows.Add(new PredictionRow
            {
                Score = score,
                ScoreLowerBound = lower,
                ScoreUpperBound = upper,
                IntervalConfidence = confidence
            });
            i++;
        }

        return rows;
    }

    /// <summary>
    /// ② regression wave (heteroscedastic): scores the σ(x) residual model over the main model's output
    /// (which carries the "Features" column the σ-model reads) and materializes the raw σ per row, in the
    /// same order a later cursor over <paramref name="scoredByMainModel"/> iterates. The σ-model appends
    /// its own "Score" column (the predicted residual magnitude), which shadows the main model's Score in
    /// the transformed view — that latest "Score" is exactly what we read here.
    /// </summary>
    private List<double> ComputeResidualSigma(ITransformer residualModel, IDataView scoredByMainModel)
    {
        var sigmaView = residualModel.Transform(scoredByMainModel);
        var sigmaCol = sigmaView.Schema.GetColumnOrNull("Score")
            ?? throw new InvalidOperationException("Residual σ-model produced no Score column.");
        var sigmas = new List<double>();
        using var cursor = sigmaView.GetRowCursor(new[] { sigmaCol });
        var getter = cursor.GetGetter<float>(sigmaCol);
        float v = 0;
        while (cursor.MoveNext())
        {
            getter(ref v);
            sigmas.Add(v);
        }
        return sigmas;
    }

    private static List<PredictionRow> ExtractClusteringRows(
        DataViewRowCursor cursor,
        DataViewSchema.Column? predictedLabelCol,
        DataViewSchema.Column? scoreCol)
    {
        var rows = new List<PredictionRow>();

        ValueGetter<uint>? labelGetter = null;
        ValueGetter<VBuffer<float>>? scoreGetter = null;

        if (predictedLabelCol.HasValue)
            labelGetter = cursor.GetGetter<uint>(predictedLabelCol.Value);
        if (scoreCol.HasValue && scoreCol.Value.Type is VectorDataViewType)
            scoreGetter = cursor.GetGetter<VBuffer<float>>(scoreCol.Value);

        while (cursor.MoveNext())
        {
            int? clusterId = null;
            double[]? distances = null;

            if (labelGetter != null)
            {
                uint labelValue = 0;
                labelGetter(ref labelValue);
                clusterId = (int)labelValue;
            }

            if (scoreGetter != null)
            {
                VBuffer<float> scoreBuffer = default;
                scoreGetter(ref scoreBuffer);
                distances = scoreBuffer.DenseValues().Select(v => (double)v).ToArray();
            }

            rows.Add(new PredictionRow { ClusterId = clusterId, Distances = distances });
        }

        return rows;
    }

    private static List<PredictionRow> ExtractAnomalyRows(
        DataViewRowCursor cursor,
        DataViewSchema.Column? predictedLabelCol,
        DataViewSchema.Column? scoreCol)
    {
        var rows = new List<PredictionRow>();

        ValueGetter<bool>? labelGetter = null;
        ValueGetter<float>? scoreGetter = null;

        if (predictedLabelCol.HasValue)
            labelGetter = cursor.GetGetter<bool>(predictedLabelCol.Value);
        if (scoreCol.HasValue)
            scoreGetter = cursor.GetGetter<float>(scoreCol.Value);

        while (cursor.MoveNext())
        {
            bool? isAnomaly = null;
            double? anomalyScore = null;

            if (labelGetter != null)
            {
                bool labelValue = false;
                labelGetter(ref labelValue);
                isAnomaly = labelValue;
            }

            if (scoreGetter != null)
            {
                float scoreValue = 0;
                scoreGetter(ref scoreValue);
                anomalyScore = scoreValue;
            }

            rows.Add(new PredictionRow { IsAnomaly = isAnomaly, AnomalyScore = anomalyScore });
        }

        return rows;
    }

    /// <summary>
    /// Time-series anomaly models emit a single vector column (<c>Prediction</c> = [alert, raw score,
    /// detector-specific third slot]) instead of PredictedLabel/Score — the generic score-based
    /// extraction read none of it and returned all-null rows with a 200/exit-0 (silent no-op, D22,
    /// same family as D20/D21). Map the authoritative slots onto the anomaly row fields.
    /// </summary>
    private static List<PredictionRow> ExtractTimeSeriesAnomalyRows(
        DataViewRowCursor cursor, DataViewSchema schema)
    {
        var rows = new List<PredictionRow>();

        var predCol = schema.GetColumnOrNull(TimeSeriesAnomalyOutput.PredictionColumnName);
        ValueGetter<VBuffer<double>>? predGetter = predCol.HasValue
            ? cursor.GetGetter<VBuffer<double>>(predCol.Value)
            : null;

        while (cursor.MoveNext())
        {
            bool? isAnomaly = null;
            double? rawScore = null;

            if (predGetter != null)
            {
                VBuffer<double> pred = default;
                predGetter(ref pred);
                var values = pred.DenseValues().ToArray();
                if (values.Length > TimeSeriesAnomalyOutput.AlertSlot)
                    isAnomaly = values[TimeSeriesAnomalyOutput.AlertSlot] != 0;
                if (values.Length > TimeSeriesAnomalyOutput.RawScoreSlot)
                    rawScore = values[TimeSeriesAnomalyOutput.RawScoreSlot];
            }

            rows.Add(new PredictionRow { IsAnomaly = isAnomaly, AnomalyScore = rawScore });
        }

        return rows;
    }

    private static bool IsClassificationTask(string taskType) =>
        taskType is "binary-classification" or "multiclass-classification"
            or "text-classification" or "image-classification";

    private static bool IsLabelRequiredForTransform(string taskType) =>
        IsClassificationTask(taskType) || taskType == "ranking";

    /// <summary>
    /// Tasks whose saved model expects a single <c>Features</c> vector input rather than the raw named
    /// feature columns. This covers two groups: (1) the manually-built pipelines
    /// (<c>Concatenate("Features", …) → trainer(featureColumnName:"Features")</c>) for
    /// anomaly/clustering/time-series-anomaly (D8); and (2) AutoML tabular tasks
    /// (binary/multiclass/regression), whose <c>InferColumns</c> loads all numeric features into one
    /// <c>Features</c> vector at train time — so the fitted model's first transform reads <c>Features</c>,
    /// not the individual columns. The CLI predict path gets <c>Features</c> for free from
    /// <c>InferColumns</c>; serve loads named scalar columns and so must build it (D12 — otherwise binary
    /// serve <c>/predict</c> throws "Could not find input column 'Features'"). The
    /// <see cref="EnsureFeaturesColumn"/> guard no-ops when a <c>Features</c> column already exists, so
    /// models that instead featurize named columns internally are unaffected.
    /// </summary>
    private static bool RequiresFeaturesVectorInput(string taskType) =>
        taskType is "anomaly-detection" or "clustering" or "time-series-anomaly"
                 or "regression" or "forecasting" or "ranking"
        || IsClassificationTask(taskType);

    /// <summary>
    /// For models that expect a single <c>Features</c> vector input (see <see cref="RequiresFeaturesVectorInput"/>),
    /// concatenates the schema's numeric feature columns into <c>Features</c> when the loaded view doesn't
    /// already have one — matching the single-vector shape the model was trained on (D8). No-op for other tasks.
    /// </summary>
    private IDataView EnsureFeaturesColumn(IDataView data, InputSchemaInfo schema, string taskType, string? labelColumn)
    {
        if (!RequiresFeaturesVectorInput(taskType)) return data;
        if (data.Schema.GetColumnOrNull("Features") is not null) return data;

        // Select numeric, non-label columns — mirroring how RunAnomalyDetection/RunClustering pick
        // feature columns at train time. Driven by DataType (not Purpose), since Purpose may be absent
        // when the schema was deserialized from config.json, whereas DataType is always populated.
        var featureColumns = schema.Columns
            .Where(c => c.DataType.Equals(SchemaDataTypes.Numeric, StringComparison.OrdinalIgnoreCase)
                     && !c.Purpose.Equals("Exclude", StringComparison.OrdinalIgnoreCase)
                     && (labelColumn is null || !c.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase)))
            .Select(c => c.Name)
            .ToArray();

        if (featureColumns.Length == 0) return data;

        return _mlContext.Transforms.Concatenate("Features", featureColumns).Fit(data).Transform(data);
    }
}
