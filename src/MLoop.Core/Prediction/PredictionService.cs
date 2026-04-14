using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;

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
    /// Loads the model from disk on every call — use the <see cref="Predict(Dictionary{string, object}[], InputSchemaInfo, ITransformer, string, string?)"/>
    /// overload with a pre-loaded model for warm-path scenarios (e.g. API serve with model caching).
    /// </summary>
    public PredictionResult Predict(
        Dictionary<string, object>[] rows,
        InputSchemaInfo schema,
        string modelPath,
        string taskType,
        string? labelColumn = null)
    {
        if (rows.Length == 0)
        {
            return new PredictionResult
            {
                TaskType = taskType,
                Rows = new List<PredictionRow>(),
                Warnings = new List<string> { "No input rows provided." }
            };
        }

        var model = _mlContext.Model.Load(modelPath, out _);
        return Predict(rows, schema, model, taskType, labelColumn);
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
        string? labelColumn = null)
    {
        ArgumentNullException.ThrowIfNull(model);

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

        MapCategoricalValues(rows, schema, warnings);

        bool needsDummyLabel = IsLabelRequiredForTransform(taskType);
        using var csvStream = CsvStreamBuilder.Build(rows, schema, injectDummyLabel: needsDummyLabel);
        var dataSource = new StreamDataSource(csvStream);

        var loaderOptions = BuildTextLoaderOptions(schema, taskType, labelColumn);
        var textLoader = _mlContext.Data.CreateTextLoader(loaderOptions);
        var dataView = textLoader.Load(dataSource);

        IDataView predictions;
        try
        {
            predictions = model.Transform(dataView);
        }
        catch (Exception ex) when (ex.Message.Contains("Schema mismatch") ||
                                   ex.Message.Contains("Vector<Single"))
        {
            throw new InvalidOperationException(
                $"Feature vector dimension mismatch during prediction. " +
                $"This typically occurs when text columns (FeaturizeText/TF-IDF) in the prediction data " +
                $"have different value distributions than the training data. " +
                $"Workaround: Force text columns to use categorical encoding via column_overrides in mloop.yaml. " +
                $"Original error: {ex.Message}", ex);
        }

        predictions = RestoreOriginalLabels(predictions);

        var result = ExtractResults(predictions, taskType);

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
                "multiclass-classification" or "text-classification" or "image-classification"
                    => DataKind.String, // MapValueToKey needs String
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

    private static DataKind MapDataTypeToDataKind(string dataType) => dataType switch
    {
        "Numeric" => DataKind.Single,
        "Categorical" => DataKind.String,
        "Text" => DataKind.String,
        "Boolean" => DataKind.Boolean,
        _ => DataKind.String
    };

    /// <summary>
    /// For classification tasks, restores original label values from Key type using MapKeyToValue.
    /// </summary>
    internal IDataView RestoreOriginalLabels(IDataView predictions)
    {
        var predictedLabelCol = predictions.Schema.GetColumnOrNull("PredictedLabel");
        if (predictedLabelCol.HasValue && predictedLabelCol.Value.Type is KeyDataViewType)
        {
            var keyToValue = _mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel");
            predictions = keyToValue.Fit(predictions).Transform(predictions);
        }
        return predictions;
    }

    /// <summary>
    /// Eagerly materializes all prediction rows from the IDataView using cursor iteration.
    /// </summary>
    internal static PredictionResult ExtractResults(IDataView predictions, string taskType)
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
            rows = ExtractRegressionRows(cursor, scoreCol);
        }
        else if (taskType == "clustering")
        {
            rows = ExtractClusteringRows(cursor, predictedLabelCol, scoreCol);
        }
        else if (taskType == "anomaly-detection")
        {
            rows = ExtractAnomalyRows(cursor, predictedLabelCol, scoreCol);
        }
        else
        {
            // ranking, recommendation, etc. — just score
            rows = ExtractRegressionRows(cursor, scoreCol);
        }

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
        ValueGetter<VBuffer<float>>? scoreGetter = null;
        ValueGetter<float>? probGetter = null;

        if (predictedLabelCol.HasValue)
            labelGetter = cursor.GetGetter<ReadOnlyMemory<char>>(predictedLabelCol.Value);
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
        DataViewSchema.Column? scoreCol)
    {
        var rows = new List<PredictionRow>();

        ValueGetter<float>? scoreGetter = null;
        if (scoreCol.HasValue)
            scoreGetter = cursor.GetGetter<float>(scoreCol.Value);

        while (cursor.MoveNext())
        {
            double? score = null;
            if (scoreGetter != null)
            {
                float scoreValue = 0;
                scoreGetter(ref scoreValue);
                score = scoreValue;
            }

            rows.Add(new PredictionRow { Score = score });
        }

        return rows;
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

    private static bool IsClassificationTask(string taskType) =>
        taskType is "binary-classification" or "multiclass-classification"
            or "text-classification" or "image-classification";

    private static bool IsLabelRequiredForTransform(string taskType) =>
        IsClassificationTask(taskType) || taskType == "ranking";
}
