using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Engine for evaluating trained models on test data
/// </summary>
public class EvaluationEngine
{
    private readonly MLContext _mlContext;

    public EvaluationEngine()
    {
        _mlContext = new MLContext(seed: 42);
    }

    /// <summary>
    /// Evaluates a trained model on test data
    /// </summary>
    public async Task<Dictionary<string, double>> EvaluateAsync(
        string modelPath,
        string testDataPath,
        string labelColumn,
        string taskType,
        CancellationToken cancellationToken = default,
        InputSchemaInfo? trainedSchema = null,
        string? groupColumn = null,
        string? userColumn = null,
        string? itemColumn = null)
    {
        return await Task.Run(() =>
        {
            List<string> tempFiles = new();
            try
            {
                // DL tasks need their native runtime loaded before deserializing the model (BUG-40).
                MLoop.Core.Runtime.RuntimeManager.EnsureRuntimeForTask(taskType);

                // Load the trained model
                var trainedModel = _mlContext.Model.Load(modelPath, out var modelSchema);

                // Directory-based tasks (image classification, object detection) consume an image
                // directory, not a CSV — load via the matching directory loader and score before the
                // CSV-oriented LoadTestData path. Object detection is scored by mAP over predicted
                // boxes; image classification is multiclass over the loader's "Label" column (which
                // the training pipeline maps to a key, same as the CSV image-classification branch).
                if (MLoop.Core.Data.DataLoaderFactory.IsDirectoryBased(taskType))
                {
                    var dirData = MLoop.Core.Data.DataLoaderFactory.Create(taskType, _mlContext)
                        .LoadData(testDataPath, labelColumn, taskType);
                    var dirScored = trainedModel.Transform(dirData);

                    if (taskType.Equals("object-detection", StringComparison.OrdinalIgnoreCase))
                        return MLoop.Core.Evaluation.ObjectDetectionEvaluator.Evaluate(_mlContext, dirScored);

                    return EvaluateMulticlassClassification(dirScored, "Label");
                }

                // F-26: ranking/recommendation models reference the group/user/item columns
                // individually (MapValueToKey), so they must stay addressable at evaluate time — the
                // same preservation the train and predict paths apply (F-23). Without it InferColumns
                // merges them into the Features range and model.Transform crashes outright.
                var preserveColumns = new[] { groupColumn, userColumn, itemColumn }
                    .Where(c => !string.IsNullOrWhiteSpace(c))
                    .Select(c => c!)
                    .ToList();

                // Load test data (no label renaming - model pipeline transforms label column)
                var testData = LoadTestData(testDataPath, labelColumn, taskType, trainedSchema, preserveColumns, out tempFiles);

                // Make predictions on test data
                // Model pipeline transforms the label column (e.g. MapValueToKey for classification)
                IDataView predictions;
                try
                {
                    predictions = trainedModel.Transform(testData);
                }
                catch (Exception ex) when (ex.Message.Contains("Schema mismatch") ||
                                           ex.Message.Contains("Vector<Single"))
                {
                    throw new InvalidOperationException(
                        $"Feature vector dimension mismatch during evaluation. " +
                        $"The test data's columns don't match the schema the model was trained on " +
                        $"(a feature column may be missing, renamed, or have a different type). The saved model " +
                        $"embeds its fitted featurizers, so this is a column-structure mismatch, not an unseen-value issue. " +
                        $"Ensure the test data has the same columns (names and types) as the training data. " +
                        $"Original error: {ex.Message}", ex);
                }

                // Evaluate based on task type using the original label column name
                // After model.Transform, classification labels are in Key type (via MapValueToKey)
                Dictionary<string, double> metrics;

                if (taskType.Equals("regression", StringComparison.OrdinalIgnoreCase))
                {
                    metrics = EvaluateRegression(predictions, labelColumn);
                }
                else if (taskType.Equals("classification", StringComparison.OrdinalIgnoreCase) ||
                         taskType.Equals("binary-classification", StringComparison.OrdinalIgnoreCase))
                {
                    metrics = EvaluateBinaryClassification(predictions, labelColumn);
                }
                else if (taskType.Equals("multiclass-classification", StringComparison.OrdinalIgnoreCase))
                {
                    metrics = EvaluateMulticlassClassification(predictions, labelColumn);
                }
                else if (taskType.Equals("anomaly-detection", StringComparison.OrdinalIgnoreCase))
                {
                    metrics = EvaluateAnomalyDetection(predictions, labelColumn);
                }
                else if (taskType.Equals("clustering", StringComparison.OrdinalIgnoreCase))
                {
                    metrics = EvaluateClustering(predictions, labelColumn);
                }
                else if (taskType.Equals("ranking", StringComparison.OrdinalIgnoreCase))
                {
                    metrics = EvaluateRanking(predictions, labelColumn, groupColumn);
                }
                else if (taskType.Equals("forecasting", StringComparison.OrdinalIgnoreCase))
                {
                    metrics = EvaluateForecasting(predictions, labelColumn);
                }
                else if (taskType.Equals("time-series-anomaly", StringComparison.OrdinalIgnoreCase))
                {
                    metrics = EvaluateTimeSeriesAnomaly(predictions);
                }
                else if (taskType.Equals("recommendation", StringComparison.OrdinalIgnoreCase))
                {
                    metrics = EvaluateRecommendation(predictions, labelColumn);
                }
                else if (taskType.Equals("image-classification", StringComparison.OrdinalIgnoreCase) ||
                         taskType.Equals("text-classification", StringComparison.OrdinalIgnoreCase) ||
                         taskType.Equals("ner", StringComparison.OrdinalIgnoreCase) ||
                         taskType.Equals("question-answering", StringComparison.OrdinalIgnoreCase))
                {
                    metrics = EvaluateMulticlassClassification(predictions, "Label");
                }
                else if (taskType.Equals("sentence-similarity", StringComparison.OrdinalIgnoreCase))
                {
                    metrics = EvaluateRegression(predictions, labelColumn);
                }
                else
                {
                    throw new NotSupportedException($"Task type '{taskType}' is not supported for evaluation.");
                }

                return metrics;
            }
            finally
            {
                // BUG-13: Clean up temp files AFTER all lazy data consumption is complete
                foreach (var tempFile in tempFiles)
                {
                    if (File.Exists(tempFile))
                    {
                        try { File.Delete(tempFile); } catch (IOException) { }
                    }
                }
            }
        }, cancellationToken);
    }

    private IDataView LoadTestData(string testDataPath, string labelColumn, string taskType, InputSchemaInfo? trainedSchema, IReadOnlyList<string>? preserveColumns, out List<string> tempFiles)
    {
        // Apply the single shared inference preprocessing sequence (encoding → flatten → index →
        // schema-based exclude / data-dependent fallback). This is the same path predict uses, which
        // is what makes the test feature vector reproduce the model's training-time width. Converging
        // here fixed BUG-43 (CP949 encoding) and BUG-44 (index/exclude columns left in) and closed the
        // flatten/constant gaps that the previous per-engine reimplementation hid.
        string loadPath = InferenceDataPreprocessor.Prepare(testDataPath, labelColumn, trainedSchema, out tempFiles);

        // Infer column types from the data
        var columnInference = _mlContext.Auto().InferColumns(
            loadPath,
            labelColumnName: labelColumn,
            separatorChar: ',');

        // BUG-12: Override column types with trained schema to fix InferColumns misdetection
        // BUG-18: Skip override for label column — let InferColumns determine its type naturally.
        if (trainedSchema != null && columnInference.TextLoaderOptions.Columns != null)
        {
            var schemaLookup = trainedSchema.Columns.ToDictionary(c => c.Name, c => c.DataType);
            foreach (var col in columnInference.TextLoaderOptions.Columns)
            {
                if (col.Name == labelColumn)
                    continue;

                if (col.Name != null && schemaLookup.TryGetValue(col.Name, out var expectedType))
                {
                    var expectedKind = expectedType switch
                    {
                        "Numeric" => DataKind.Single,
                        "Categorical" => DataKind.String,
                        "Text" => DataKind.String,
                        "Boolean" => DataKind.Boolean,
                        _ => col.DataKind
                    };
                    if (col.DataKind != expectedKind)
                    {
                        col.DataKind = expectedKind;
                    }
                }
            }
        }

        // BUG-15/BUG-25b: Multiclass classification — InferColumns may detect label as Boolean
        // when values are 0/1/2 etc. CsvDataLoader converts Boolean→String for multiclass
        // during training. Override BEFORE loading so TextLoader can parse all values.
        bool isMulticlass = taskType.Equals("multiclass-classification", StringComparison.OrdinalIgnoreCase);
        bool isRegressionEval = taskType.Equals("regression", StringComparison.OrdinalIgnoreCase);
        if (columnInference.TextLoaderOptions.Columns != null)
        {
            foreach (var col in columnInference.TextLoaderOptions.Columns)
            {
                if (col.Name != null && col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase) && col.DataKind == DataKind.Boolean)
                {
                    if (isMulticlass)
                        col.DataKind = DataKind.String;
                    else if (isRegressionEval)
                        col.DataKind = DataKind.Single;
                }
            }
        }

        // F-26: split the preserved group/user/item columns back out of any merged numeric range so
        // the model's key transforms can find them individually (mirrors CsvDataLoader at train time
        // and PredictionEngine at predict time — the shared ApplyColumnPreservation helper from F-23).
        CsvDataLoader.ApplyColumnPreservation(columnInference, loadPath, preserveColumns);

        // Create text loader with inferred schema
        var textLoader = _mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);

        // Load the data
        var dataView = textLoader.Load(loadPath);

        // BUG-19: Binary classification with string labels (OK/NG, Yes/No, etc.)
        // CsvDataLoader converts string labels to Boolean during training, but
        // EvaluationEngine loads raw CSV where labels are still strings.
        // Apply the same conversion here to match the model's expected input type.
        bool isBinary = taskType.Equals("binary-classification", StringComparison.OrdinalIgnoreCase)
            || taskType.Equals("classification", StringComparison.OrdinalIgnoreCase);

        if (isBinary)
        {
            var labelSchema = dataView.Schema.GetColumnOrNull(labelColumn);
            if (labelSchema.HasValue && labelSchema.Value.Type is TextDataViewType)
            {
                dataView = ConvertStringLabelToBoolean(dataView, labelColumn);
            }
        }

        return dataView;
    }

    /// <summary>
    /// Converts a String label column to Boolean for binary classification evaluation.
    /// Mirrors CsvDataLoader.ConvertStringLabelToBoolean logic.
    /// </summary>
    internal IDataView ConvertStringLabelToBoolean(IDataView dataView, string labelColumn)
    {
        var uniqueValues = new HashSet<string>();
        var labelCol = dataView.Schema[labelColumn];

        using (var cursor = dataView.GetRowCursor(new[] { labelCol }))
        {
            var getter = cursor.GetGetter<ReadOnlyMemory<char>>(labelCol);
            while (cursor.MoveNext() && uniqueValues.Count <= 3)
            {
                ReadOnlyMemory<char> val = default;
                getter(ref val);
                var str = val.ToString().Trim();
                if (!string.IsNullOrEmpty(str))
                    uniqueValues.Add(str);
            }
        }

        if (uniqueValues.Count != 2)
            return dataView;

        var sorted = uniqueValues.OrderBy(v => v, StringComparer.OrdinalIgnoreCase).ToArray();

        var lookupData = _mlContext.Data.LoadFromEnumerable(new[]
        {
            new LabelMapping { Key = sorted[0], Value = false },
            new LabelMapping { Key = sorted[1], Value = true }
        });

        var pipeline = _mlContext.Transforms.Conversion.MapValue(
            labelColumn,
            lookupData,
            lookupData.Schema["Key"],
            lookupData.Schema["Value"],
            labelColumn);

        return pipeline.Fit(dataView).Transform(dataView);
    }

    private sealed class LabelMapping
    {
        public string Key { get; set; } = "";
        public bool Value { get; set; }
    }


    private Dictionary<string, double> EvaluateRegression(IDataView predictions, string labelColumn)
    {
        var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: labelColumn, scoreColumnName: "Score");

        return new Dictionary<string, double>
        {
            { "r_squared", metrics.RSquared },
            { "rmse", metrics.RootMeanSquaredError },
            { "mae", metrics.MeanAbsoluteError },
            { "mse", metrics.MeanSquaredError }
        };
    }

    private Dictionary<string, double> EvaluateBinaryClassification(IDataView predictions, string labelColumn)
    {
        var metrics = _mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: labelColumn, scoreColumnName: "Score");

        return new Dictionary<string, double>
        {
            { "accuracy", metrics.Accuracy },
            { "auc", metrics.AreaUnderRocCurve },
            { "f1_score", metrics.F1Score },
            { "precision", metrics.PositivePrecision },
            { "recall", metrics.PositiveRecall }
        };
    }

    private Dictionary<string, double> EvaluateMulticlassClassification(IDataView predictions, string labelColumn)
    {
        var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: labelColumn, scoreColumnName: "Score");

        return new Dictionary<string, double>
        {
            { "macro_accuracy", metrics.MacroAccuracy },
            { "micro_accuracy", metrics.MicroAccuracy },
            { "log_loss", metrics.LogLoss }
        };
    }

    private Dictionary<string, double> EvaluateRecommendation(IDataView predictions, string labelColumn)
    {
        var metricsDict = new Dictionary<string, double>();

        try
        {
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: labelColumn);
            metricsDict["rmse"] = double.IsNaN(metrics.RootMeanSquaredError) ? 0 : metrics.RootMeanSquaredError;
            metricsDict["mae"] = double.IsNaN(metrics.MeanAbsoluteError) ? 0 : metrics.MeanAbsoluteError;
            metricsDict["r_squared"] = double.IsNaN(metrics.RSquared) ? 0 : metrics.RSquared;
        }
        catch
        {
            // Evaluation may fail with schema mismatch
        }

        return metricsDict;
    }

    private Dictionary<string, double> EvaluateTimeSeriesAnomaly(IDataView predictions)
    {
        var metricsDict = new Dictionary<string, double>();

        var predCol = predictions.Schema.GetColumnOrNull("Prediction");
        if (predCol.HasValue)
        {
            long anomalyCount = 0;
            long totalCount = 0;

            using var cursor = predictions.GetRowCursor(new[] { predCol.Value });
            var getter = cursor.GetGetter<VBuffer<double>>(predCol.Value);

            while (cursor.MoveNext())
            {
                VBuffer<double> pred = default;
                getter(ref pred);
                var values = pred.DenseValues().ToArray();
                totalCount++;
                if (values.Length > 0 && values[0] != 0)
                    anomalyCount++;
            }

            metricsDict["anomaly_count"] = anomalyCount;
            metricsDict["total_count"] = totalCount;
            metricsDict["detection_rate"] = totalCount > 0 ? (double)anomalyCount / totalCount : 0;
        }

        return metricsDict;
    }

    private Dictionary<string, double> EvaluateForecasting(IDataView predictions, string labelColumn)
    {
        var metricsDict = new Dictionary<string, double>();

        // Forecasting evaluation: extract forecast vs actual from the transformed data
        // The SSA model outputs ForecastedValues as VBuffer<float>
        try
        {
            var forecastCol = predictions.Schema.GetColumnOrNull("ForecastedValues");
            var actualCol = predictions.Schema.GetColumnOrNull(labelColumn);

            if (forecastCol.HasValue && actualCol.HasValue)
            {
                // Get actual values and last forecast
                var actualValues = new List<float>();
                VBuffer<float> lastForecast = default;

                using var cursor = predictions.GetRowCursor(new[] { forecastCol.Value, actualCol.Value });
                var forecastGetter = cursor.GetGetter<VBuffer<float>>(forecastCol.Value);
                var actualGetter = cursor.GetGetter<float>(actualCol.Value);

                while (cursor.MoveNext())
                {
                    float val = 0;
                    actualGetter(ref val);
                    actualValues.Add(val);
                    forecastGetter(ref lastForecast);
                }

                var forecastedValues = lastForecast.DenseValues().ToArray();
                if (forecastedValues.Length > 0 && actualValues.Count > forecastedValues.Length)
                {
                    var horizon = forecastedValues.Length;
                    var holdoutStart = actualValues.Count - horizon;
                    var holdoutActual = actualValues.Skip(holdoutStart).Take(horizon).ToArray();

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
                    metricsDict["horizon"] = horizon;
                }
            }
        }
        catch
        {
            // Forecasting evaluation may fail with schema mismatch
        }

        return metricsDict;
    }

    private Dictionary<string, double> EvaluateRanking(IDataView predictions, string labelColumn, string? groupColumn)
    {
        var metricsDict = new Dictionary<string, double>();

        try
        {
            // GroupId is the mapped Key column created during training pipeline
            var groupIdCol = "GroupId";
            if (predictions.Schema.GetColumnOrNull(groupIdCol) == null && !string.IsNullOrEmpty(groupColumn))
                groupIdCol = groupColumn;

            var metrics = _mlContext.Ranking.Evaluate(predictions,
                labelColumnName: labelColumn,
                rowGroupColumnName: groupIdCol);

            var ndcg = metrics.NormalizedDiscountedCumulativeGains;
            var dcg = metrics.DiscountedCumulativeGains;

            for (int level = 0; level < ndcg.Count; level++)
                metricsDict[$"ndcg_at_{level + 1}"] = double.IsNaN(ndcg[level]) ? 0 : ndcg[level];
            for (int level = 0; level < dcg.Count; level++)
                metricsDict[$"dcg_at_{level + 1}"] = double.IsNaN(dcg[level]) ? 0 : dcg[level];

            var primaryNdcg = ndcg.Count > 0 ? ndcg[ndcg.Count - 1] : 0;
            metricsDict["ndcg"] = double.IsNaN(primaryNdcg) ? 0 : primaryNdcg;
        }
        catch
        {
            // Evaluation may fail if group column format is incompatible
        }

        return metricsDict;
    }

    private Dictionary<string, double> EvaluateClustering(IDataView predictions, string labelColumn)
    {
        var metricsDict = new Dictionary<string, double>();

        // ML.NET built-in clustering evaluation.
        // F-24: featureColumnName is REQUIRED for ML.NET to compute the Davies-Bouldin Index — without
        // it DBI comes back 0 (the evaluate-path twin of F-22, fixed identically in AutoMLRunner's
        // K-search). The clustering pipeline concatenates features into "Features", same as training.
        try
        {
            var metrics = _mlContext.Clustering.Evaluate(predictions, scoreColumnName: "Score", featureColumnName: "Features");
            metricsDict["average_distance"] = double.IsNaN(metrics.AverageDistance) ? 0 : metrics.AverageDistance;
            metricsDict["davies_bouldin_index"] = double.IsNaN(metrics.DaviesBouldinIndex) ? 0 : metrics.DaviesBouldinIndex;
            metricsDict["normalized_mutual_information"] = double.IsNaN(metrics.NormalizedMutualInformation) ? 0 : metrics.NormalizedMutualInformation;
        }
        catch
        {
            // Evaluation may fail if label column has incompatible format
        }

        // Cluster distribution from PredictedLabel
        var predictedLabelCol = predictions.Schema.GetColumnOrNull("PredictedLabel");
        if (predictedLabelCol.HasValue)
        {
            var clusterCounts = new Dictionary<uint, long>();

            using (var cursor = predictions.GetRowCursor(new[] { predictedLabelCol.Value }))
            {
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
                metricsDict["cluster_count"] = clusterCounts.Count;
                metricsDict["largest_cluster_ratio"] = totalCount > 0 ? (double)largestCluster / totalCount : 0;
            }
        }

        return metricsDict;
    }

    private Dictionary<string, double> EvaluateAnomalyDetection(IDataView predictions, string labelColumn)
    {
        var metricsDict = new Dictionary<string, double>();

        // Try ML.NET built-in anomaly evaluation
        try
        {
            var metrics = _mlContext.AnomalyDetection.Evaluate(predictions, labelColumnName: labelColumn);
            metricsDict["auc"] = double.IsNaN(metrics.AreaUnderRocCurve) ? 0 : metrics.AreaUnderRocCurve;
            metricsDict["detection_rate_at_fp5"] = metrics.DetectionRateAtFalsePositiveCount;
        }
        catch
        {
            // Label format may not match expected type - fall back to manual
        }

        // Manual counting from PredictedLabel
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

        return metricsDict;
    }
}
