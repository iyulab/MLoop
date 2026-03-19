using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.FileSystem;

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
        string? groupColumn = null)
    {
        return await Task.Run(() =>
        {
            string? tempFile = null;
            try
            {
                // Load the trained model
                var trainedModel = _mlContext.Model.Load(modelPath, out var modelSchema);

                // Load test data (no label renaming - model pipeline transforms label column)
                var testData = LoadTestData(testDataPath, labelColumn, taskType, trainedSchema, out tempFile);

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
                        $"This typically occurs when categorical columns in the test data contain values " +
                        $"not seen during training (OneHotEncoding creates different dimensions). " +
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
                else
                {
                    throw new NotSupportedException($"Task type '{taskType}' is not supported for evaluation.");
                }

                return metrics;
            }
            finally
            {
                // BUG-13: Clean up temp file AFTER all lazy data consumption is complete
                if (tempFile != null && File.Exists(tempFile))
                {
                    try { File.Delete(tempFile); } catch { }
                }
            }
        }, cancellationToken);
    }

    private IDataView LoadTestData(string testDataPath, string labelColumn, string taskType, InputSchemaInfo? trainedSchema, out string? tempFilePath)
    {
        // Ensure UTF-8 BOM for ML.NET compatibility
        string loadPath = testDataPath;
        tempFilePath = null;

        byte[] bom = new byte[3];
        using (var fs = File.OpenRead(testDataPath))
        {
            fs.ReadExactly(bom, 0, 3);
        }

        bool hasBom = bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF;
        if (!hasBom)
        {
            tempFilePath = Path.GetTempFileName();
            var allLines = File.ReadAllLines(testDataPath, System.Text.Encoding.UTF8);
            File.WriteAllLines(tempFilePath, allLines, new System.Text.UTF8Encoding(true));
            loadPath = tempFilePath;
        }

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

        // ML.NET built-in clustering evaluation
        try
        {
            var metrics = _mlContext.Clustering.Evaluate(predictions, scoreColumnName: "Score");
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
