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
        InputSchemaInfo? trainedSchema = null)
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
}
