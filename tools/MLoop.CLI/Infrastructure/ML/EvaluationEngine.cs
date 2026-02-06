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
                var testData = LoadTestData(testDataPath, labelColumn, trainedSchema, out tempFile);

                // Make predictions on test data
                // Model pipeline transforms the label column (e.g. MapValueToKey for classification)
                var predictions = trainedModel.Transform(testData);

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

    private IDataView LoadTestData(string testDataPath, string labelColumn, InputSchemaInfo? trainedSchema, out string? tempFilePath)
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
        if (trainedSchema != null && columnInference.TextLoaderOptions.Columns != null)
        {
            var schemaLookup = trainedSchema.Columns.ToDictionary(c => c.Name, c => c.DataType);
            foreach (var col in columnInference.TextLoaderOptions.Columns)
            {
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

        // Load the data - keep original column names intact
        // NOTE: IDataView is lazy-loaded, so the file must remain available
        // until all data consumption is complete. Caller is responsible for
        // cleaning up tempFilePath after processing.
        return textLoader.Load(loadPath);
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
