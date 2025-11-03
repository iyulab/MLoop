using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

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
        CancellationToken cancellationToken = default)
    {
        return await Task.Run(() =>
        {
            // Load the trained model
            var trainedModel = _mlContext.Model.Load(modelPath, out var modelSchema);

            // Load test data
            var testData = LoadTestData(testDataPath, labelColumn);

            // Make predictions on test data
            var predictions = trainedModel.Transform(testData);

            // Evaluate based on task type
            Dictionary<string, double> metrics;

            if (taskType.Equals("regression", StringComparison.OrdinalIgnoreCase))
            {
                metrics = EvaluateRegression(predictions);
            }
            else if (taskType.Equals("classification", StringComparison.OrdinalIgnoreCase) ||
                     taskType.Equals("binary-classification", StringComparison.OrdinalIgnoreCase))
            {
                metrics = EvaluateBinaryClassification(predictions);
            }
            else if (taskType.Equals("multiclass-classification", StringComparison.OrdinalIgnoreCase))
            {
                metrics = EvaluateMulticlassClassification(predictions);
            }
            else
            {
                throw new NotSupportedException($"Task type '{taskType}' is not supported for evaluation.");
            }

            return metrics;
        }, cancellationToken);
    }

    private IDataView LoadTestData(string testDataPath, string labelColumn)
    {
        // Infer column types from the data
        var columnInference = _mlContext.Auto().InferColumns(
            testDataPath,
            labelColumnName: labelColumn,
            separatorChar: ',');

        // Create text loader with inferred schema
        var textLoader = _mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);

        // Load the data
        var dataView = textLoader.Load(testDataPath);

        // Ensure the label column is named "Label" for ML.NET evaluation
        // ML.NET evaluation expects a column named "Label"
        if (labelColumn != "Label")
        {
            // Copy the label column to a new column named "Label"
            dataView = _mlContext.Transforms.CopyColumns("Label", labelColumn)
                .Fit(dataView)
                .Transform(dataView);
        }

        return dataView;
    }

    private Dictionary<string, double> EvaluateRegression(IDataView predictions)
    {
        var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

        return new Dictionary<string, double>
        {
            { "r_squared", metrics.RSquared },
            { "rmse", metrics.RootMeanSquaredError },
            { "mae", metrics.MeanAbsoluteError },
            { "mse", metrics.MeanSquaredError }
        };
    }

    private Dictionary<string, double> EvaluateBinaryClassification(IDataView predictions)
    {
        var metrics = _mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

        return new Dictionary<string, double>
        {
            { "accuracy", metrics.Accuracy },
            { "auc", metrics.AreaUnderRocCurve },
            { "f1_score", metrics.F1Score },
            { "precision", metrics.PositivePrecision },
            { "recall", metrics.PositiveRecall }
        };
    }

    private Dictionary<string, double> EvaluateMulticlassClassification(IDataView predictions)
    {
        var metrics = _mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

        return new Dictionary<string, double>
        {
            { "accuracy", metrics.MacroAccuracy },
            { "micro_accuracy", metrics.MicroAccuracy },
            { "log_loss", metrics.LogLoss }
        };
    }
}
