using Microsoft.ML;
using Microsoft.ML.AutoML;
using MLoop.Core.Contracts;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// ML.NET-based prediction engine for making predictions with trained models
/// </summary>
public class PredictionEngine : IPredictionEngine
{
    private readonly MLContext _mlContext;

    public PredictionEngine()
    {
        _mlContext = new MLContext(seed: 42);
    }

    public async Task<int> PredictAsync(
        string modelPath,
        string inputDataPath,
        string outputPath,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found: {modelPath}");
        }

        if (!File.Exists(inputDataPath))
        {
            throw new FileNotFoundException($"Input data file not found: {inputDataPath}");
        }

        // Load the trained model
        DataViewSchema modelSchema;
        var trainedModel = _mlContext.Model.Load(modelPath, out modelSchema);

        // Load input data
        // Read the header to get column names
        var firstLine = File.ReadLines(inputDataPath).FirstOrDefault();
        if (string.IsNullOrEmpty(firstLine))
        {
            throw new InvalidOperationException("Input file is empty");
        }

        var columns = firstLine.Split(',');

        // Use column inference with a dummy label (it won't be used for prediction)
        // We need to provide a label column name to satisfy InferColumns
        var dummyLabel = columns.Length > 0 ? columns[0] : "dummy";

        var columnInference = _mlContext.Auto().InferColumns(
            inputDataPath,
            labelColumnName: dummyLabel,
            separatorChar: ',');

        var textLoader = _mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        var inputData = textLoader.Load(inputDataPath);

        // Make predictions
        var predictions = trainedModel.Transform(inputData);

        // Count rows before saving
        long? count = predictions.GetRowCount();
        var rowCount = count.HasValue ? (int)count.Value : 0;

        // If GetRowCount returns null, count manually
        if (!count.HasValue)
        {
            using (var cursor = predictions.GetRowCursor(predictions.Schema))
            {
                while (cursor.MoveNext())
                {
                    rowCount++;
                }
            }
        }

        // Save predictions to CSV (without schema metadata for cleaner output)
        await using (var fileStream = File.Create(outputPath))
        {
            _mlContext.Data.SaveAsText(predictions, fileStream, separatorChar: ',', headerRow: true, schema: false);
        }

        return rowCount;
    }
}
