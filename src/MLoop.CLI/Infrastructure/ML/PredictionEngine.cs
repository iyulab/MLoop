using Microsoft.ML;
using Microsoft.ML.AutoML;
using MLoop.Core.Contracts;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// ML.NET-based prediction engine for making predictions with trained models
/// </summary>
public class PredictionEngine : IPredictionEngine
{
    private readonly MLContext _mlContext;
    private readonly CategoricalMapper _categoricalMapper;

    public PredictionEngine()
    {
        _mlContext = new MLContext(seed: 42);
        _categoricalMapper = new CategoricalMapper();
    }

    public async Task<int> PredictAsync(
        string modelPath,
        string inputDataPath,
        string outputPath,
        CancellationToken cancellationToken = default)
    {
        return await PredictAsync(modelPath, inputDataPath, outputPath, null, cancellationToken);
    }

    public async Task<int> PredictAsync(
        string modelPath,
        string inputDataPath,
        string outputPath,
        InputSchemaInfo? trainedSchema,
        CancellationToken cancellationToken = default)
    {
        return await PredictAsync(modelPath, inputDataPath, outputPath, trainedSchema,
            CategoricalMapper.UnknownValueStrategy.Auto, cancellationToken);
    }

    public async Task<int> PredictAsync(
        string modelPath,
        string inputDataPath,
        string outputPath,
        InputSchemaInfo? trainedSchema,
        CategoricalMapper.UnknownValueStrategy strategy,
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

        // Preprocess categorical values if schema is provided
        string processedDataPath = inputDataPath;
        bool useTempFile = false;

        if (trainedSchema != null)
        {
            var mappingResult = _categoricalMapper.PreprocessPredictionData(
                inputDataPath,
                trainedSchema,
                strategy);

            if (!mappingResult.Success)
            {
                throw new InvalidOperationException(
                    $"Categorical preprocessing failed:\n{mappingResult.ErrorMessage}");
            }

            // Log auto-selection info if strategy was auto-selected
            if (mappingResult.AppliedStrategy.HasValue)
            {
                Console.WriteLine($"[Auto] {mappingResult.StrategyReason}");
            }

            if (mappingResult.TempFilePath != null && mappingResult.TempFilePath != inputDataPath)
            {
                processedDataPath = mappingResult.TempFilePath;
                useTempFile = true;
            }
        }

        try
        {
            // Load the trained model
            DataViewSchema modelSchema;
            var trainedModel = _mlContext.Model.Load(modelPath, out modelSchema);

            // Load input data
            // Read the header to get column names
            var firstLine = File.ReadLines(processedDataPath).FirstOrDefault();
            if (string.IsNullOrEmpty(firstLine))
            {
                throw new InvalidOperationException("Input file is empty");
            }

            var columns = firstLine.Split(',');

            // Use column inference with a dummy label (it won't be used for prediction)
            // We need to provide a label column name to satisfy InferColumns
            var dummyLabel = columns.Length > 0 ? columns[0] : "dummy";

            var columnInference = _mlContext.Auto().InferColumns(
                processedDataPath,
                labelColumnName: dummyLabel,
                separatorChar: ',');

            var textLoader = _mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var inputData = textLoader.Load(processedDataPath);

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
        finally
        {
            // Clean up temporary file if created
            if (useTempFile && File.Exists(processedDataPath))
            {
                try
                {
                    File.Delete(processedDataPath);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }
    }
}
