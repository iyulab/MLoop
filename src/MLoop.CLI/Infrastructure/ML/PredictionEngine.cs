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

        // Create UTF-8 BOM version of the file for ML.NET
        // ML.NET's InferColumns doesn't have encoding parameter and relies on BOM detection
        string mlnetCompatiblePath = processedDataPath;
        bool createdTempFile = false;

        try
        {
            // Check if file has UTF-8 BOM
            byte[] bom = new byte[3];
            using (var fs = File.OpenRead(processedDataPath))
            {
                fs.Read(bom, 0, 3);
            }

            bool hasBom = bom[0] == 0xEF && bom[1] == 0xBB && bom[2] == 0xBF;

            if (!hasBom)
            {
                // Create temp file with UTF-8 BOM for ML.NET compatibility
                var tempFile = Path.GetTempFileName();
                var allLines = File.ReadAllLines(processedDataPath, System.Text.Encoding.UTF8);
                File.WriteAllLines(tempFile, allLines, new System.Text.UTF8Encoding(true)); // true = add BOM
                mlnetCompatiblePath = tempFile;
                createdTempFile = true;
            }

            // Load the trained model
            DataViewSchema modelSchema;
            var trainedModel = _mlContext.Model.Load(modelPath, out modelSchema);

            // Load input data
            // Read the header to get column names with UTF-8 encoding
            string firstLine;
            using (var reader = new StreamReader(mlnetCompatiblePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine() ?? string.Empty;
            }

            if (string.IsNullOrEmpty(firstLine))
            {
                throw new InvalidOperationException("Input file is empty");
            }

            var columns = firstLine.Split(',');

            // Use column inference with a dummy label (it won't be used for prediction)
            // We need to provide a label column name to satisfy InferColumns
            var dummyLabel = columns.Length > 0 ? columns[0] : "dummy";

            var columnInference = _mlContext.Auto().InferColumns(
                mlnetCompatiblePath,
                labelColumnName: dummyLabel,
                separatorChar: ',');

            var textLoader = _mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var inputData = textLoader.Load(mlnetCompatiblePath);

            // DEBUG: Print input data columns
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine($"[DEBUG] Input DataView columns ({inputData.Schema.Count}):");
            foreach (var col in inputData.Schema)
            {
                var typeName = col.Type != null ? col.Type.ToString() : "null";
                Console.WriteLine($"  - '{col.Name}' ({typeName})");
            }
            Console.WriteLine($"[DEBUG] Model schema columns ({modelSchema?.Count ?? 0}):");
            if (modelSchema != null)
            {
                foreach (var col in modelSchema)
                {
                    var typeName = col.Type != null ? col.Type.ToString() : "null";
                    Console.WriteLine($"  - '{col.Name}' ({typeName})");
                }
            }

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
            // Clean up temporary files if created
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

            if (createdTempFile && File.Exists(mlnetCompatiblePath))
            {
                try
                {
                    File.Delete(mlnetCompatiblePath);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }
    }
}
