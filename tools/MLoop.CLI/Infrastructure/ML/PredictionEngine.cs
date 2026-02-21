using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Contracts;
using MLoop.Core.Data;

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
        CancellationToken cancellationToken = default,
        string? labelColumnOverride = null)
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
                fs.ReadExactly(bom, 0, 3);
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

            // Find the label column: prefer override, then trained schema
            string? labelColumn = labelColumnOverride;
            if (labelColumn == null && trainedSchema != null)
            {
                var labelInfo = trainedSchema.Columns.FirstOrDefault(c => c.Purpose == "Label");
                if (labelInfo != null)
                {
                    labelColumn = labelInfo.Name;
                }
            }

            // Apply same preprocessing as CsvDataLoader.LoadData to ensure schema consistency
            // between training and prediction data. Must run BEFORE reading headers
            // because these steps may remove columns (DateTime, sparse, index).
            mlnetCompatiblePath = CsvDataLoader.RemoveIndexColumns(mlnetCompatiblePath);
            mlnetCompatiblePath = CsvDataLoader.RemoveDateTimeColumns(mlnetCompatiblePath, labelColumn);
            mlnetCompatiblePath = CsvDataLoader.RemoveSparseColumns(mlnetCompatiblePath, labelColumn);

            // Read the header to get column names (after preprocessing)
            string firstLine;
            using (var reader = new StreamReader(mlnetCompatiblePath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine() ?? string.Empty;
            }

            if (string.IsNullOrEmpty(firstLine))
            {
                throw new InvalidOperationException("Input file is empty");
            }

            var columns = CsvFieldParser.ParseFields(firstLine);

            // If label column is expected but missing from prediction data,
            // add a dummy label column so the ML.NET model schema is satisfied
            var labelExistsInData = labelColumn != null && columns.Contains(labelColumn);

            if (!labelExistsInData && labelColumn != null)
            {
                var tempWithLabel = Path.GetTempFileName();
                var allLines = File.ReadAllLines(mlnetCompatiblePath, System.Text.Encoding.UTF8);
                using (var writer = new StreamWriter(tempWithLabel, false, new System.Text.UTF8Encoding(true)))
                {
                    // Add label column to header
                    writer.WriteLine(labelColumn + "," + allLines[0]);
                    // Add empty label value for each data row
                    for (int i = 1; i < allLines.Length; i++)
                    {
                        writer.WriteLine("," + allLines[i]);
                    }
                }

                // Update paths
                if (createdTempFile)
                {
                    File.Delete(mlnetCompatiblePath);
                }
                mlnetCompatiblePath = tempWithLabel;
                createdTempFile = true;

                // Update state: label now exists in the modified data
                firstLine = labelColumn + "," + firstLine;
                columns = CsvFieldParser.ParseFields(firstLine);
                labelExistsInData = true;
            }

            var dummyLabel = labelExistsInData ? labelColumn! : (columns.Length > 0 ? columns[0] : "dummy");

            var columnInference = _mlContext.Auto().InferColumns(
                mlnetCompatiblePath,
                labelColumnName: dummyLabel,
                separatorChar: ',');

            // If we have the label column in the data, ensure it's marked as ignored/label in inference
            if (labelExistsInData && !string.IsNullOrEmpty(labelColumn) && columnInference.ColumnInformation != null)
            {
                // Remove from feature columns if present
                columnInference.ColumnInformation.NumericColumnNames.Remove(labelColumn);
                columnInference.ColumnInformation.CategoricalColumnNames.Remove(labelColumn);
                columnInference.ColumnInformation.TextColumnNames.Remove(labelColumn);
            }

            // BUG-11: Fix column type mismatches between InferColumns and trained model.
            // InferColumns may misdetect types (e.g. "0" as Boolean instead of Single)
            // when prediction data has limited/dummy values. Override with trained schema types.
            if (trainedSchema != null && columnInference.TextLoaderOptions.Columns != null)
            {
                var schemaLookup = trainedSchema.Columns.ToDictionary(c => c.Name, c => c.DataType);
                foreach (var col in columnInference.TextLoaderOptions.Columns)
                {
                    if (col.Name != null && schemaLookup.TryGetValue(col.Name, out var expectedType))
                    {
                        var expectedKind = expectedType switch
                        {
                            "Numeric" => Microsoft.ML.Data.DataKind.Single,
                            "Categorical" => Microsoft.ML.Data.DataKind.String,
                            "Text" => Microsoft.ML.Data.DataKind.String,
                            "Boolean" => Microsoft.ML.Data.DataKind.Boolean,
                            _ => col.DataKind // keep inferred type for unknown schema types
                        };
                        if (col.DataKind != expectedKind)
                        {
                            col.DataKind = expectedKind;
                        }
                    }
                }
            }

            // BUG-15: If label column was inferred as Boolean, convert to String
            // (same as CsvDataLoader BUG-15 fix) for MapValueToKey compatibility
            if (labelColumn != null && columnInference.TextLoaderOptions.Columns != null)
            {
                foreach (var col in columnInference.TextLoaderOptions.Columns)
                {
                    if (col.Name != null &&
                        col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase) &&
                        col.DataKind == DataKind.Boolean)
                    {
                        col.DataKind = DataKind.String;
                    }
                }
            }

            // BUG-16: Enable RFC 4180 quoting for CSV fields containing commas
            // (e.g. bbox: "[935.49, 26.14, 123.27, 138.12]", attributes: "{'key': val, ...}")
            // CsvDataLoader already sets this but PredictionEngine was missing it.
            columnInference.TextLoaderOptions.AllowQuoting = true;

            var textLoader = _mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var inputData = textLoader.Load(mlnetCompatiblePath);

            // Keep all columns including label for model.Transform
            // Classification models (e.g. multiclass) include MapValueToKey on the label column
            // in the pipeline, so the label column must be present in the input schema.
            // The label values are ignored during prediction.
            IDataView processedData = inputData;

            // Make predictions
            var predictions = trainedModel.Transform(processedData);

            // IMP-R2-07: Restore original class names for PredictedLabel
            // ML.NET classification models use MapValueToKey on the label column during training,
            // which outputs numeric Key values (0, 1, 2, ...) in PredictedLabel.
            // MapKeyToValue reverses this mapping to show original class names.
            var predictedLabelCol = predictions.Schema.GetColumnOrNull("PredictedLabel");
            if (predictedLabelCol.HasValue && predictedLabelCol.Value.Type is KeyDataViewType)
            {
                var keyToValue = _mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel");
                predictions = keyToValue.Fit(predictions).Transform(predictions);
            }

            // Select only prediction output columns (exclude duplicated input features)
            // ML.NET prediction output columns: PredictedLabel, Score, Probability (for classification)
            // For regression: Score only
            var predictionColumns = new List<string>();
            foreach (var col in predictions.Schema)
            {
                // Include standard prediction output columns
                if (col.Name == "PredictedLabel" || col.Name == "Score" || col.Name == "Probability")
                {
                    predictionColumns.Add(col.Name);
                }
            }

            // If no prediction columns found, fall back to all columns (shouldn't happen)
            IDataView outputData;
            if (predictionColumns.Count > 0)
            {
                outputData = _mlContext.Transforms.SelectColumns(predictionColumns.ToArray())
                    .Fit(predictions)
                    .Transform(predictions);
            }
            else
            {
                outputData = predictions;
            }

            // Count rows before saving
            long? count = outputData.GetRowCount();
            var rowCount = count.HasValue ? (int)count.Value : 0;

            // If GetRowCount returns null, count manually
            if (!count.HasValue)
            {
                using (var cursor = outputData.GetRowCursor(outputData.Schema))
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
                _mlContext.Data.SaveAsText(outputData, fileStream, separatorChar: ',', headerRow: true, schema: false);
            }

            // BUG-14: Fix empty headers for vector columns (e.g., multiclass Score)
            // SaveAsText outputs empty column names for VBuffer columns
            FixVectorColumnHeaders(outputPath, outputData.Schema);

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

    /// <summary>
    /// Fixes empty column headers generated by SaveAsText for VBuffer (vector) columns.
    /// ML.NET's SaveAsText outputs empty header cells for each element of a vector column.
    /// This method replaces them with proper names like Score.0, Score.1, etc.
    /// </summary>
    private static void FixVectorColumnHeaders(string filePath, DataViewSchema schema)
    {
        // Build expected header names from schema
        var expectedHeaders = new List<string>();
        bool hasVectorColumns = false;

        foreach (var col in schema)
        {
            if (col.IsHidden) continue; // Skip hidden columns (e.g., Key-type PredictedLabel)

            if (col.Type is VectorDataViewType vectorType)
            {
                hasVectorColumns = true;
                for (int i = 0; i < vectorType.Size; i++)
                {
                    expectedHeaders.Add($"{col.Name}.{i}");
                }
            }
            else
            {
                expectedHeaders.Add(col.Name);
            }
        }

        if (!hasVectorColumns) return;

        // Read file, fix header, rewrite
        var lines = File.ReadAllLines(filePath, System.Text.Encoding.UTF8);
        if (lines.Length > 0)
        {
            lines[0] = string.Join(",", expectedHeaders);
            File.WriteAllLines(filePath, lines, new System.Text.UTF8Encoding(true));
        }
    }
}
