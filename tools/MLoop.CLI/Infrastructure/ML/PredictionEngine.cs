using System.Globalization;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Contracts;
using MLoop.Core.Models;
using MLoop.Core.Data;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;

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
        string? labelColumnOverride = null,
        IEnumerable<string>? preserveColumns = null,
        RegressionInterval? interval = null,
        string? taskType = null)
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

        // mlnetCompatiblePath holds the file fed to ML.NET after preprocessing; tempFiles collects
        // every temp the shared preprocessor (and the dummy-label step below) creates, cleaned up in
        // finally once the lazy IDataView consumption is complete.
        string mlnetCompatiblePath = processedDataPath;
        List<string> tempFiles = new();

        try
        {
            // Load the trained model
            DataViewSchema modelSchema;
            var trainedModel = _mlContext.Model.Load(modelPath, out modelSchema);

            // Find the label column: prefer override, then trained schema
            // Treat empty string as null (unsupervised tasks like anomaly-detection have no label)
            string? labelColumn = string.IsNullOrEmpty(labelColumnOverride) ? null : labelColumnOverride;
            if (labelColumn == null && trainedSchema != null)
            {
                var labelInfo = trainedSchema.Columns.FirstOrDefault(c => c.Purpose == "Label");
                if (labelInfo != null && !string.IsNullOrEmpty(labelInfo.Name))
                {
                    labelColumn = labelInfo.Name;
                }
            }

            // Apply the single shared inference preprocessing sequence (encoding → flatten → index →
            // schema-based exclude / data-dependent fallback) — identical to evaluate. This replaces
            // predict's hand-rolled UTF-8 BOM check (which read CP949 as UTF-8, BUG-43) and the
            // per-engine index/exclude reimplementation, and adds the previously-missing flatten and
            // constant-column steps that the divergence had silently dropped.
            mlnetCompatiblePath = InferenceDataPreprocessor.Prepare(processedDataPath, labelColumn, trainedSchema, out tempFiles);

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

                // The previous path (if a preprocessor temp) is already tracked in tempFiles and gets
                // cleaned up in finally; the original input is never tracked, so it stays untouched.
                mlnetCompatiblePath = tempWithLabel;
                tempFiles.Add(tempWithLabel);

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

            // EVAL-1: shared non-label reconciliation — overrides feature types from the trained
            // schema (BUG-11, incl. the "String" raw type name BUG-42), enables RFC 4180 quoting
            // (BUG-16), and splits preserved group/user/item columns out of any merged Features range
            // (F-23). The label is reconciled separately below because predict and evaluate handle it
            // differently — a legitimate divergence the helper deliberately leaves alone.
            CsvDataLoader.ReconcileInferredSchemaForInference(columnInference, trainedSchema, labelColumn, mlnetCompatiblePath, preserveColumns);

            // BUG-11 (label): predict aligns the label column's type to the trained schema too. The
            // label value is ignored at predict time, but its type must satisfy the model's input
            // schema (e.g. MapValueToKey). evaluate deliberately skips this (BUG-18), so the shared
            // helper leaves the label untouched; predict applies it here via the same type table.
            if (trainedSchema != null && labelColumn != null && columnInference.TextLoaderOptions.Columns != null)
            {
                var labelType = trainedSchema.Columns
                    .FirstOrDefault(c => c.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase))?.DataType;
                if (labelType != null)
                {
                    foreach (var col in columnInference.TextLoaderOptions.Columns)
                    {
                        if (col.Name != null && col.Name.Equals(labelColumn, StringComparison.OrdinalIgnoreCase))
                            col.DataKind = CsvDataLoader.MapTrainedTypeToDataKind(labelType, col.DataKind);
                    }
                }
            }

            // BUG-15: If the label column is (still) Boolean, convert to String for MapValueToKey
            // compatibility — same as CsvDataLoader's BUG-15 training fix.
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

            var textLoader = _mlContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var inputData = textLoader.Load(mlnetCompatiblePath);

            // Keep all columns including label for model.Transform
            // Classification models (e.g. multiclass) include MapValueToKey on the label column
            // in the pipeline, so the label column must be present in the input schema.
            // The label values are ignored during prediction.
            IDataView processedData = inputData;

            // D24: clustering's saved model now expects a single "Features" vector built from every
            // feature column (train-side fix, AutoMLRunner.RunClusteringAsync) — including the CSV's
            // first column, which InferColumns always treats as *some* label (there being no real one
            // for label-less clustering) and therefore excludes from its own "Features" merge above.
            // Left alone, this predict path's "Features" would carry one fewer dimension than the
            // model expects. Re-concatenate the placeholder label back in — but only when it truly is
            // a placeholder (labelColumn is null, i.e. no schema column actually carries Purpose=Label);
            // a real declared label must stay excluded, matching the train-time featurizer.
            if (string.Equals(taskType, "clustering", StringComparison.OrdinalIgnoreCase)
                && labelColumn is null
                && processedData.Schema.GetColumnOrNull("Features") is not null)
            {
                processedData = _mlContext.Transforms.Concatenate("Features", dummyLabel, "Features")
                    .Fit(processedData)
                    .Transform(processedData);
            }

            // Make predictions
            IDataView predictions;
            try
            {
                predictions = trainedModel.Transform(processedData);
            }
            catch (Exception ex) when (ex.Message.Contains("Schema mismatch") ||
                                       ex.Message.Contains("Vector<Single"))
            {
                throw new InvalidOperationException(
                    $"Feature vector dimension mismatch during prediction. " +
                    $"The prediction data's columns don't match the schema the model was trained on " +
                    $"(a feature column may be missing, renamed, or have a different type). The saved model " +
                    $"embeds its fitted featurizers, so this is a column-structure mismatch, not a text/value " +
                    $"distribution issue. Ensure the prediction CSV has the same columns (names and types) as " +
                    $"the training data. Original error: {ex.Message}", ex);
            }

            // IMP-R2-07: Restore original class names for PredictedLabel
            // ML.NET classification models use MapValueToKey on the label column during training,
            // which outputs numeric Key values (0, 1, 2, ...) in PredictedLabel.
            // MapKeyToValue reverses this mapping to show original class names.
            // Only classification keys carry a KeyValues mapping back to the original label strings.
            // Clustering's PredictedLabel is also a key (the cluster id) but has no KeyValues, so
            // MapKeyToValue would throw "Metadata KeyValues does not exist" — which broke
            // `mloop predict` for clustering. Guard via the shared PredictionService.HasKeyValues.
            var predictedLabelCol = predictions.Schema.GetColumnOrNull("PredictedLabel");
            if (predictedLabelCol.HasValue && predictedLabelCol.Value.Type is KeyDataViewType
                && MLoop.Core.Prediction.PredictionService.HasKeyValues(predictedLabelCol.Value))
            {
                var keyToValue = _mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel");
                predictions = keyToValue.Fit(predictions).Transform(predictions);
            }

            // ② regression wave: mirror serve /predict — when the model carries a conformal band, wrap
            // the regression Score into [ScoreLowerBound, ScoreUpperBound] so the CSV output matches the
            // serve JSON (the two predict paths must not drift — the D-series lesson). Heteroscedastic
            // models widen the band per row via the sibling σ-model; others use the constant half-width.
            if (interval != null && predictions.Schema.GetColumnOrNull("Score") != null)
            {
                predictions = ApplyConformalBand(predictions, interval, modelPath);
            }

            // Select only prediction output columns (exclude duplicated input features)
            // ML.NET prediction output columns: PredictedLabel, Score, Probability (for classification)
            // For regression: Score only
            // Note: After MapKeyToValue("PredictedLabel"), the schema contains both the original
            // hidden key-type PredictedLabel and the new visible string PredictedLabel.
            // Must skip hidden columns to avoid duplicate names in SelectColumns.
            var predictionColumns = new List<string>();
            foreach (var col in predictions.Schema)
            {
                if (col.IsHidden) continue; // Skip hidden columns (e.g., key-type PredictedLabel after MapKeyToValue)
                // Include standard prediction output columns
                if (col.Name == "PredictedLabel" || col.Name == "Score" || col.Name == "Probability"
                    || col.Name == "ScoreLowerBound" || col.Name == "ScoreUpperBound")
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
            // Categorical preprocessing temp (created before the shared preprocessor ran).
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

            // Every temp the shared preprocessor and the dummy-label step created.
            foreach (var tempFile in tempFiles)
            {
                if (File.Exists(tempFile))
                {
                    try { File.Delete(tempFile); } catch { }
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

    /// <summary>
    /// Prediction using the shared PredictionService (Dict[] → PredictionResult path). The single source
    /// for structured predictions — the API and the CLI <c>--json</c> path both go through here so their
    /// output (rows, conformal band, normalized <c>confidence</c>) never drifts. When <paramref name="interval"/>
    /// carries a heteroscedastic band, the service loads the sibling <c>residual-model.zip</c> itself.
    /// </summary>
    public PredictionResult PredictWithService(
        Dictionary<string, object>[] rows,
        InputSchemaInfo schema,
        string modelPath,
        string taskType,
        string? labelColumn = null,
        RegressionInterval? interval = null)
    {
        var service = new PredictionService(_mlContext);
        return service.Predict(rows, schema, modelPath, taskType, labelColumn, interval);
    }

    /// <summary>
    /// Converts a CSV file to Dictionary rows for PredictionService consumption.
    /// Applies CLI-specific file preprocessing (encoding detection, index column removal,
    /// schema-based excluded column removal).
    /// </summary>
    /// <param name="csvPath">Path to input CSV file</param>
    /// <param name="trainedSchema">Optional trained schema for column exclusion</param>
    /// <param name="labelColumn">Optional label column name to exclude from rows</param>
    /// <returns>Array of dictionaries, one per row, with header names as keys</returns>
    public static Dictionary<string, object>[] LoadCsvAsRows(
        string csvPath,
        InputSchemaInfo? trainedSchema = null,
        string? labelColumn = null)
    {
        if (!File.Exists(csvPath))
        {
            throw new FileNotFoundException($"CSV file not found: {csvPath}");
        }

        // Read all lines with encoding detection (handles CP949/EUC-KR)
        var allLines = File.ReadAllLines(csvPath, System.Text.Encoding.UTF8);
        if (allLines.Length < 2)
        {
            return Array.Empty<Dictionary<string, object>>();
        }

        var headers = CsvFieldParser.ParseFields(allLines[0]);

        // Determine which columns to exclude
        var excludedIndices = new HashSet<int>();

        if (trainedSchema != null)
        {
            var excludedNames = new HashSet<string>(
                trainedSchema.Columns
                    .Where(c => c.Purpose == "Exclude")
                    .Select(c => c.Name),
                StringComparer.OrdinalIgnoreCase);

            for (int i = 0; i < headers.Length; i++)
            {
                if (excludedNames.Contains(headers[i].Trim()))
                {
                    excludedIndices.Add(i);
                }
            }
        }

        // Exclude label column from feature rows
        if (labelColumn != null)
        {
            for (int i = 0; i < headers.Length; i++)
            {
                if (headers[i].Trim().Equals(labelColumn, StringComparison.OrdinalIgnoreCase))
                {
                    excludedIndices.Add(i);
                    break;
                }
            }
        }

        // Build active header list (non-excluded)
        var activeHeaders = new List<(int Index, string Name)>();
        for (int i = 0; i < headers.Length; i++)
        {
            if (!excludedIndices.Contains(i))
            {
                activeHeaders.Add((i, headers[i].Trim()));
            }
        }

        // Parse data rows
        var rows = new List<Dictionary<string, object>>();
        for (int lineIdx = 1; lineIdx < allLines.Length; lineIdx++)
        {
            var line = allLines[lineIdx];
            if (string.IsNullOrWhiteSpace(line)) continue;

            var fields = CsvFieldParser.ParseFields(line);
            var row = new Dictionary<string, object>(activeHeaders.Count);

            foreach (var (colIdx, name) in activeHeaders)
            {
                if (colIdx >= fields.Length)
                {
                    row[name] = "";
                    continue;
                }

                var value = fields[colIdx];

                // Try numeric parsing: float with InvariantCulture, fallback to string
                if (float.TryParse(value, NumberStyles.Float | NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var numericValue))
                {
                    row[name] = numericValue;
                }
                else
                {
                    row[name] = value;
                }
            }

            rows.Add(row);
        }

        return rows.ToArray();
    }

    /// <summary>
    /// ② regression wave: adds [ScoreLowerBound, ScoreUpperBound] columns to a scored regression view.
    /// Heteroscedastic models (interval carries q + β and a sibling residual-model.zip exists) get a
    /// per-row width q·(max(σ(x),0)+β): the point Score is preserved as PointScore, the σ-model overwrites
    /// Score with its residual estimate, the band is computed from both, then Score is restored to the
    /// point prediction. Any failure (no σ-model, missing Features) falls back to the constant half-width
    /// so the CSV always carries the band — matching serve's behaviour (the two paths must not drift).
    /// </summary>
    private IDataView ApplyConformalBand(IDataView predictions, RegressionInterval interval, string modelPath)
    {
        var residualPath = Path.Combine(Path.GetDirectoryName(modelPath) ?? ".", ExperimentLayout.ResidualModelFileName);
        if (interval.IsHeteroscedastic && File.Exists(residualPath))
        {
            try
            {
                var aux = _mlContext.Model.Load(residualPath, out _);

                var withPoint = _mlContext.Transforms.CopyColumns("PointScore", "Score").Fit(predictions).Transform(predictions);
                var withSigma = aux.Transform(withPoint); // aux "Score" = σ(x), shadows the point Score (preserved as PointScore)
                if (withSigma.Schema.GetColumnOrNull("Score") is not null)
                {
                    var band = _mlContext.Transforms.CustomMapping<PointSigma, ScoreBand>(
                        (input, output) =>
                        {
                            // Single source for the per-row half-width: q·(max(σ,0)+β) lives only in
                            // RegressionInterval.WidthFor, shared with PredictionService's regression path
                            // so the CSV and the serve JSON can't drift on the band (PRED-1). input.Score
                            // here is the σ-model's raw residual estimate.
                            double half = interval.WidthFor(input.Score);
                            output.ScoreLowerBound = (float)(input.PointScore - half);
                            output.ScoreUpperBound = (float)(input.PointScore + half);
                        },
                        contractName: null).Fit(withSigma).Transform(withSigma);
                    // Restore Score to the point prediction (aux had shadowed it with σ).
                    return _mlContext.Transforms.CopyColumns("Score", "PointScore").Fit(band).Transform(band);
                }
            }
            catch
            {
                // fall through to constant-width band below
            }
        }

        float halfWidth = (float)interval.HalfWidth;
        var bandTransform = _mlContext.Transforms.CustomMapping<ScoreOnly, ScoreBand>(
            (input, output) =>
            {
                output.ScoreLowerBound = input.Score - halfWidth;
                output.ScoreUpperBound = input.Score + halfWidth;
            },
            contractName: null);
        return bandTransform.Fit(predictions).Transform(predictions);
    }

    // ② regression wave: CustomMapping shapes for wrapping a regression Score in its conformal band.
    private sealed class ScoreOnly
    {
        public float Score { get; set; }
    }

    private sealed class ScoreBand
    {
        public float ScoreLowerBound { get; set; }
        public float ScoreUpperBound { get; set; }
    }

    // Heteroscedastic band input: the preserved point prediction plus the σ-model's residual estimate
    // (which the σ-model emits into "Score").
    private sealed class PointSigma
    {
        public float PointScore { get; set; }
        public float Score { get; set; }
    }
}
