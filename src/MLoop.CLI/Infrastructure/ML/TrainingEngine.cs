using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.AutoML;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Hooks;
using MLoop.Core.Models;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Complete training engine that orchestrates AutoML and experiment storage
/// </summary>
public class TrainingEngine : ITrainingEngine
{
    private readonly MLContext _mlContext;
    private readonly IDataProvider _dataLoader;
    private readonly IExperimentStore _experimentStore;
    private readonly IFileSystemManager _fileSystem;
    private readonly AutoMLRunner _autoMLRunner;
    private readonly HookEngine? _hookEngine;
    private readonly string? _projectRoot;

    public TrainingEngine(
        IFileSystemManager fileSystem,
        IExperimentStore experimentStore,
        string? projectRoot = null,
        ILogger? logger = null)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
        _experimentStore = experimentStore ?? throw new ArgumentNullException(nameof(experimentStore));
        _projectRoot = projectRoot;

        // Initialize ML.NET components
        _mlContext = new MLContext(seed: 42);
        _dataLoader = new CsvDataLoader(_mlContext);
        _autoMLRunner = new AutoMLRunner(_mlContext, _dataLoader);

        // Initialize HookEngine if project root provided
        if (!string.IsNullOrEmpty(projectRoot) && logger != null)
        {
            _hookEngine = new HookEngine(projectRoot, logger);
        }
    }

    public async Task<TrainingResult> TrainAsync(
        TrainingConfig config,
        IProgress<TrainingProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var modelName = config.ModelName;

        // Generate experiment ID for this model
        var experimentId = await _experimentStore.GenerateIdAsync(modelName, cancellationToken);
        var experimentPath = _experimentStore.GetExperimentPath(modelName, experimentId);

        // Track original data file and temp file for cleanup (encoding conversion)
        string originalDataFile = config.DataFile;
        string? tempEncodingFile = null;

        try
        {
            // Create experiment directory
            await _fileSystem.CreateDirectoryAsync(experimentPath, cancellationToken);

            // Handle encoding detection/conversion for non-UTF8 files (e.g., CP949/EUC-KR)
            // This ensures Korean and other non-ASCII text is read correctly throughout training
            var (convertedPath, detection) = EncodingDetector.ConvertToUtf8WithBom(config.DataFile);

            // Use converted path for all training operations, original path for metadata
            string dataFilePath = config.DataFile;
            if (detection.WasConverted)
            {
                tempEncodingFile = convertedPath;
                dataFilePath = convertedPath;
                Console.WriteLine($"[Info] Converted {detection.EncodingName} → UTF-8: {Path.GetFileName(originalDataFile)}");
            }

            // Validate data quality before training (label column + dataset size)
            var dataQualityValidator = new DataQualityValidator(_mlContext);
            var qualityResult = dataQualityValidator.ValidateTrainingData(dataFilePath, config.LabelColumn);

            if (!qualityResult.IsValid)
            {
                // Data quality issue detected - fail fast with clear error
                throw new InvalidOperationException(
                    $"{qualityResult.ErrorMessage}\n" +
                    $"{qualityResult.ErrorMessageEn ?? ""}\n" +
                    $"{string.Join("\n", qualityResult.Suggestions)}");
            }

            // Show warnings if any
            foreach (var warning in qualityResult.Warnings)
            {
                Console.WriteLine($"[Warning] {warning}");
            }

            // Capture input schema before training (using enhanced detection)
            var inputSchema = CaptureInputSchemaEnhanced(dataFilePath, config.LabelColumn);

            // Execute PreTrain hooks
            if (_hookEngine != null && _hookEngine.HasHooks(HookType.PreTrain))
            {
                // Load training data for hook context
                var loader = _mlContext.Data.CreateTextLoader(
                    new[] { new TextLoader.Column("Features", DataKind.Single, 0, int.MaxValue) },
                    separatorChar: ',',
                    hasHeader: true);
                var trainData = loader.Load(dataFilePath);

                var hookContext = new HookContext
                {
                    HookType = HookType.PreTrain,
                    HookName = string.Empty,  // Will be set by HookEngine
                    MLContext = _mlContext,
                    DataView = trainData,
                    Model = null,
                    ExperimentResult = null,
                    Metrics = null,
                    ProjectRoot = _projectRoot!,
                    Logger = new TrainingEngineLogger(),
                    Metadata = new Dictionary<string, object>
                    {
                        ["LabelColumn"] = config.LabelColumn,
                        ["TaskType"] = config.Task,
                        ["ModelName"] = modelName,
                        ["ExperimentId"] = experimentId,
                        ["TimeLimit"] = config.TimeLimitSeconds
                    }
                };

                var hooksPassed = await _hookEngine.ExecuteHooksAsync(HookType.PreTrain, hookContext);
                if (!hooksPassed)
                {
                    throw new InvalidOperationException("Training aborted by PreTrain hook");
                }
            }

            // Run AutoML
            var autoMLResult = await _autoMLRunner.RunAsync(config, progress, cancellationToken);

            stopwatch.Stop();

            // Save model
            var modelPath = _fileSystem.CombinePath(experimentPath, "model.zip");
            _mlContext.Model.Save(autoMLResult.Model, null, modelPath);

            // Execute PostTrain hooks
            if (_hookEngine != null && _hookEngine.HasHooks(HookType.PostTrain))
            {
                var hookContext = new HookContext
                {
                    HookType = HookType.PostTrain,
                    HookName = string.Empty,  // Will be set by HookEngine
                    MLContext = _mlContext,
                    DataView = null,
                    Model = autoMLResult.Model,
                    ExperimentResult = autoMLResult,
                    Metrics = autoMLResult.Metrics,
                    ProjectRoot = _projectRoot!,
                    Logger = new TrainingEngineLogger(),
                    Metadata = new Dictionary<string, object>
                    {
                        ["LabelColumn"] = config.LabelColumn,
                        ["TaskType"] = config.Task,
                        ["ModelName"] = modelName,
                        ["ExperimentId"] = experimentId,
                        ["TimeLimit"] = config.TimeLimitSeconds,
                        ["BestTrainer"] = autoMLResult.BestTrainer,
                        ["ModelPath"] = modelPath
                    }
                };

                var hooksPassed = await _hookEngine.ExecuteHooksAsync(HookType.PostTrain, hookContext);
                if (!hooksPassed)
                {
                    // PostTrain hooks can't abort (model already trained), just log warning
                    Console.WriteLine("[Warning] PostTrain hook requested abort, but model is already trained");
                }
            }

            // Prepare experiment data
            var experimentData = new ExperimentData
            {
                ModelName = modelName,
                ExperimentId = experimentId,
                Timestamp = DateTime.UtcNow,
                Status = "completed",
                Task = config.Task,
                Config = new ExperimentConfig
                {
                    DataFile = originalDataFile, // Store original path, not converted temp file
                    LabelColumn = config.LabelColumn,
                    TimeLimitSeconds = config.TimeLimitSeconds,
                    Metric = config.Metric,
                    TestSplit = config.TestSplit,
                    InputSchema = inputSchema
                },
                Result = new ExperimentResult
                {
                    BestTrainer = autoMLResult.BestTrainer,
                    TrainingTimeSeconds = stopwatch.Elapsed.TotalSeconds
                },
                Metrics = autoMLResult.Metrics
            };

            // Save experiment metadata
            await _experimentStore.SaveAsync(modelName, experimentData, cancellationToken);

            return new TrainingResult
            {
                ExperimentId = experimentId,
                BestTrainer = autoMLResult.BestTrainer,
                Metrics = autoMLResult.Metrics,
                TrainingTimeSeconds = stopwatch.Elapsed.TotalSeconds,
                ModelPath = modelPath,
                SchemaInfo = inputSchema != null
                    ? string.Join(",", inputSchema.Columns.Select(c => c.Name))
                    : null,
                RowCount = autoMLResult.RowCount
            };
        }
        catch (Exception ex)
        {
            stopwatch.Stop();

            // Save failed experiment
            var experimentData = new ExperimentData
            {
                ModelName = modelName,
                ExperimentId = experimentId,
                Timestamp = DateTime.UtcNow,
                Status = "failed",
                Task = config.Task,
                Config = new ExperimentConfig
                {
                    DataFile = originalDataFile, // Store original path, not converted temp file
                    LabelColumn = config.LabelColumn,
                    TimeLimitSeconds = config.TimeLimitSeconds,
                    Metric = config.Metric,
                    TestSplit = config.TestSplit
                },
                Result = new ExperimentResult
                {
                    BestTrainer = "none",
                    TrainingTimeSeconds = stopwatch.Elapsed.TotalSeconds
                }
            };

            await _experimentStore.SaveAsync(modelName, experimentData, cancellationToken);

            throw new InvalidOperationException(
                $"Training failed for experiment {modelName}/{experimentId}: {ex.Message}",
                ex);
        }
        finally
        {
            // Clean up temp file from encoding conversion
            // ML.NET may have lazily loaded data, but by now training is complete
            if (tempEncodingFile != null && File.Exists(tempEncodingFile))
            {
                try { File.Delete(tempEncodingFile); } catch { /* Ignore cleanup errors */ }
            }
        }
    }

    /// <summary>
    /// Captures input schema information from the data file
    /// </summary>
    private InputSchemaInfo? CaptureInputSchema(string dataFile, string labelColumn)
    {
        try
        {
            // Infer column information
            var columnInference = _mlContext.Auto().InferColumns(
                dataFile,
                labelColumnName: labelColumn,
                separatorChar: ',');

            if (columnInference == null || columnInference.ColumnInformation == null)
            {
                return null;
            }

            var columns = new List<ColumnSchema>();
            var columnInfo = columnInference.ColumnInformation;

            // Get all columns from the file with UTF-8 encoding
            string? firstLine;
            using (var reader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine();
            }

            if (string.IsNullOrEmpty(firstLine))
            {
                return null;
            }

            var columnNames = firstLine.Split(',');

            // Read all data lines to collect categorical values with UTF-8 encoding
            var allLines = File.ReadAllLines(dataFile, System.Text.Encoding.UTF8);
            var dataLines = allLines.Skip(1).ToArray(); // Skip header

            foreach (var colName in columnNames)
            {
                var purpose = GetColumnPurpose(colName, columnInfo);
                var dataType = GetColumnDataType(colName, columnInfo);

                // For categorical columns, collect all unique values
                List<string>? categoricalValues = null;
                int? uniqueCount = null;

                if (dataType == "Categorical" && purpose == "Feature")
                {
                    var colIndex = Array.IndexOf(columnNames, colName);
                    if (colIndex >= 0)
                    {
                        var uniqueValues = new HashSet<string>();

                        foreach (var line in dataLines)
                        {
                            var values = line.Split(',');
                            if (colIndex < values.Length)
                            {
                                var value = values[colIndex].Trim();
                                if (!string.IsNullOrEmpty(value))
                                {
                                    uniqueValues.Add(value);
                                }
                            }
                        }

                        categoricalValues = uniqueValues.OrderBy(v => v).ToList();
                        uniqueCount = uniqueValues.Count;
                    }
                }

                columns.Add(new ColumnSchema
                {
                    Name = colName,
                    DataType = dataType,
                    Purpose = purpose,
                    CategoricalValues = categoricalValues,
                    UniqueValueCount = uniqueCount
                });
            }

            return new InputSchemaInfo
            {
                Columns = columns,
                CapturedAt = DateTime.UtcNow
            };
        }
        catch
        {
            // If schema capture fails, return null (non-critical)
            return null;
        }
    }

    private string GetColumnPurpose(string columnName, Microsoft.ML.AutoML.ColumnInformation columnInfo)
    {
        if (columnInfo.LabelColumnName == columnName)
            return "Label";
        if (columnInfo.IgnoredColumnNames?.Contains(columnName) == true)
            return "Ignore";
        if (columnInfo.CategoricalColumnNames?.Contains(columnName) == true ||
            columnInfo.NumericColumnNames?.Contains(columnName) == true ||
            columnInfo.TextColumnNames?.Contains(columnName) == true)
            return "Feature";

        // Fallback: treat as Feature if not explicitly ignored or label
        return "Feature";
    }

    private string GetColumnDataType(string columnName, Microsoft.ML.AutoML.ColumnInformation columnInfo)
    {
        if (columnInfo.CategoricalColumnNames?.Contains(columnName) == true)
            return "Categorical";
        if (columnInfo.NumericColumnNames?.Contains(columnName) == true)
            return "Numeric";
        if (columnInfo.TextColumnNames?.Contains(columnName) == true)
            return "Text";

        // ML.NET InferColumns may not populate column lists correctly
        // Return Unknown and let CaptureInputSchema infer from actual data
        return "Unknown";
    }

    /// <summary>
    /// Captures input schema with enhanced type detection using actual data values
    /// </summary>
    private InputSchemaInfo? CaptureInputSchemaEnhanced(string dataFile, string labelColumn)
    {
        try
        {
            // Get all columns from the file with UTF-8 encoding
            string? firstLine;
            using (var reader = new StreamReader(dataFile, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                firstLine = reader.ReadLine();
            }

            if (string.IsNullOrEmpty(firstLine))
            {
                return null;
            }

            var columnNames = firstLine.Split(',');

            // Read sample of data lines for type inference
            var allLines = File.ReadAllLines(dataFile, System.Text.Encoding.UTF8);
            var dataLines = allLines.Skip(1).Take(1000).ToArray(); // Sample first 1000 rows

            // Try ML.NET InferColumns first
            Microsoft.ML.AutoML.ColumnInformation? columnInfo = null;
            try
            {
                var columnInference = _mlContext.Auto().InferColumns(
                    dataFile,
                    labelColumnName: labelColumn,
                    separatorChar: ',');
                columnInfo = columnInference?.ColumnInformation;
            }
            catch
            {
                // InferColumns may fail for complex data
            }

            var columns = new List<ColumnSchema>();

            foreach (var colName in columnNames)
            {
                var colIndex = Array.IndexOf(columnNames, colName);
                var purpose = GetColumnPurposeEnhanced(colName, labelColumn, columnInfo);
                var (dataType, categoricalValues, uniqueCount) = InferColumnTypeFromData(
                    colName, colIndex, dataLines, columnInfo);

                columns.Add(new ColumnSchema
                {
                    Name = colName,
                    DataType = dataType,
                    Purpose = purpose,
                    CategoricalValues = categoricalValues,
                    UniqueValueCount = uniqueCount
                });
            }

            return new InputSchemaInfo
            {
                Columns = columns,
                CapturedAt = DateTime.UtcNow
            };
        }
        catch
        {
            return null;
        }
    }

    private string GetColumnPurposeEnhanced(
        string columnName,
        string labelColumn,
        Microsoft.ML.AutoML.ColumnInformation? columnInfo)
    {
        // Exact match for label
        if (columnName == labelColumn)
            return "Label";

        // Check ML.NET inference
        if (columnInfo != null)
        {
            if (columnInfo.LabelColumnName == columnName)
                return "Label";
            if (columnInfo.IgnoredColumnNames?.Contains(columnName) == true)
                return "Ignore";
        }

        // Default: treat as Feature
        return "Feature";
    }

    private (string DataType, List<string>? CategoricalValues, int? UniqueCount) InferColumnTypeFromData(
        string columnName,
        int colIndex,
        string[] dataLines,
        Microsoft.ML.AutoML.ColumnInformation? columnInfo)
    {
        // First check ML.NET inference
        if (columnInfo != null)
        {
            if (columnInfo.CategoricalColumnNames?.Contains(columnName) == true)
            {
                var (values, count) = CollectCategoricalValues(colIndex, dataLines);
                return ("Categorical", values, count);
            }
            if (columnInfo.NumericColumnNames?.Contains(columnName) == true)
                return ("Numeric", null, null);
            if (columnInfo.TextColumnNames?.Contains(columnName) == true)
                return ("Text", null, null);
        }

        // Fallback: Infer type from actual values
        if (colIndex < 0 || dataLines.Length == 0)
            return ("Unknown", null, null);

        var uniqueValues = new HashSet<string>();
        int numericCount = 0;
        int totalCount = 0;

        foreach (var line in dataLines)
        {
            var values = line.Split(',');
            if (colIndex >= values.Length)
                continue;

            var value = values[colIndex].Trim();
            if (string.IsNullOrEmpty(value))
                continue;

            totalCount++;
            uniqueValues.Add(value);

            // Check if value is numeric
            if (double.TryParse(value, out _))
            {
                numericCount++;
            }
        }

        if (totalCount == 0)
            return ("Unknown", null, null);

        // Determine type based on data analysis
        double numericRatio = (double)numericCount / totalCount;
        int uniqueCount = uniqueValues.Count;

        // High numeric ratio (>90%) = Numeric column
        if (numericRatio > 0.9)
        {
            return ("Numeric", null, null);
        }

        // Low unique count relative to total = likely Categorical
        // Or explicit non-numeric values = Categorical
        if (uniqueCount <= 100 || (numericRatio < 0.5 && uniqueCount < totalCount * 0.1))
        {
            var categoricalValues = uniqueValues.OrderBy(v => v).ToList();
            return ("Categorical", categoricalValues, uniqueCount);
        }

        // High cardinality text
        return ("Text", null, uniqueCount);
    }

    private (List<string>? Values, int Count) CollectCategoricalValues(int colIndex, string[] dataLines)
    {
        var uniqueValues = new HashSet<string>();

        foreach (var line in dataLines)
        {
            var values = line.Split(',');
            if (colIndex < values.Length)
            {
                var value = values[colIndex].Trim();
                if (!string.IsNullOrEmpty(value))
                {
                    uniqueValues.Add(value);
                }
            }
        }

        return (uniqueValues.OrderBy(v => v).ToList(), uniqueValues.Count);
    }

    private class TrainingEngineLogger : ILogger
    {
        public void Debug(string message) { } // Silent during training
        public void Info(string message) => Console.WriteLine(message);
        public void Warning(string message) => Console.WriteLine($"[Warning] {message}");
        public void Error(string message) => Console.WriteLine($"[Error] {message}");
        public void Error(string message, Exception exception)
        {
            Console.WriteLine($"[Error] {message}");
            Console.WriteLine($"  {exception.Message}");
        }
    }
}
