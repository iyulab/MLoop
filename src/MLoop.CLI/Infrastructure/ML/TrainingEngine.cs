using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.AutoML;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;

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

    public TrainingEngine(
        IFileSystemManager fileSystem,
        IExperimentStore experimentStore)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
        _experimentStore = experimentStore ?? throw new ArgumentNullException(nameof(experimentStore));

        // Initialize ML.NET components
        _mlContext = new MLContext(seed: 42);
        _dataLoader = new CsvDataLoader(_mlContext);
        _autoMLRunner = new AutoMLRunner(_mlContext, _dataLoader);
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

        try
        {
            // Create experiment directory
            await _fileSystem.CreateDirectoryAsync(experimentPath, cancellationToken);

            // Validate data quality before training (label column + dataset size)
            var dataQualityValidator = new DataQualityValidator(_mlContext);
            var qualityResult = dataQualityValidator.ValidateTrainingData(config.DataFile, config.LabelColumn);

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

            // Capture input schema before training
            var inputSchema = CaptureInputSchema(config.DataFile, config.LabelColumn);

            // Run AutoML
            var autoMLResult = await _autoMLRunner.RunAsync(config, progress, cancellationToken);

            stopwatch.Stop();

            // Save model
            var modelPath = _fileSystem.CombinePath(experimentPath, "model.zip");
            _mlContext.Model.Save(autoMLResult.Model, null, modelPath);

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
                    DataFile = config.DataFile,
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
                ModelPath = modelPath
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
                    DataFile = config.DataFile,
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
        return "Unknown";
    }

    private string GetColumnDataType(string columnName, Microsoft.ML.AutoML.ColumnInformation columnInfo)
    {
        if (columnInfo.CategoricalColumnNames?.Contains(columnName) == true)
            return "Categorical";
        if (columnInfo.NumericColumnNames?.Contains(columnName) == true)
            return "Numeric";
        if (columnInfo.TextColumnNames?.Contains(columnName) == true)
            return "Text";
        return "Unknown";
    }
}
