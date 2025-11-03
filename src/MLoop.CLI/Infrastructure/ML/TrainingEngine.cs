using System.Diagnostics;
using Microsoft.ML;
using MLoop.Core.Contracts;
using MLoop.Core.Models;
using MLoop.Core.Data;
using MLoop.Core.AutoML;
using MLoop.CLI.Infrastructure.FileSystem;

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

        // Generate experiment ID
        var experimentId = await _experimentStore.GenerateIdAsync(cancellationToken);
        var experimentPath = _experimentStore.GetExperimentPath(experimentId);

        try
        {
            // Create experiment directory
            await _fileSystem.CreateDirectoryAsync(experimentPath, cancellationToken);

            // Run AutoML
            var autoMLResult = await _autoMLRunner.RunAsync(config, progress, cancellationToken);

            stopwatch.Stop();

            // Save model
            var modelPath = _fileSystem.CombinePath(experimentPath, "model.zip");
            _mlContext.Model.Save(autoMLResult.Model, null, modelPath);

            // Prepare experiment data
            var experimentData = new ExperimentData
            {
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
                    TestSplit = config.TestSplit
                },
                Result = new ExperimentResult
                {
                    BestTrainer = autoMLResult.BestTrainer,
                    TrainingTimeSeconds = stopwatch.Elapsed.TotalSeconds
                },
                Metrics = autoMLResult.Metrics
            };

            // Save experiment metadata
            await _experimentStore.SaveAsync(experimentData, cancellationToken);

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

            await _experimentStore.SaveAsync(experimentData, cancellationToken);

            throw new InvalidOperationException(
                $"Training failed for experiment {experimentId}: {ex.Message}",
                ex);
        }
    }
}
