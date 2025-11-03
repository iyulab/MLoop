using MLoop.Core.Models;

namespace MLoop.Core.Contracts;

/// <summary>
/// Training engine for ML models using AutoML
/// </summary>
public interface ITrainingEngine
{
    /// <summary>
    /// Trains a model using AutoML
    /// </summary>
    Task<TrainingResult> TrainAsync(
        TrainingConfig config,
        IProgress<TrainingProgress>? progress = null,
        CancellationToken cancellationToken = default);
}
