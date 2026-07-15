using Microsoft.ML;
using MLoop.Core.Contracts;
using MLoop.Core.Models;

namespace MLoop.Core.AutoML;

/// <summary>
/// Optional deep-learning capability provided by MLoop.Core.DeepLearning. Registered once at
/// application startup via <see cref="DeepLearningRegistry"/>. Tabular-only consumers never
/// reference the DL package, so this stays null and DL task requests fail with an actionable error.
/// </summary>
public interface IDeepLearningModule
{
    bool CanHandleTask(string task);

    Task<AutoMLResult> TrainAsync(
        MLContext mlContext, Action<string> log, string task,
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken cancellationToken);

    /// <summary>DL directory/COCO loader for the task, or null if the task is not DL-backed.</summary>
    IDataProvider? CreateDataLoader(string task, MLContext mlContext, Action<string>? log);
}
