using Microsoft.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.DeepLearning;

/// <summary>
/// The concrete <see cref="IDeepLearningModule"/> implementation. Register once at process
/// startup via <c>DeepLearningRegistry.Register(new DeepLearningModule())</c> so
/// <c>MLoop.Core.AutoMLRunner</c> and <c>MLoop.Core.Data.DataLoaderFactory</c> can dispatch
/// deep-learning tasks without MLoop.Core taking a compile-time dependency on TorchSharp/Vision.
/// </summary>
public sealed class DeepLearningModule : IDeepLearningModule
{
    private static readonly HashSet<string> DlTasks = new(StringComparer.OrdinalIgnoreCase)
    {
        "image-classification", "text-classification", "sentence-similarity",
        "ner", "object-detection", "question-answering"
    };

    public bool CanHandleTask(string task) => DlTasks.Contains(task);

    public Task<AutoMLResult> TrainAsync(MLContext mlContext, Action<string> log, string task,
        IDataView trainSet, IDataView testSet, TrainingConfig config,
        IProgress<TrainingProgress>? progress, CancellationToken ct) => task.ToLowerInvariant() switch
    {
        "image-classification" => DeepLearningHandlers.RunImageClassificationAsync(mlContext, log, trainSet, testSet, config, progress, ct),
        "text-classification"  => DeepLearningHandlers.RunTextClassificationAsync(mlContext, log, trainSet, testSet, config, progress, ct),
        "sentence-similarity"  => DeepLearningHandlers.RunSentenceSimilarityAsync(mlContext, log, trainSet, testSet, config, progress, ct),
        "ner"                  => DeepLearningHandlers.RunNerAsync(mlContext, log, trainSet, testSet, config, progress, ct),
        "object-detection"     => DeepLearningHandlers.RunObjectDetectionAsync(mlContext, log, trainSet, testSet, config, progress, ct),
        "question-answering"   => DeepLearningHandlers.RunQuestionAnsweringAsync(mlContext, log, trainSet, testSet, config, progress, ct),
        _ => throw new NotSupportedException($"Task '{task}' is not a deep-learning task")
    };

    public IDataProvider? CreateDataLoader(string task, MLContext mlContext, Action<string>? log) => task.ToLowerInvariant() switch
    {
        "image-classification" => new ImageDirectoryLoader(mlContext, log),
        "object-detection"     => new ObjectDetectionDataLoader(mlContext, log),
        _ => null
    };
}
