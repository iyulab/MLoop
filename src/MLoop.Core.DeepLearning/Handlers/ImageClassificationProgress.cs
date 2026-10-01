using System.Diagnostics;
using Microsoft.ML.Vision;
using MLoop.Core.Models;

namespace MLoop.Core.DeepLearning;

/// <summary>
/// Turns the TensorFlow image-classification trainer's <see cref="ImageClassificationTrainer.Options.MetricsCallback"/>
/// into training progress: one <see cref="TrainingPhase.Featurize"/> event per image run through the
/// pretrained network, then one <see cref="TrainingPhase.Epoch"/> event per epoch.
/// </summary>
/// <remarks>
/// <see cref="EpochProgress"/> covers the TorchSharp trainers by reading their log; this trainer
/// does not log its progress, it calls back. Without this the bar stood at 0% for the whole fit —
/// ten minutes on 970 images, almost all of it the featurization pass that precedes any epoch.
/// The trainer stops early once accuracy stops improving, so the epoch count it is given is a cap,
/// and the events say so.
/// </remarks>
public sealed class ImageClassificationProgress
{
    /// <summary>The epoch cap the trainer is given — the library default, stated where it is used.</summary>
    public const int MaxEpochs = 200;

    private readonly IProgress<TrainingProgress>? _progress;
    private readonly string _trainerName;
    private readonly int _images;
    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private int _featurized;
    private bool _training;

    /// <param name="progress">Where the events go; <c>null</c> reports nothing.</param>
    /// <param name="trainerName">The trainer the events name.</param>
    /// <param name="images">Images the fit featurizes (the whole training set, including the
    /// share the trainer holds out to validate each epoch).</param>
    public ImageClassificationProgress(IProgress<TrainingProgress>? progress, string trainerName, int images)
    {
        _progress = progress;
        _trainerName = trainerName;
        _images = images;
    }

    /// <summary>Pass as <see cref="ImageClassificationTrainer.Options.MetricsCallback"/>.</summary>
    public void OnMetrics(ImageClassificationTrainer.ImageClassificationMetrics metrics)
    {
        if (metrics.Bottleneck is not null)
        {
            var done = Math.Min(++_featurized, _images);
            Report(TrainingPhase.Featurize, step: done, steps: _images);
        }
        else if (metrics.Train is { DatasetUsed: ImageClassificationTrainer.ImageClassificationMetrics.Dataset.Train } train)
        {
            if (!_training)
            {
                // Epochs are timed on their own; the featurization pass is not part of their pace.
                _training = true;
                _clock.Restart();
            }
            // The trainer numbers epochs from zero; what the user wants is how many are done.
            Report(TrainingPhase.Epoch, epoch: Math.Min(train.Epoch + 1, MaxEpochs));
        }
    }

    private void Report(TrainingPhase phase, int epoch = 0, int step = 0, int steps = 0) =>
        _progress?.Report(new TrainingProgress
        {
            Phase = phase,
            Epoch = epoch,
            MaxEpochs = phase == TrainingPhase.Epoch ? MaxEpochs : 0,
            StopsEarly = phase == TrainingPhase.Epoch,
            Step = step,
            Steps = steps,
            TrialNumber = 0,
            TrainerName = _trainerName,
            Metric = 0,
            MetricName = "",
            ElapsedSeconds = _clock.Elapsed.TotalSeconds
        });
}
