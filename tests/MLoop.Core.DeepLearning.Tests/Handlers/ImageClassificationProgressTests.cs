using Microsoft.ML.Vision;
using MLoop.Core.DeepLearning;
using MLoop.Core.Models;
using static Microsoft.ML.Vision.ImageClassificationTrainer;

namespace MLoop.Core.DeepLearning.Tests.Handlers;

public class ImageClassificationProgressTests
{
    private sealed class Recorder : IProgress<TrainingProgress>
    {
        public List<TrainingProgress> Events { get; } = [];
        public void Report(TrainingProgress value) => Events.Add(value);
    }

    private static ImageClassificationMetrics Featurized(ImageClassificationMetrics.Dataset dataset, int index) => new()
    {
        Bottleneck = new BottleneckMetrics { DatasetUsed = dataset, Index = index }
    };

    private static ImageClassificationMetrics Epoch(ImageClassificationMetrics.Dataset dataset, int epoch) => new()
    {
        Train = new TrainMetrics { DatasetUsed = dataset, Epoch = epoch }
    };

    [Fact]
    public void Featurization_counts_every_image_across_the_train_and_validation_passes()
    {
        // The trainer featurizes its train share and its held-out validation share in two passes,
        // each numbering images from its own start; the user sees one count over all images.
        var recorder = new Recorder();
        var progress = new ImageClassificationProgress(recorder, "IC", images: 5);

        for (int i = 0; i < 4; i++) progress.OnMetrics(Featurized(ImageClassificationMetrics.Dataset.Train, i));
        progress.OnMetrics(Featurized(ImageClassificationMetrics.Dataset.Validation, 0));

        Assert.All(recorder.Events, e => Assert.Equal(TrainingPhase.Featurize, e.Phase));
        Assert.Equal([1, 2, 3, 4, 5], recorder.Events.Select(e => e.Step));
        Assert.All(recorder.Events, e => Assert.Equal(5, e.Steps));
    }

    [Fact]
    public void An_epoch_is_reported_once_counted_from_one_and_marked_as_capped()
    {
        // Each epoch is reported on the train set and again on the validation set.
        var recorder = new Recorder();
        var progress = new ImageClassificationProgress(recorder, "IC", images: 5);

        progress.OnMetrics(Epoch(ImageClassificationMetrics.Dataset.Train, 0));
        progress.OnMetrics(Epoch(ImageClassificationMetrics.Dataset.Validation, 0));
        progress.OnMetrics(Epoch(ImageClassificationMetrics.Dataset.Train, 1));

        Assert.Equal([1, 2], recorder.Events.Select(e => e.Epoch));
        Assert.All(recorder.Events, e =>
        {
            Assert.Equal(TrainingPhase.Epoch, e.Phase);
            Assert.Equal(ImageClassificationProgress.MaxEpochs, e.MaxEpochs);
            Assert.True(e.StopsEarly);
        });
    }
}
