using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using MLoop.Core.Evaluation;

namespace MLoop.Core.DeepLearning.Tests.Evaluation;

/// <summary>
/// A held-out set is a handful of images, and a detector keeps boxes down to a low score: some of those boxes name a class
/// that none of the held-out images contains. ML.NET's mAP evaluator indexes a per-class table built from the ground truth with
/// the class of every predicted box, so such a box made the evaluation throw — after the training that produced the model had
/// finished. Built on scored rows, so none of it needs libtorch.
/// </summary>
public class UngroundedClassEvaluationTests
{
    private sealed class ScoredRow
    {
        [VectorType] public string[] Label { get; set; } = [];
        [VectorType] public float[] BoundingBoxes { get; set; } = [];
        [VectorType] public string[] PredictedLabel { get; set; } = [];
        [VectorType] public float[] PredictedBoundingBoxes { get; set; } = [];
        [VectorType] public float[] Score { get; set; } = [];
    }

    /// <summary>Two held-out images that contain only "red"; the detector also saw a "green" in the first, at 0.3.</summary>
    private static IDataView WithAGreenFalsePositive(MLContext ml) => ml.Data.LoadFromEnumerable(new[]
    {
        new ScoredRow
        {
            Label = ["red"], BoundingBoxes = [10f, 10f, 40f, 40f],
            PredictedLabel = ["red", "green"],
            PredictedBoundingBoxes = [10f, 10f, 40f, 40f, 60f, 60f, 90f, 90f],
            Score = [0.9f, 0.3f],
        },
        new ScoredRow
        {
            Label = ["red"], BoundingBoxes = [20f, 20f, 50f, 50f],
            PredictedLabel = ["red"], PredictedBoundingBoxes = [20f, 20f, 50f, 50f], Score = [0.8f],
        },
    });

    private static IDataView WithoutIt(MLContext ml) => ml.Data.LoadFromEnumerable(new[]
    {
        new ScoredRow
        {
            Label = ["red"], BoundingBoxes = [10f, 10f, 40f, 40f],
            PredictedLabel = ["red"], PredictedBoundingBoxes = [10f, 10f, 40f, 40f], Score = [0.9f],
        },
        new ScoredRow
        {
            Label = ["red"], BoundingBoxes = [20f, 20f, 50f, 50f],
            PredictedLabel = ["red"], PredictedBoundingBoxes = [20f, 20f, 50f, 50f], Score = [0.8f],
        },
    });

    /// <summary>The companion of the fix: the shape that made the library throw still does, so the tests below see what they guard.</summary>
    [Fact]
    public void TheLibraryEvaluator_ThrowsOnAClassWithNoGroundTruth()
    {
        var ml = new MLContext(seed: 42);
        var scored = WithAGreenFalsePositive(ml);

        Assert.Throws<KeyNotFoundException>(() => ml.MulticlassClassification.EvaluateObjectDetection(
            scored, scored.Schema["Label"], scored.Schema["BoundingBoxes"], scored.Schema["PredictedLabel"],
            scored.Schema["PredictedBoundingBoxes"], scored.Schema["Score"]));
    }

    [Fact]
    public void Evaluate_ScoresAsIfTheUngroundedBoxWereNotThere()
    {
        var ml = new MLContext(seed: 42);

        var with = ObjectDetectionEvaluator.Evaluate(ml, WithAGreenFalsePositive(ml));
        var without = ObjectDetectionEvaluator.Evaluate(ml, WithoutIt(ml));

        Assert.Equal(without["map_50"], with["map_50"]);
        Assert.Equal(without["map_50_95"], with["map_50_95"]);
        Assert.Equal(1.0, with["map_50"], precision: 6);
    }

    [Fact]
    public void Evaluate_AtAnOperatingThreshold_DoesNotThrowEither_AndKeepsTheRankedPair()
    {
        var ml = new MLContext(seed: 42);

        var metrics = ObjectDetectionEvaluator.Evaluate(ml, WithAGreenFalsePositive(ml), 0.2);

        Assert.Equal(1.0, metrics["map_50"], precision: 6);
        Assert.Equal(1.0, metrics["map_50_ranked"], precision: 6);
    }

    [Fact]
    public void AClassTheImagesDoContain_IsStillScored()
    {
        var ml = new MLContext(seed: 42);
        // The green box is on a green object this time, in the wrong place: it counts against the model.
        var scored = ml.Data.LoadFromEnumerable(new[]
        {
            new ScoredRow
            {
                Label = ["red", "green"], BoundingBoxes = [10f, 10f, 40f, 40f, 60f, 60f, 90f, 90f],
                PredictedLabel = ["red", "green"],
                PredictedBoundingBoxes = [10f, 10f, 40f, 40f, 200f, 200f, 230f, 230f],
                Score = [0.9f, 0.9f],
            },
        });

        var metrics = ObjectDetectionEvaluator.Evaluate(ml, scored);

        Assert.True(metrics["map_50"] < 0.75, $"the missed green object must cost: {metrics["map_50"]}");
    }
}
