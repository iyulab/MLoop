using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Evaluation;
using MLoop.Core.Prediction;

namespace MLoop.Core.DeepLearning.Tests.Prediction;

/// <summary>
/// The score a box needs to be reported. A detector is trained to keep every box down to 0.05 and a user is shown the boxes of 0.5
/// and over unless they ask for another cut; the mAP the quality gate judges is scored at that cut and the ranking mAP over all of it.
/// Built on scored rows, so none of it needs libtorch.
/// </summary>
public class DetectionScoreThresholdTests
{
    private sealed class ScoredRow
    {
        public string ImagePath { get; set; } = "";
        [VectorType] public string[] Label { get; set; } = [];
        [VectorType] public float[] BoundingBoxes { get; set; } = [];
        [VectorType] public string[] PredictedLabel { get; set; } = [];
        [VectorType] public float[] PredictedBoundingBoxes { get; set; } = [];
        [VectorType] public float[] Score { get; set; } = [];
    }

    private static IDataView Scored(MLContext ml) => ml.Data.LoadFromEnumerable(new[]
    {
        new ScoredRow
        {
            ImagePath = "a.jpg",
            Label = ["red"], BoundingBoxes = [10f, 10f, 40f, 40f],
            PredictedLabel = ["red", "red", "blue"],
            PredictedBoundingBoxes = [10f, 10f, 40f, 40f, 60f, 60f, 90f, 90f, 5f, 70f, 30f, 95f],
            Score = [0.62f, 0.21f, 0.07f],
        },
        new ScoredRow
        {
            ImagePath = "b.jpg",
            Label = ["blue"], BoundingBoxes = [20f, 20f, 50f, 50f],
            PredictedLabel = ["blue"], PredictedBoundingBoxes = [20f, 20f, 50f, 50f], Score = [0.3f],
        },
    });

    [Theory]
    [InlineData(0.5, 1, 0)]     // the default: only the 0.62 box of the first image
    [InlineData(0.2, 2, 1)]     // 0.62 and 0.21 in the first, 0.3 in the second
    [InlineData(0.05, 3, 1)]    // everything the detector kept
    [InlineData(0.9, 0, 0)]
    public void Predict_LeavesOutBoxesBelowTheThreshold(double threshold, int firstImage, int secondImage)
    {
        var ml = new MLContext(seed: 42);

        var results = ObjectDetectionPredictor.Predict(ml, Scored(ml), threshold);

        Assert.Equal(firstImage, results[0].Detections.Count);
        Assert.Equal(secondImage, results[1].Detections.Count);
        Assert.All(results.SelectMany(r => r.Detections), box => Assert.True(box.Score >= threshold));
    }

    [Fact]
    public void Predict_WithoutAThreshold_CutsAtTheDefault()
    {
        var ml = new MLContext(seed: 42);

        var results = ObjectDetectionPredictor.Predict(ml, Scored(ml));

        Assert.Equal(DetectionScoreThreshold.Default, 0.5);
        Assert.Single(results[0].Detections);
        Assert.Empty(results[1].Detections);
    }

    [Fact]
    public void WithScoreAtLeast_KeepsLabelsBoxesAndScoresOfTheSurvivorsInStep()
    {
        var ml = new MLContext(seed: 42);

        var kept = ObjectDetectionEvaluator.WithScoreAtLeast(ml, Scored(ml), 0.2);

        // Read the filtered view back as the predictor reads a scored one (it carries no ImagePath, so by cursor).
        using var cursor = kept.GetRowCursor(kept.Schema);
        var labels = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(kept.Schema["PredictedLabel"]);
        var boxes = cursor.GetGetter<VBuffer<float>>(kept.Schema["PredictedBoundingBoxes"]);
        var scores = cursor.GetGetter<VBuffer<float>>(kept.Schema["Score"]);
        var truth = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(kept.Schema["Label"]);

        VBuffer<ReadOnlyMemory<char>> label = default, actual = default;
        VBuffer<float> box = default, score = default;

        Assert.True(cursor.MoveNext());
        labels(ref label); boxes(ref box); scores(ref score); truth(ref actual);
        Assert.Equal(["red", "red"], label.DenseValues().Select(l => l.ToString()));
        Assert.Equal([10f, 10f, 40f, 40f, 60f, 60f, 90f, 90f], box.DenseValues());
        Assert.Equal([0.62f, 0.21f], score.DenseValues());
        Assert.Equal(["red"], actual.DenseValues().Select(l => l.ToString()));

        Assert.True(cursor.MoveNext());
        labels(ref label); scores(ref score);
        Assert.Equal(["blue"], label.DenseValues().Select(l => l.ToString()));
        Assert.Equal([0.3f], score.DenseValues());
        Assert.False(cursor.MoveNext());
    }

    [Theory]
    [InlineData(0.5, true)]
    [InlineData(0.05, true)]
    [InlineData(1.0, true)]
    [InlineData(0.0, false)]
    [InlineData(-0.1, false)]
    [InlineData(1.01, false)]
    [InlineData(double.NaN, false)]
    public void IsValid_AcceptsOnlyAboveZeroAndAtMostOne(double threshold, bool valid)
    {
        Assert.Equal(valid, DetectionScoreThreshold.IsValid(threshold));
    }

    [Fact]
    public void TheFloorIsBelowTheDefault_SoALowerCutShowsMoreOfANewModel()
    {
        Assert.True(DetectionScoreThreshold.Floor < DetectionScoreThreshold.Default);
    }
}
