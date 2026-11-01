using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using MLoop.Core.Prediction;

namespace MLoop.Core.Evaluation;

/// <summary>
/// Scores object-detection predictions with mean Average Precision (mAP) using ML.NET's built-in
/// <see cref="TorchSharpCatalog.EvaluateObjectDetection"/> evaluator (the same AutoFormerV2 metric
/// the trainer is built on) rather than a hand-rolled IoU/AP implementation.
/// </summary>
/// <remarks>
/// The evaluator consumes the scored <see cref="IDataView"/> produced by the object-detection
/// model (see <c>TrainingRunner.RunObjectDetectionAsync</c>): the loader's original string label
/// column (<c>Label</c>, a vector of class names) and box column (<c>BoundingBoxes</c>) pass
/// through, and the trainer emits <c>PredictedBoundingBoxes</c> plus <c>Score</c> and a key-typed
/// <c>PredictedLabel</c> that a trailing <c>MapKeyToValue</c> turns back into a visible string
/// column. <see cref="TorchSharpCatalog.EvaluateObjectDetection"/> reads both label columns
/// as <em>class-name strings</em> (not keys), so column resolution here takes the visible string
/// columns — the original <c>Label</c> and the mapped-back <c>PredictedLabel</c>.
/// </remarks>
public static class ObjectDetectionEvaluator
{
    /// <summary>Actual label column (vector of class-name strings) from the loader.</summary>
    public const string ActualLabelColumn = "Label";

    /// <summary>Actual bounding-box column (x0 y0 x1 y1 per object), passed through from the loader.</summary>
    public const string ActualBoundingBoxColumn = "BoundingBoxes";

    /// <summary>Predicted label column (vector of class-name strings) after <c>MapKeyToValue</c>.</summary>
    public const string PredictedLabelColumn = "PredictedLabel";

    /// <summary>Predicted bounding-box column emitted by the ObjectDetection trainer.</summary>
    public const string PredictedBoundingBoxColumn = "PredictedBoundingBoxes";

    /// <summary>Confidence score column emitted by the ObjectDetection trainer.</summary>
    public const string ScoreColumn = "Score";

    /// <summary>
    /// Computes mAP metrics over a scored object-detection <see cref="IDataView"/>.
    /// </summary>
    /// <param name="mlContext">The ML.NET context.</param>
    /// <param name="scoredData">The output of <c>model.Transform(testData)</c> for an OD model.</param>
    /// <returns>
    /// <c>map_50</c> (mAP at IoU 0.5, PASCAL VOC) and <c>map_50_95</c> (mAP averaged over IoU
    /// 0.5–0.95, COCO). Values are in <c>[0, 1]</c>.
    /// </returns>
    /// <exception cref="InvalidOperationException">A required column is missing from the schema.</exception>
    public static Dictionary<string, double> Evaluate(MLContext mlContext, IDataView scoredData)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(scoredData);

        var schema = scoredData.Schema;
        var labelCol = ResolveColumn(schema, ActualLabelColumn);
        var actualBoxCol = ResolveColumn(schema, ActualBoundingBoxColumn);
        var predictedLabelCol = ResolveColumn(schema, PredictedLabelColumn);
        var predictedBoxCol = ResolveColumn(schema, PredictedBoundingBoxColumn);
        var scoreCol = ResolveColumn(schema, ScoreColumn);

        var metrics = mlContext.MulticlassClassification.EvaluateObjectDetection(
            scoredData, labelCol, actualBoxCol, predictedLabelCol, predictedBoxCol, scoreCol);

        return new Dictionary<string, double>
        {
            { "map_50", metrics.MAP50 },
            { "map_50_95", metrics.MAP50_95 }
        };
    }

    /// <summary>
    /// The mAP a user gets at <paramref name="scoreThreshold"/> and the mAP over everything the model kept.
    /// <c>map_50</c> and <c>map_50_95</c> are scored on the boxes at or above the threshold — what <c>predict</c> shows by default,
    /// and what the quality gate judges; <c>map_50_ranked</c> and <c>map_50_95_ranked</c> are scored on every box the model
    /// kept (down to <see cref="DetectionScoreThreshold.Floor"/> for a model trained now, 0.5 for one trained before), which is
    /// mAP as it is usually reported. A detector that ranks boxes well and is never confident enough scores 0 on the first
    /// pair and above 0 on the second.
    /// </summary>
    public static Dictionary<string, double> Evaluate(MLContext mlContext, IDataView scoredData, double scoreThreshold)
    {
        var ranked = Evaluate(mlContext, scoredData);
        var operating = scoreThreshold <= DetectionScoreThreshold.Floor
            ? ranked
            : Evaluate(mlContext, WithScoreAtLeast(mlContext, scoredData, scoreThreshold));

        return new Dictionary<string, double>
        {
            { "map_50", operating["map_50"] },
            { "map_50_95", operating["map_50_95"] },
            { "map_50_ranked", ranked["map_50"] },
            { "map_50_95_ranked", ranked["map_50_95"] },
        };
    }

    /// <summary>
    /// The scored data with every predicted box below <paramref name="scoreThreshold"/> removed, the labels, boxes and scores
    /// of the remaining ones kept in step. The ground-truth columns pass through.
    /// </summary>
    public static IDataView WithScoreAtLeast(MLContext mlContext, IDataView scoredData, double scoreThreshold)
    {
        var schema = scoredData.Schema;
        var labelCol = ResolveColumn(schema, ActualLabelColumn);
        var actualBoxCol = ResolveColumn(schema, ActualBoundingBoxColumn);
        var predictedLabelCol = ResolveColumn(schema, PredictedLabelColumn);
        var predictedBoxCol = ResolveColumn(schema, PredictedBoundingBoxColumn);
        var scoreCol = ResolveColumn(schema, ScoreColumn);

        var rows = new List<KeptRow>();
        using (var cursor = scoredData.GetRowCursor(new[] { labelCol, actualBoxCol, predictedLabelCol, predictedBoxCol, scoreCol }))
        {
            var labelGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(labelCol);
            var actualBoxGetter = cursor.GetGetter<VBuffer<float>>(actualBoxCol);
            var predictedLabelGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(predictedLabelCol);
            var predictedBoxGetter = cursor.GetGetter<VBuffer<float>>(predictedBoxCol);
            var scoreGetter = cursor.GetGetter<VBuffer<float>>(scoreCol);

            VBuffer<ReadOnlyMemory<char>> label = default, predictedLabel = default;
            VBuffer<float> actualBox = default, predictedBox = default, score = default;
            while (cursor.MoveNext())
            {
                labelGetter(ref label);
                actualBoxGetter(ref actualBox);
                predictedLabelGetter(ref predictedLabel);
                predictedBoxGetter(ref predictedBox);
                scoreGetter(ref score);

                var scores = score.DenseValues().ToArray();
                var labels = predictedLabel.DenseValues().Select(l => l.ToString()).ToArray();
                var boxes = predictedBox.DenseValues().ToArray();

                var keptLabels = new List<string>();
                var keptBoxes = new List<float>();
                var keptScores = new List<float>();
                for (var i = 0; i < scores.Length; i++)
                {
                    if (scores[i] < scoreThreshold)
                        continue;

                    keptScores.Add(scores[i]);
                    keptLabels.Add(i < labels.Length ? labels[i] : string.Empty);
                    for (var c = 0; c < 4; c++)
                        keptBoxes.Add(i * 4 + c < boxes.Length ? boxes[i * 4 + c] : 0f);
                }

                rows.Add(new KeptRow
                {
                    Label = label.DenseValues().Select(l => l.ToString()).ToArray(),
                    BoundingBoxes = actualBox.DenseValues().ToArray(),
                    PredictedLabel = [.. keptLabels],
                    PredictedBoundingBoxes = [.. keptBoxes],
                    Score = [.. keptScores],
                });
            }
        }

        return mlContext.Data.LoadFromEnumerable(rows);
    }

    /// <summary>One scored image with the columns the evaluator reads, under the names it reads them by.</summary>
    private sealed class KeptRow
    {
        [VectorType] public string[] Label { get; set; } = [];
        [VectorType] public float[] BoundingBoxes { get; set; } = [];
        [VectorType] public string[] PredictedLabel { get; set; } = [];
        [VectorType] public float[] PredictedBoundingBoxes { get; set; } = [];
        [VectorType] public float[] Score { get; set; } = [];
    }

    /// <summary>
    /// Resolves the visible column with the given name. The schema indexer returns the latest
    /// (non-hidden) definition, which is what the evaluator needs — e.g. the string
    /// <c>PredictedLabel</c> that shadows the trainer's hidden key-typed column.
    /// </summary>
    private static DataViewSchema.Column ResolveColumn(DataViewSchema schema, string name)
    {
        var col = schema.GetColumnOrNull(name);
        if (col is not null)
            return col.Value;

        throw new InvalidOperationException(
            $"Object-detection evaluation requires a '{name}' column in the model output, but it " +
            $"was not found. Available columns: " +
            $"{string.Join(", ", schema.Select(c => $"{c.Name}:{c.Type}{(c.IsHidden ? " (hidden)" : "")}"))}.");
    }
}
