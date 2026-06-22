using Microsoft.ML;
using Microsoft.ML.TorchSharp;

namespace MLoop.Core.Evaluation;

/// <summary>
/// Scores object-detection predictions with mean Average Precision (mAP) using ML.NET's built-in
/// <see cref="TorchSharpCatalog.EvaluateObjectDetection"/> evaluator (the same AutoFormerV2 metric
/// the trainer is built on) rather than a hand-rolled IoU/AP implementation.
/// </summary>
/// <remarks>
/// The evaluator consumes the scored <see cref="IDataView"/> produced by the object-detection
/// model (see <c>AutoMLRunner.RunObjectDetectionAsync</c>): the loader's original string label
/// column (<c>Label</c>, a vector of class names) and box column (<c>BoundingBoxes</c>) pass
/// through, and the trainer emits <c>PredictedBoundingBoxes</c> plus <c>Score</c> and a key-typed
/// <c>PredictedLabel</c> that a trailing <c>MapKeyToValue</c> turns back into a visible string
/// column. <see cref="EvaluateObjectDetection.EvaluateObjectDetection"/> reads both label columns
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
