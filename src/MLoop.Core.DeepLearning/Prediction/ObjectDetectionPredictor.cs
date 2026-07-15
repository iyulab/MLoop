using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Prediction;

/// <summary>
/// A single detected object: its class label, confidence score, and bounding box in
/// <c>x0 y0 x1 y1</c> pixel coordinates (top-left / bottom-right), the same convention the
/// loaders normalize to.
/// </summary>
public sealed record ObjectDetectionBox(
    string Label, float Score, float X0, float Y0, float X1, float Y1);

/// <summary>All objects detected in one image, keyed by the image's source path.</summary>
public sealed record ImageDetectionResult(
    string ImagePath, IReadOnlyList<ObjectDetectionBox> Detections);

/// <summary>
/// Extracts per-image detections (label + score + box) from the scored <see cref="IDataView"/>
/// produced by an object-detection model. The structural twin of
/// <see cref="MLoop.Core.Evaluation.ObjectDetectionEvaluator"/>: that one scores the predictions
/// with mAP, this one surfaces the raw detections for <c>mloop predict</c>.
/// </summary>
/// <remarks>
/// The scored schema (see <c>AutoMLRunner.RunObjectDetectionAsync</c>) carries the loader's
/// <c>ImagePath</c> string through, and the trainer emits <c>PredictedBoundingBoxes</c>
/// (a flat <c>VBuffer&lt;float&gt;</c> of four values per object), <c>Score</c> (one confidence per
/// object) and a key-typed <c>PredictedLabel</c> that a trailing <c>MapKeyToValue</c> turns back
/// into a visible vector of class-name strings. As with the evaluator, the visible string
/// <c>PredictedLabel</c> is read — not the hidden key column.
/// </remarks>
public static class ObjectDetectionPredictor
{
    /// <summary>Image source path column, passed through from the loader.</summary>
    public const string ImagePathColumn = "ImagePath";

    /// <summary>Predicted label column (vector of class-name strings) after <c>MapKeyToValue</c>.</summary>
    public const string PredictedLabelColumn = "PredictedLabel";

    /// <summary>Predicted bounding-box column emitted by the ObjectDetection trainer (4 floats per box).</summary>
    public const string PredictedBoundingBoxColumn = "PredictedBoundingBoxes";

    /// <summary>Confidence score column emitted by the ObjectDetection trainer (one per box).</summary>
    public const string ScoreColumn = "Score";

    /// <summary>Number of float coordinates per bounding box (<c>x0 y0 x1 y1</c>).</summary>
    private const int BoxStride = 4;

    /// <summary>
    /// Reads detections for every image in the scored data view.
    /// </summary>
    /// <param name="mlContext">The ML.NET context (reserved for symmetry with the evaluator).</param>
    /// <param name="scoredData">The output of <c>model.Transform(data)</c> for an OD model.</param>
    /// <returns>One <see cref="ImageDetectionResult"/> per input image, in input order.</returns>
    /// <exception cref="InvalidOperationException">A required column is missing from the schema.</exception>
    public static IReadOnlyList<ImageDetectionResult> Predict(MLContext mlContext, IDataView scoredData)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(scoredData);

        var schema = scoredData.Schema;
        var imagePathCol = ResolveColumn(schema, ImagePathColumn);
        var predictedLabelCol = ResolveColumn(schema, PredictedLabelColumn);
        var predictedBoxCol = ResolveColumn(schema, PredictedBoundingBoxColumn);
        var scoreCol = ResolveColumn(schema, ScoreColumn);

        var results = new List<ImageDetectionResult>();

        using var cursor = scoredData.GetRowCursor(
            new[] { imagePathCol, predictedLabelCol, predictedBoxCol, scoreCol });

        var imagePathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(imagePathCol);
        var labelGetter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(predictedLabelCol);
        var boxGetter = cursor.GetGetter<VBuffer<float>>(predictedBoxCol);
        var scoreGetter = cursor.GetGetter<VBuffer<float>>(scoreCol);

        ReadOnlyMemory<char> imagePath = default;
        VBuffer<ReadOnlyMemory<char>> labels = default;
        VBuffer<float> boxes = default;
        VBuffer<float> scores = default;

        while (cursor.MoveNext())
        {
            imagePathGetter(ref imagePath);
            labelGetter(ref labels);
            boxGetter(ref boxes);
            scoreGetter(ref scores);

            var labelValues = labels.DenseValues().ToArray();
            var boxValues = boxes.DenseValues().ToArray();
            var scoreValues = scores.DenseValues().ToArray();

            // The trainer emits one score and one label per detected object, and four box
            // coordinates per object. They are parallel arrays; iterate by score count and guard
            // the label/box indices so a ragged buffer degrades gracefully rather than throwing.
            var detections = new List<ObjectDetectionBox>(scoreValues.Length);
            for (int i = 0; i < scoreValues.Length; i++)
            {
                var label = i < labelValues.Length ? labelValues[i].ToString() : string.Empty;
                var b = i * BoxStride;
                detections.Add(new ObjectDetectionBox(
                    label,
                    scoreValues[i],
                    b + 0 < boxValues.Length ? boxValues[b + 0] : 0f,
                    b + 1 < boxValues.Length ? boxValues[b + 1] : 0f,
                    b + 2 < boxValues.Length ? boxValues[b + 2] : 0f,
                    b + 3 < boxValues.Length ? boxValues[b + 3] : 0f));
            }

            results.Add(new ImageDetectionResult(imagePath.ToString(), detections));
        }

        return results;
    }

    /// <summary>
    /// Resolves the visible column with the given name. The schema indexer returns the latest
    /// (non-hidden) definition — the string <c>PredictedLabel</c> that shadows the trainer's hidden
    /// key-typed column.
    /// </summary>
    private static DataViewSchema.Column ResolveColumn(DataViewSchema schema, string name)
    {
        var col = schema.GetColumnOrNull(name);
        if (col is not null)
            return col.Value;

        throw new InvalidOperationException(
            $"Object-detection prediction requires a '{name}' column in the model output, but it " +
            $"was not found. Available columns: " +
            $"{string.Join(", ", schema.Select(c => $"{c.Name}:{c.Type}{(c.IsHidden ? " (hidden)" : "")}"))}.");
    }
}
