using MLoop.Core.ImageAnomaly;

namespace MLoop.API;

// The bodies of the prediction and embedding endpoints (PredictEndpoint, EmbedEndpoint, and the image-model answers beside them),
// as the experiment ones are in ExperimentResponses.cs: the names and kinds of the fields are in the types, and the OpenAPI
// document describes each instead of `object`. A null property is omitted by the server.

/// <summary>
/// <c>POST /predict</c>: the model that answered, when, and one row per input — a prediction row for a table model, a
/// forecast step for a forecasting model, the boxes of an image for an object-detection model.
/// </summary>
public sealed record PredictResponse<TRow>(
    string ModelName,
    string ExperimentId,
    string ModelSha256,
    DateTime PredictedAt,
    string Task,
    int Count,
    IReadOnlyList<TRow> Predictions,
    IReadOnlyList<string>? Warnings);

/// <summary>Builds a <see cref="PredictResponse{TRow}"/> with the row type read from the predictions given; the count is theirs.</summary>
public static class PredictResponse
{
    public static PredictResponse<TRow> Create<TRow>(
        string modelName, string experimentId, string modelSha256, DateTime predictedAt, string task,
        IReadOnlyList<TRow> predictions, IReadOnlyList<string>? warnings) =>
        new(modelName, experimentId, modelSha256, predictedAt, task, predictions.Count, predictions, warnings);
}

/// <summary>The error of an image that cannot be read: which one (its index in the request) and why.</summary>
public sealed record ImageErrorResponse(string Error, int Index);

/// <summary><c>POST /predict</c> for an image-anomaly model.</summary>
public sealed record ImageAnomalyResponse(
    string ModelName,
    string ExperimentId,
    string Task,
    DateTime PredictedAt,
    int Count,
    EncoderRef Encoder,
    GridView Grid,
    ScoreBaseline? Baseline,
    IReadOnlyList<ImageScoreView> Images,
    IReadOnlyList<string> Warnings);

/// <summary>The encoder that made the patch vectors the scores are measured on.</summary>
public sealed record EncoderRef(string Id, string? Sha256, string Preprocessor);

/// <summary>The grid of patches an image is cut into.</summary>
public sealed record GridView(int Rows, int Columns, int PatchSize, int CropSize);

/// <summary>One image: its anomaly score, the crop the grid lies on (in image pixels), and each patch's distance.</summary>
public sealed record ImageScoreView(double Score, CropView Crop, float[] Distances);

/// <summary>The part of the image the model looked at, in image pixels.</summary>
public sealed record CropView(double X0, double Y0, double X1, double Y1);
