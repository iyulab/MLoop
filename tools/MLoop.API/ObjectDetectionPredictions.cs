using System.Diagnostics;
using System.Security.Cryptography;
using Microsoft.ML;
using MLoop.API.Caching;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Prediction;
using MLoop.DataStore.Interfaces;

namespace MLoop.API;

/// <summary>One object found in an image: its class, the model's score, and its box in pixels (top-left, bottom-right).</summary>
public sealed record DetectedObject(string Label, float Score, float X0, float Y0, float X1, float Y1);

/// <summary>An image's answer from an object-detection model — the image named by its SHA-256, as it is logged.</summary>
public sealed record ImageDetections(string ImageSha256, IReadOnlyList<DetectedObject> Detections, string? PredictionId = null);

/// <summary>
/// <c>POST /predict</c> for an object-detection model: images by content in, a list of boxes per image out
/// (<see cref="ObjectDetectionPredictor.Detect"/>, the same reading <c>mloop predict</c> does). Each image is
/// logged like a row — input its SHA-256, output its detections — so feedback can name it.
/// </summary>
public static class ObjectDetectionPredictions
{
    public static async Task<IResult> RespondAsync(
        IReadOnlyList<byte[]> images,
        List<string> imageFiles,
        string modelName,
        string experimentId,
        string modelPath,
        MLContext mlContext,
        IModelCache modelCache,
        IExperimentStore experimentStore,
        IPredictionLogger predictionLogger,
        IConfiguration configuration,
        ILogger logger,
        Stopwatch stopwatch,
        CancellationToken ct)
    {
        // The caller's finally releases these, as it does an image-classification request's.
        foreach (var image in images)
        {
            var file = MLoop.Core.Data.ScratchFiles.New("mloop_predict_image_", ".img");
            imageFiles.Add(file);
            await File.WriteAllBytesAsync(file, image, ct);
        }

        MLoop.Core.Runtime.RuntimeManager.EnsureRuntimeForTask("object-detection");
        var model = modelCache.GetOrLoad(modelPath);
        var labelColumn = (await experimentStore.LoadAsync(modelName, experimentId, ct))?.Config?.LabelColumn;

        var found = ObjectDetectionPredictor.Detect(mlContext, model, imageFiles, labelColumn);
        var predictions = found
            .Select((result, i) => new ImageDetections(
                Convert.ToHexStringLower(SHA256.HashData(images[i])),
                [.. result.Detections.Select(d => new DetectedObject(d.Label, d.Score, d.X0, d.Y0, d.X1, d.Y1))]))
            .ToList();

        logger.LogInformation("Object detection for '{ModelName}': {Images} image(s), {Objects} object(s) in {ElapsedMs}ms",
            modelName, predictions.Count, predictions.Sum(p => p.Detections.Count), stopwatch.ElapsedMilliseconds);

        var warnings = new List<string>();
        if (configuration.GetValue("Serve:LogPredictions", true))
        {
            try
            {
                var loggedAt = DateTimeOffset.UtcNow;
                var ids = predictions.Select(_ => PredictionIds.New()).ToArray();
                await predictionLogger.LogBatchAsync(
                    modelName,
                    experimentId,
                    predictions.Select((p, i) => new PredictionLogEntry(
                        modelName, experimentId,
                        new Dictionary<string, object> { ["imageSha256"] = p.ImageSha256 },
                        p.Detections,
                        p.Detections.Count > 0 ? p.Detections.Max(d => d.Score) : null,
                        loggedAt, ids[i])),
                    ct);
                predictions = [.. predictions.Select((p, i) => p with { PredictionId = ids[i] })];
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                logger.LogWarning(ex, "Detections for '{ModelName}' were served but not logged", modelName);
                warnings.Add("These predictions were not logged, so they carry no predictionId: " + ex.Message);
            }
        }

        return Results.Ok(new
        {
            modelName,
            experimentId,
            predictedAt = DateTime.UtcNow,
            task = "object-detection",
            count = predictions.Count,
            predictions,
            warnings
        });
    }
}
