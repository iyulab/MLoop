using System.Diagnostics;
using MLoop.Core.Embedding;
using MLoop.Core.ImageAnomaly;

namespace MLoop.API;

/// <summary>
/// <c>POST /predict</c> for an image-anomaly model: images by content in, a score and a grid of patch distances per image
/// out (<see cref="ImageAnomalyScorer"/>, the same reading <c>mloop predict</c> does). The model is its bank, read once and kept
/// while its files stay as they are; an encoder that is gone or has other weights than the bank was built with is a 409, as
/// <c>POST /embed</c> answers an encoder that is not installed.
/// </summary>
public static class ImageAnomalyPredictions
{
    public static async Task<IResult> RespondAsync(
        IReadOnlyList<byte[]> images,
        string modelName,
        string experimentId,
        string modelPath,
        EmbedderPool pool,
        ILogger logger,
        Stopwatch stopwatch,
        CancellationToken ct)
    {
        var (scorer, refusal) = ImageAnomalyScorer.Open(modelPath, pool);
        if (scorer is null)
            return Results.Conflict(new { error = refusal });

        IReadOnlyList<ImageScore> scores;
        try
        {
            scores = await Task.Run(() => scorer.Score(images, ct), ct);
        }
        catch (UnreadableImageException ex)
        {
            return Results.BadRequest(new { error = ex.Message, index = ex.Index });
        }

        var model = scorer.Model;
        logger.LogInformation("Image anomaly for '{ModelName}': {Images} image(s) in {ElapsedMs}ms", modelName, scores.Count, stopwatch.ElapsedMilliseconds);
        return Results.Ok(new
        {
            modelName,
            experimentId,
            task = "image-anomaly",
            predictedAt = DateTime.UtcNow,
            count = scores.Count,
            encoder = new { model.Encoder.Id, model.Encoder.Sha256, model.Encoder.Preprocessor },
            grid = new { model.Grid.Rows, model.Grid.Columns, model.Grid.PatchSize, model.Grid.CropSize },
            baseline = model.Baseline,
            images = scores.Select(s => new
            {
                score = s.Score,
                crop = new { s.Crop.X0, s.Crop.Y0, s.Crop.X1, s.Crop.Y1 },
                distances = s.Distances,
            }),
            warnings = new List<string>(),
        });
    }
}
