using System.Diagnostics;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;
using Microsoft.ML;
using MLoop.API.Caching;
using MLoop.CLI.Infrastructure;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;
using Serilog;

namespace MLoop.API;

/// <summary>
/// <c>POST /predict</c> — single and batch predictions, image content for vision models.
/// </summary>
public static class PredictEndpoint
{
    public static void MapPredict(this WebApplication app)
    {
        // Prediction endpoint (secured)
        app.MapPost("/predict", async (
            HttpRequest request,
            [FromQuery] string? name,
            [FromQuery] string? experiment,
            IModelRegistry registry,
            IExperimentStore experimentStore,
            MLContext mlContext,
            IModelCache modelCache,
            MLoop.Core.Embedding.EmbedderPool embedderPool,
            IPredictionLogger predictionLogger,
            IConfiguration configuration,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Resolve(name);

            // The body: rows as JSON, or — for an image model — images by their contents, as POST /embed takes them
            // (base64 under "images", or files in a multipart form). An empty body is no rows: forecasting needs none.
            JsonElement input = default;
            List<byte[]>? images = null;
            if (request.HasFormContentType)
            {
                images = await ImageInputs.FromFormAsync(await request.ReadFormAsync(ct), ct);
            }
            else
            {
                using var reader = new StreamReader(request.Body);
                var text = await reader.ReadToEndAsync(ct);
                if (!string.IsNullOrWhiteSpace(text))
                {
                    try
                    {
                        input = JsonDocument.Parse(text).RootElement.Clone();
                    }
                    catch (JsonException ex)
                    {
                        return Results.BadRequest(new { error = $"The body is not JSON: {ex.Message}" });
                    }
                    if (ImageInputs.TryGetImages(input, out var encoded))
                    {
                        var (decoded, unreadable) = ImageInputs.FromJson(encoded);
                        if (decoded is null)
                            return Results.BadRequest(new { error = unreadable });
                        images = decoded;
                    }
                }
            }
            if (images is not null)
            {
                if (ImageInputs.Refusal(images) is { } refused)
                    return refused;
                for (var i = 0; i < images.Count; i++)
                {
                    try
                    {
                        MLoop.Core.Embedding.ImagePreprocessor.EnsureDecodable(images[i], i);
                    }
                    catch (MLoop.Core.Embedding.UnreadableImageException ex)
                    {
                        return Results.BadRequest(new { error = ex.Message, index = ex.Index });
                    }
                }
            }

            var predictionCount = images?.Count ?? (input.ValueKind == JsonValueKind.Array ? input.GetArrayLength() : 1);
            var imageFiles = new List<string>();

            try
            {
                logger.LogInformation("Prediction request for '{ModelName}' with {Count} input(s)", modelName, predictionCount);

                // The model that answers: production by default, or ?experiment= — a shadow prediction by a
                // candidate, as `mloop predict --experiment`. Production does not move, and logged rows carry
                // the candidate's id, so drift (production's rows) and feedback keep the two apart.
                string answeringId;
                string modelPath;
                string? taskOfModel;
                if (!string.IsNullOrWhiteSpace(experiment))
                {
                    answeringId = experiment.Trim();
                    var (candidatePath, refusal) = ExperimentModels.Find(experimentStore, modelName, answeringId);
                    if (candidatePath is null)
                        return Results.NotFound(new { error = refusal });
                    modelPath = candidatePath;
                    taskOfModel = (await experimentStore.LoadAsync(modelName, answeringId, ct))?.Task;
                }
                else
                {
                    var productionModel = await registry.GetProductionAsync(modelName, ct);

                    if (productionModel == null)
                    {
                        logger.LogWarning("Prediction failed: No production model found for '{ModelName}'", modelName);
                        return Results.NotFound(new { error = $"No production model found for '{modelName}'. Train and promote a model first, or predict with a candidate: ?experiment=<exp-id>." });
                    }

                    answeringId = productionModel.ExperimentId;
                    taskOfModel = productionModel.Task;
                    modelPath = Path.Combine(registry.GetProductionPath(modelName), ExperimentLayout.ModelFileName);
                }

                if (!File.Exists(modelPath))
                {
                    logger.LogError("Model file not found at {ModelPath}", modelPath);
                    return Results.NotFound(new { error = $"Model file not found: {modelPath}" });
                }

                var taskType = taskOfModel ?? "regression";

                // Forecasting is horizon-based (stateful SSA replaying its training series), not
                // row-based — PredictionService.Predict rejects it outright. Body is an optional
                // {"horizon":N}; omitted (or a non-object/no body) uses the model's trained horizon.
                // Bypasses InputSchema/row-parsing entirely — neither applies to a horizon replay.
                if (string.Equals(taskType, "forecasting", StringComparison.OrdinalIgnoreCase))
                {
                    int? requestedHorizon = null;
                    if (input.ValueKind == JsonValueKind.Object &&
                        input.TryGetProperty("horizon", out var horizonProp) &&
                        horizonProp.ValueKind == JsonValueKind.Number)
                    {
                        requestedHorizon = horizonProp.GetInt32();
                    }

                    var (forecast, forecastError) = await ForecastReplayService.ComputeForecastAsync(
                        mlContext, modelPath, answeringId, requestedHorizon);

                    if (forecast is null)
                    {
                        logger.LogWarning("Forecast failed for '{ModelName}': {Error}", modelName, forecastError);
                        return Results.Problem(
                            title: "Forecast failed",
                            detail: forecastError ?? "Forecast failed.",
                            statusCode: StatusCodes.Status400BadRequest
                        );
                    }

                    var forecastRows = ForecastReplayService.BuildForecastRows(forecast);
                    logger.LogInformation("Forecast completed for '{ModelName}': {Count} steps in {ElapsedMs}ms",
                        modelName, forecastRows.Count, stopwatch.ElapsedMilliseconds);

                    return Results.Ok(new
                    {
                        modelName,
                        experimentId = answeringId,
                        modelSha256 = ModelFingerprint.Sha256(modelPath),
                        predictedAt = DateTime.UtcNow,
                        task = "forecasting",
                        count = forecastRows.Count,
                        predictions = forecastRows,
                        warnings = new List<string>()
                    });
                }

                // Object detection answers each image with a list of boxes, not one row's value: its own branch,
                // with no input schema or rows. Images come by content, as image classification takes them.
                if (MLoop.Core.Models.TaskTypes.Canonical(taskType) == "object-detection")
                {
                    if (images is null)
                        return Results.BadRequest(new { error = $"'{modelName}' is an object-detection model: send images by content — {{\"images\": [base64…]}} or a multipart form." });
                    return await ObjectDetectionPredictions.RespondAsync(
                        images, imageFiles, modelName, answeringId, modelPath, mlContext, modelCache, experimentStore,
                        predictionLogger, configuration, logger, stopwatch, ct);
                }

                // Image anomaly answers each image with a score and a grid of patch distances against the bank beside the model:
                // no input schema or rows either.
                if (MLoop.Core.Models.TaskTypes.Canonical(taskType) == "image-anomaly")
                {
                    if (images is null)
                        return Results.BadRequest(new { error = $"'{modelName}' is an image-anomaly model: send images by content — {{\"images\": [base64…]}} or a multipart form." });
                    return await ImageAnomalyPredictions.RespondAsync(
                        images, modelName, answeringId, modelPath, embedderPool, logger, stopwatch, ct);
                }

                // Load experiment data to get InputSchema
                var expData = await experimentStore.LoadAsync(modelName, answeringId, ct);
                var schema = expData?.Config?.InputSchema;

                if (schema == null)
                {
                    logger.LogError("InputSchema not available for '{ModelName}' experiment {ExperimentId}. " +
                        "This model was trained before schema capture was implemented. Please retrain.",
                        modelName, answeringId);
                    return Results.Problem(
                        title: "Schema not available",
                        detail: $"InputSchema not found for model '{modelName}' experiment {answeringId}. " +
                            "This model was trained before schema capture was implemented. Please retrain the model.",
                        statusCode: StatusCodes.Status500InternalServerError
                    );
                }

                // The rows: parsed from JSON, or one per image — written to a scratch file the model's image column reads,
                // and logged by the image's SHA-256 rather than a path that is gone once the request ends.
                Dictionary<string, object>[] rows;
                Dictionary<string, object>[] loggedRows;
                if (images is not null)
                {
                    if (MLoop.Core.Models.TaskTypes.Canonical(taskType) != "image-classification")
                        return Results.BadRequest(new { error = $"Images are predicted by an image-classification or object-detection model; '{modelName}' answers {taskType}. Send its rows as JSON." });
                    foreach (var image in images)
                    {
                        var file = MLoop.Core.Data.ScratchFiles.New("mloop_predict_image_", ".img");
                        imageFiles.Add(file);
                        await File.WriteAllBytesAsync(file, image, ct);
                    }
                    rows = [.. imageFiles.Select(file => new Dictionary<string, object> { [MLoop.Core.Data.ImageDirectoryLoader.ImagePathColumn] = file })];
                    loggedRows = [.. images.Select(image => new Dictionary<string, object> { ["imageSha256"] = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(image)) })];
                }
                else
                {
                    rows = ParseJsonInput(input);
                    loggedRows = rows;
                }

                // Run prediction through shared PredictionService
                var labelColumn = expData?.Config?.LabelColumn;
                var predictionService = new PredictionService(mlContext);

                logger.LogDebug("Running prediction for '{ModelName}' with task '{TaskType}', label '{LabelColumn}'",
                    modelName, taskType, labelColumn);

                // DL tasks need their native runtime loaded before the cache deserializes the model.
                MLoop.Core.Runtime.RuntimeManager.EnsureRuntimeForTask(taskType);

                var transformer = modelCache.GetOrLoad(modelPath);

                // ② regression wave: surface the conformal prediction interval around each Score when the
                // model stored a band at train time (regression only, primary 90% level). Heteroscedastic
                // models also carry a sibling σ-model (residual-model.zip) for per-row widths — load it through
                // the same cache so the band narrows/widens per prediction.
                var interval = RegressionInterval.FromMetrics(taskType, expData?.Metrics);
                ITransformer? residualModel = null;
                if (interval?.IsHeteroscedastic == true)
                {
                    var residualPath = MLoop.Core.Models.ScoringModel.ResidualPathBeside(modelPath);
                    if (File.Exists(residualPath))
                        residualModel = modelCache.GetOrLoad(residualPath);
                }

                var result = predictionService.Predict(rows, schema, transformer, taskType, labelColumn, interval, residualModel);

                logger.LogInformation("Predictions completed for '{ModelName}': {Count} predictions in {ElapsedMs}ms (avg: {AvgMs}ms/prediction)",
                    modelName, result.Rows.Count, stopwatch.ElapsedMilliseconds,
                    stopwatch.ElapsedMilliseconds / Math.Max(1.0, result.Rows.Count));

                // Served predictions are logged like `mloop predict --log`, so logs, feedback and the
                // retraining triggers see the traffic the model actually answers. Each row carries the id it
                // was logged under — the id feedback names. A failure to log never fails the prediction.
                var predictions = result.Rows;
                var warnings = result.Warnings;
                if (configuration.GetValue("Serve:LogPredictions", true) && predictions.Count == rows.Length)
                {
                    try
                    {
                        predictions = [.. await ServedPredictionLog.LogAsync(
                            predictionLogger, modelName, answeringId, loggedRows, predictions, ct)];
                    }
                    catch (Exception ex) when (ex is not OperationCanceledException)
                    {
                        logger.LogWarning(ex, "Predictions for '{ModelName}' were served but not logged", modelName);
                        warnings = [.. warnings ?? [], "These predictions were not logged, so they carry no predictionId: " + ex.Message];
                    }
                }

                return Results.Ok(new
                {
                    modelName,
                    experimentId = answeringId,
                    modelSha256 = ModelFingerprint.Sha256(modelPath),
                    predictedAt = DateTime.UtcNow,
                    task = result.TaskType,
                    count = predictions.Count,
                    predictions,
                    warnings
                });
            }
            catch (ArgumentException ex)
            {
                // Caller error (e.g. rows share no column with the trained schema — the silent-GIGO guard):
                // 400 with the actionable message, not a 500.
                logger.LogWarning(ex, "Prediction rejected for '{ModelName}': invalid input rows", modelName);
                return Results.Problem(
                    title: "Invalid prediction input",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status400BadRequest
                );
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Prediction failed for '{ModelName}' with {Count} input(s) after {ElapsedMs}ms",
                    modelName, predictionCount, stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Prediction failed",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
            finally
            {
                foreach (var file in imageFiles)
                    MLoop.Core.Data.ScratchFiles.Release(file);
            }
        })
        .WithName("Predict")
        .WithTags("Prediction")
        .Accepts<JsonElement>("application/json", "multipart/form-data")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<object>(StatusCodes.Status400BadRequest)
        .Produces<object>(StatusCodes.Status404NotFound)
        .Produces<object>(StatusCodes.Status413PayloadTooLarge)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization();

        app.MapEmbed();
    }

    // The single value a logged prediction records as its answer — what feedback is compared against.
    static Dictionary<string, object>[] ParseJsonInput(JsonElement input)
    {
        var items = new List<JsonElement>();
        if (input.ValueKind == JsonValueKind.Array)
            foreach (var item in input.EnumerateArray()) items.Add(item);
        else
            items.Add(input);

        return items.Select(item =>
        {
            var row = new Dictionary<string, object>();
            foreach (var prop in item.EnumerateObject())
            {
                row[prop.Name] = prop.Value.ValueKind switch
                {
                    JsonValueKind.Number => prop.Value.GetDouble(),
                    JsonValueKind.String => prop.Value.GetString() ?? string.Empty,
                    JsonValueKind.True => true,
                    JsonValueKind.False => false,
                    _ => prop.Value.ToString()
                };
            }
            return row;
        }).ToArray();
    }
}
