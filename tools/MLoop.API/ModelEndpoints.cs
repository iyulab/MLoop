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
/// Model lifecycle endpoints: production info, model and cache listing, promotion, its history, and rollback.
/// </summary>
public static class ModelEndpoints
{
    public static void MapModelEndpoints(this WebApplication app)
    {
        // Model info endpoint (secured)
        app.MapGet("/info", async (
            [FromQuery] string? name,
            IModelRegistry registry,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Resolve(name);

            try
            {
                logger.LogInformation("Retrieving model information for '{ModelName}'", modelName);
                var productionModel = await registry.GetProductionAsync(modelName, ct);

                if (productionModel == null)
                {
                    logger.LogWarning("No production model found for '{ModelName}'", modelName);
                    return Results.NotFound(new { error = $"No production model found for '{modelName}'. Train and promote a model first." });
                }

                var modelPath = registry.GetProductionPath(modelName);
                var modelFile = Path.Combine(modelPath, ExperimentLayout.ModelFileName);
                var metadataFile = Path.Combine(modelPath, ExperimentLayout.MetadataFileName);

                object? metadata = null;
                if (File.Exists(metadataFile))
                {
                    var json = await File.ReadAllTextAsync(metadataFile, ct);
                    metadata = JsonSerializer.Deserialize<Dictionary<string, object>>(json);
                }

                logger.LogInformation("Model information retrieved for '{ModelName}' experiment {ExperimentId} in {ElapsedMs}ms",
                    modelName, productionModel.ExperimentId, stopwatch.ElapsedMilliseconds);

                return Results.Ok(new
                {
                    modelName = modelName,
                    experimentId = productionModel.ExperimentId,
                    promotedAt = productionModel.PromotedAt,
                    metrics = productionModel.Metrics,
                    task = productionModel.Task,
                    bestTrainer = productionModel.BestTrainer,
                    metadata
                });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to retrieve model information for '{ModelName}' after {ElapsedMs}ms",
                    modelName, stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Failed to retrieve model information",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("GetModelInfo")
        .WithTags("Model")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<object>(StatusCodes.Status404NotFound)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization();

        // Model cache stats (admin)
        app.MapGet("/cache/stats", (IModelCache modelCache) =>
        {
            var stats = modelCache.GetStats();
            return Results.Ok(stats);
        })
        .WithName("GetModelCacheStats")
        .WithTags("Cache")
        .Produces<ModelCacheStats>(StatusCodes.Status200OK)
        .RequireAuthorization("Admin");

        // Clear all cached models (admin)
        app.MapPost("/cache/clear", (IModelCache modelCache, ILogger<Program> logger) =>
        {
            modelCache.Clear();
            logger.LogInformation("Model cache cleared by admin request");
            return Results.NoContent();
        })
        .WithName("ClearModelCache")
        .WithTags("Cache")
        .Produces(StatusCodes.Status204NoContent)
        .RequireAuthorization("Admin");

        // List all models endpoint (secured)
        app.MapGet("/models", async (
            [FromQuery] string? name,
            IModelRegistry registry,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = string.IsNullOrWhiteSpace(name) ? null : name.Trim().ToLowerInvariant();

            try
            {
                logger.LogInformation("Listing models (filter: '{ModelName}')", modelName ?? "all");
                var models = await registry.ListAsync(modelName, ct);
                var modelList = models.ToList();

                logger.LogInformation("Retrieved {Count} model(s) in {ElapsedMs}ms", modelList.Count, stopwatch.ElapsedMilliseconds);

                return Results.Ok(new
                {
                    count = modelList.Count,
                    filter = modelName,
                    models = modelList.Select(m => new
                    {
                        modelName = m.ModelName,
                        experimentId = m.ExperimentId,
                        promotedAt = m.PromotedAt,
                        task = m.Task,
                        bestTrainer = m.BestTrainer,
                        metrics = m.Metrics
                    })
                });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to list models after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Failed to list models",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("ListModels")
        .WithTags("Model")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization();

        // Promote experiment to production (secured)
        app.MapPost("/promote", async (
            [FromBody] PromoteRequest request,
            IModelRegistry registry,
            IExperimentStore experimentStore,
            IPromotionManager promotionManager,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Resolve(request.Name);
            var experimentId = request.ExperimentId;

            try
            {
                logger.LogInformation("Promoting experiment '{ExperimentId}' for model '{ModelName}'",
                    experimentId, modelName);

                // Verify experiment exists
                if (!experimentStore.ExperimentExists(modelName, experimentId))
                {
                    return Results.NotFound(new { error = $"Experiment '{experimentId}' not found for model '{modelName}'." });
                }

                // Load and validate experiment
                var experiment = await experimentStore.LoadAsync(modelName, experimentId, ct);

                if (!experiment.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
                {
                    return Results.BadRequest(new
                    {
                        error = $"Cannot promote experiment with status '{experiment.Status}'. Only completed experiments can be promoted."
                    });
                }

                // The same gate `mloop promote` and training apply; `force` is the deliberate override.
                var refusal = MLoop.CLI.Infrastructure.FileSystem.ModelRegistry.QualityGateRefusal(experiment, experiment.Config?.Metric);
                if (refusal != null && !request.Force)
                {
                    return Results.Json(new
                    {
                        error = $"'{experimentId}' does not pass the quality gate: {refusal}. Send \"force\": true to promote it anyway."
                    }, statusCode: StatusCodes.Status422UnprocessableEntity);
                }
                if (refusal != null)
                {
                    logger.LogWarning("Promoting '{ExperimentId}' despite the quality gate (force): {Refusal}", experimentId, refusal);
                }

                // The project's own gate policy (mloop.yaml `gate:`), judged on its evaluation set; `force` overrides it too.
                var policyRefusal = await registry.GatePolicyRefusalAsync(modelName, experimentId, ct);
                if (policyRefusal != null && !request.Force)
                {
                    return Results.Json(new
                    {
                        error = $"'{experimentId}' does not pass the gate policy: {policyRefusal}. Send \"force\": true to promote it anyway."
                    }, statusCode: StatusCodes.Status422UnprocessableEntity);
                }
                if (policyRefusal != null)
                {
                    logger.LogWarning("Promoting '{ExperimentId}' despite the gate policy (force): {Refusal}", experimentId, policyRefusal);
                }
                var qualityWarning = MLoop.CLI.Infrastructure.FileSystem.ModelRegistry.QualityGateWarning(experiment);

                // Check current production for comparison
                var currentProduction = await registry.GetProductionAsync(modelName, ct);
                var previousExpId = currentProduction?.ExperimentId;

                // Backup current production if exists
                if (currentProduction != null && request.CreateBackup != false)
                {
                    var backupPath = await promotionManager.BackupProductionAsync(modelName, ct);
                    if (backupPath != null)
                    {
                        logger.LogInformation("Backed up current production to {BackupPath}", backupPath);
                    }
                }

                // Promote — the registry records it in the promotion history, gate outcome included.
                await registry.PromoteAsync(modelName, experimentId, PromotionActions.Promote, policyRefusal, ct);

                logger.LogInformation("Promoted experiment '{ExperimentId}' to production for model '{ModelName}' in {ElapsedMs}ms",
                    experimentId, modelName, stopwatch.ElapsedMilliseconds);

                return Results.Ok(new
                {
                    modelName,
                    experimentId,
                    previousExperimentId = previousExpId,
                    promotedAt = DateTime.UtcNow,
                    task = experiment.Task,
                    metrics = experiment.Metrics,
                    bestTrainer = experiment.Result?.BestTrainer,
                    qualityWarning
                });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Promotion failed for '{ModelName}/{ExperimentId}' after {ElapsedMs}ms",
                    modelName, experimentId, stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Promotion failed",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("PromoteExperiment")
        .WithTags("Model")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<object>(StatusCodes.Status400BadRequest)
        .Produces<object>(StatusCodes.Status404NotFound)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization("Admin");

        // Rollback (admin): production back to the experiment it held before — same target rule as `mloop rollback`
        // Promotion history: every change of the production slot, oldest first, with the gate's outcome.
        app.MapGet("/promotions", async (
            [FromQuery] string? name,
            IPromotionManager promotionManager,
            CancellationToken ct) =>
        {
            var modelName = MLoop.Core.Storage.ModelName.Resolve(name);
            var history = await promotionManager.GetHistoryAsync(modelName, ct);
            return Results.Ok(new
            {
                modelName,
                promotions = history.Select(r => new
                {
                    experimentId = r.ExperimentId,
                    previousExperimentId = r.PreviousExperimentId,
                    action = r.Action,
                    timestamp = r.Timestamp.ToUniversalTime(),
                    gate = r.Gate,
                    gateDetail = r.GateDetail,
                    metric = r.Metric,
                    metricValue = r.MetricValue,
                }),
            });
        })
        .WithName("PromotionHistory")
        .WithTags("Model")
        .Produces<object>(StatusCodes.Status200OK)
        .RequireAuthorization();

        app.MapPost("/rollback", async (
            [FromBody] RollbackRequest request,
            IModelRegistry registry,
            IExperimentStore experimentStore,
            IPromotionManager promotionManager,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var modelName = MLoop.Core.Storage.ModelName.Resolve(request.Name);
            var current = (await registry.GetProductionAsync(modelName, ct))?.ExperimentId;
            var history = await promotionManager.GetHistoryAsync(modelName, ct);
            var (target, error) = MLoop.Ops.Interfaces.RollbackTarget.Resolve(history, current, request.To?.Trim());
            if (target is null)
                return Results.Json(new { error }, statusCode: StatusCodes.Status409Conflict);

            if (!experimentStore.ExperimentExists(modelName, target))
                return Results.Json(new { error = $"'{target}' is no longer in staging, so it cannot be restored from there." },
                    statusCode: StatusCodes.Status409Conflict);

            var backupPath = await promotionManager.BackupProductionAsync(modelName, ct);
            await registry.PromoteAsync(modelName, target, PromotionActions.Rollback, cancellationToken: ct);
            logger.LogInformation("Rolled back '{ModelName}' from {Previous} to {Target}", modelName, current, target);

            return Results.Ok(new { modelName, experimentId = target, previousExperimentId = current, backupPath });
        })
        .WithName("RollbackProduction")
        .WithTags("Model")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<object>(StatusCodes.Status409Conflict)
        .RequireAuthorization("Admin");
    }
}

public record PromoteRequest(
    string ExperimentId,
    string? Name = null,
    bool? CreateBackup = true,
    bool Force = false);

public record RollbackRequest(
    string? Name = null,
    string? To = null);
