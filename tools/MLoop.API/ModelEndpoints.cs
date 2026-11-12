using System.Diagnostics;
using System.Reflection;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Http.HttpResults;
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
    /// <summary>
    /// Who is calling and through which door, for the promotion history: the <c>sub</c> of the bearer token (the same claim the request
    /// limit counts a caller by), <c>null</c> when the token carries none.
    /// </summary>
    internal static PromotionOrigin CallerOrigin(HttpContext http)
        => new(PromotionVia.Http, http.User.Identity?.IsAuthenticated == true ? http.User.FindFirst("sub")?.Value : null);

    public static void MapModelEndpoints(this WebApplication app)
    {
        // Model info endpoint (secured)
        app.MapGet("/info", async Task<Results<Ok<ModelInfoResponse>, NotFound<ErrorResponse>, ProblemHttpResult>> (
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
                    return TypedResults.NotFound(new ErrorResponse($"No production model found for '{modelName}'. Train and promote a model first."));
                }

                var modelPath = registry.GetProductionPath(modelName);
                var modelFile = Path.Combine(modelPath, ExperimentLayout.ModelFileName);
                var metadataFile = Path.Combine(modelPath, ExperimentLayout.MetadataFileName);

                Dictionary<string, object>? metadata = null;
                if (File.Exists(metadataFile))
                {
                    var json = await File.ReadAllTextAsync(metadataFile, ct);
                    metadata = JsonSerializer.Deserialize<Dictionary<string, object>>(json);
                }

                logger.LogInformation("Model information retrieved for '{ModelName}' experiment {ExperimentId} in {ElapsedMs}ms",
                    modelName, productionModel.ExperimentId, stopwatch.ElapsedMilliseconds);

                return TypedResults.Ok(new ModelInfoResponse(
                    modelName,
                    productionModel.ExperimentId,
                    productionModel.PromotedAt,
                    productionModel.Metrics,
                    productionModel.Task,
                    productionModel.BestTrainer,
                    metadata));
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to retrieve model information for '{ModelName}' after {ElapsedMs}ms",
                    modelName, stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Failed to retrieve model information",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("GetModelInfo")
        .WithTags("Model")
        .RequireAuthorization();

        // Model cache stats (admin)
        app.MapGet("/cache/stats", (IModelCache modelCache) =>
        {
            var stats = modelCache.GetStats();
            return TypedResults.Ok(stats);
        })
        .WithName("GetModelCacheStats")
        .WithTags("Cache")
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
        app.MapGet("/models", async Task<Results<Ok<ModelListResponse>, ProblemHttpResult>> (
            [FromQuery] string? name,
            IModelRegistry registry,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Filter(name);

            try
            {
                logger.LogInformation("Listing models (filter: '{ModelName}')", modelName ?? "all");
                var models = await registry.ListAsync(modelName, ct);
                var modelList = models.ToList();

                logger.LogInformation("Retrieved {Count} model(s) in {ElapsedMs}ms", modelList.Count, stopwatch.ElapsedMilliseconds);

                return TypedResults.Ok(new ModelListResponse(
                    modelList.Count,
                    modelName,
                    [.. modelList.Select(m => new ModelListItem(m.ModelName, m.ExperimentId, m.PromotedAt, m.Task, m.BestTrainer, m.Metrics))]));
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to list models after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Failed to list models",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("ListModels")
        .WithTags("Model")
        .RequireAuthorization();

        // Promote experiment to production (secured)
        app.MapPost("/promote", async Task<Results<Ok<PromoteResponse>, Ok<PromoteUnchangedResponse>, BadRequest<ErrorResponse>, NotFound<ErrorResponse>, UnprocessableEntity<ErrorResponse>, ProblemHttpResult>> (
            [FromBody] PromoteRequest request,
            HttpContext http,
            IModelRegistry registry,
            IExperimentStore experimentStore,
            IPromotionManager promotionManager,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var origin = CallerOrigin(http);
            var modelName = MLoop.Core.Storage.ModelName.Resolve(request.Name);
            if (string.IsNullOrWhiteSpace(request.ExperimentId))
                return TypedResults.BadRequest(new ErrorResponse("experimentId is required: the experiment to promote (see GET /experiments)."));
            var experimentId = request.ExperimentId.Trim();

            try
            {
                logger.LogInformation("Promoting experiment '{ExperimentId}' for model '{ModelName}' (by {By})",
                    experimentId, modelName, origin.By ?? "unnamed");

                // Verify experiment exists
                if (!experimentStore.ExperimentExists(modelName, experimentId))
                {
                    return TypedResults.NotFound(new ErrorResponse($"Experiment '{experimentId}' not found for model '{modelName}'."));
                }

                // Load and validate experiment
                var experiment = await experimentStore.LoadAsync(modelName, experimentId, ct);

                if (!experiment.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
                {
                    return TypedResults.BadRequest(new ErrorResponse(
                        $"Cannot promote experiment with status '{experiment.Status}'. Only completed experiments can be promoted."));
                }

                // Promoting what is already in production changes nothing — as `mloop promote` says: no backup of the model onto itself,
                // no record in the history, and a later rollback finds what production held before. `force` promotes it again.
                if (!request.Force
                    && (await registry.GetProductionAsync(modelName, ct))?.ExperimentId is { } inProduction
                    && inProduction.Equals(experimentId, StringComparison.OrdinalIgnoreCase))
                {
                    return TypedResults.Ok(new PromoteUnchangedResponse(
                        modelName,
                        experimentId,
                        inProduction,
                        Unchanged: true,
                        experiment.Task,
                        experiment.Metrics,
                        experiment.Result?.BestTrainer));
                }

                // The same gate `mloop promote` and training apply; `force` is the deliberate override.
                var refusal = MLoop.CLI.Infrastructure.FileSystem.ModelRegistry.QualityGateRefusal(experiment, experiment.Config?.Metric);
                if (refusal != null && !request.Force)
                {
                    return TypedResults.UnprocessableEntity(new ErrorResponse(
                        $"'{experimentId}' does not pass the quality gate: {refusal}. Send \"force\": true to promote it anyway."));
                }
                if (refusal != null)
                {
                    logger.LogWarning("Promoting '{ExperimentId}' despite the quality gate (force): {Refusal}", experimentId, refusal);
                }

                // The project's own gate policy (mloop.yaml `gate:`), judged on its evaluation set; `force` overrides it too.
                var policyRefusal = await registry.GatePolicyRefusalAsync(modelName, experimentId, ct);
                if (policyRefusal != null && !request.Force)
                {
                    return TypedResults.UnprocessableEntity(new ErrorResponse(
                        $"'{experimentId}' does not pass the gate policy: {policyRefusal}. Send \"force\": true to promote it anyway."));
                }
                if (policyRefusal != null)
                {
                    logger.LogWarning("Promoting '{ExperimentId}' despite the gate policy (force): {Refusal}", experimentId, policyRefusal);
                }
                var qualityWarning = MLoop.CLI.Infrastructure.FileSystem.ModelRegistry.QualityGateWarning(experiment);

                // Check current production for comparison
                var currentProduction = await registry.GetProductionAsync(modelName, ct);
                var previousExpId = currentProduction?.ExperimentId;

                // What training would have said about keeping this model out of production: the comparison with the model it
                // replaces (judged before the pointer moves). The gate and the gate policy were answered above.
                string? notBetterThanProduction = currentProduction != null && refusal == null && policyRefusal == null
                    && !currentProduction.ExperimentId.Equals(experimentId, StringComparison.OrdinalIgnoreCase)
                    ? await registry.NotBeatingProductionReasonAsync(modelName, experimentId, experiment.Config?.Metric ?? string.Empty, ct)
                    : null;

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
                await registry.PromoteAsync(modelName, experimentId, PromotionActions.Promote, policyRefusal, origin, ct);

                logger.LogInformation("Promoted experiment '{ExperimentId}' to production for model '{ModelName}' in {ElapsedMs}ms",
                    experimentId, modelName, stopwatch.ElapsedMilliseconds);

                return TypedResults.Ok(new PromoteResponse(
                    modelName,
                    experimentId,
                    previousExpId,
                    DateTime.UtcNow,
                    experiment.Task,
                    experiment.Metrics,
                    experiment.Result?.BestTrainer,
                    qualityWarning,
                    notBetterThanProduction));
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Promotion failed for '{ModelName}/{ExperimentId}' after {ElapsedMs}ms",
                    modelName, experimentId, stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Promotion failed",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("PromoteExperiment")
        .WithTags("Model")
        .RequireAuthorization("Admin");

        // Rollback (admin): production back to the experiment it held before — same target rule as `mloop rollback`
        // Promotion history: every change of the production slot, oldest first, with the gate's outcome.
        app.MapGet("/promotions", async Task<Ok<PromotionHistoryResponse>> (
            [FromQuery] string? name,
            IPromotionManager promotionManager,
            CancellationToken ct) =>
        {
            var modelName = MLoop.Core.Storage.ModelName.Resolve(name);
            var history = await promotionManager.GetHistoryAsync(modelName, ct);
            return TypedResults.Ok(new PromotionHistoryResponse(modelName, PromotionHistoryView.Rows(history)));
        })
        .WithName("PromotionHistory")
        .WithTags("Model")
        .RequireAuthorization();

        app.MapPost("/rollback", async Task<Results<Ok<RollbackResponse>, Conflict<ErrorResponse>>> (
            // Both fields are optional (the CLI's `mloop rollback` takes neither), so a call with no body at all means «the previous one».
            [FromBody(EmptyBodyBehavior = Microsoft.AspNetCore.Mvc.ModelBinding.EmptyBodyBehavior.Allow)] RollbackRequest? request,
            HttpContext http,
            IModelRegistry registry,
            IExperimentStore experimentStore,
            IPromotionManager promotionManager,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var modelName = MLoop.Core.Storage.ModelName.Resolve(request?.Name);
            var current = (await registry.GetProductionAsync(modelName, ct))?.ExperimentId;
            var history = await promotionManager.GetHistoryAsync(modelName, ct);
            var (target, error) = MLoop.Ops.Interfaces.RollbackTarget.Resolve(history, current, request?.To?.Trim());
            if (target is null)
                return TypedResults.Conflict(new ErrorResponse(error));

            if (!experimentStore.ExperimentExists(modelName, target))
                return TypedResults.Conflict(new ErrorResponse($"'{target}' is no longer in staging, so it cannot be restored from there."));

            var backupPath = await promotionManager.BackupProductionAsync(modelName, ct);
            var origin = CallerOrigin(http);
            await registry.PromoteAsync(modelName, target, PromotionActions.Rollback, origin: origin, cancellationToken: ct);
            logger.LogInformation("Rolled back '{ModelName}' from {Previous} to {Target} (by {By})", modelName, current, target, origin.By ?? "unnamed");

            return TypedResults.Ok(new RollbackResponse(modelName, target, current, backupPath));
        })
        .WithName("RollbackProduction")
        .WithTags("Model")
        .RequireAuthorization("Admin");
    }
}

public record PromoteRequest(
    string? ExperimentId,
    string? Name = null,
    bool? CreateBackup = true,
    bool Force = false);

public record RollbackRequest(
    string? Name = null,
    string? To = null);
