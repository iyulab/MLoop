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
/// Monitoring endpoints: prediction logs, feedback, retraining samples, input drift, and the retraining trigger.
/// </summary>
public static class MonitoringEndpoints
{
    public static void MapMonitoringEndpoints(this WebApplication app)
    {
        // Prediction logs endpoint (secured)
        app.MapGet("/logs", async (
            [FromQuery] string? name,
            [FromQuery] string? from,
            [FromQuery] string? to,
            [FromQuery] int? limit,
            IPredictionLogger predictionLogger,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = string.IsNullOrWhiteSpace(name) ? null : name.Trim().ToLowerInvariant();

            try
            {
                logger.LogInformation("Retrieving prediction logs (filter: '{ModelName}')", modelName ?? "all");

                if (!TryParseDateRange(from, to, out var fromDate, out var toDate, out var dateError))
                    return Results.BadRequest(new { error = dateError });
                var maxEntries = limit ?? 1000;

                var logs = await predictionLogger.GetLogsAsync(modelName, fromDate, toDate, maxEntries, ct);

                logger.LogInformation("Retrieved {Count} log entries in {ElapsedMs}ms",
                    logs.Count, stopwatch.ElapsedMilliseconds);

                return Results.Ok(new
                {
                    count = logs.Count,
                    filter = modelName,
                    logs = logs.Select(l => new
                    {
                        id = l.Id,
                        modelName = l.ModelName,
                        experimentId = l.ExperimentId,
                        input = l.Input,
                        output = l.Output,
                        confidence = l.Confidence,
                        timestamp = l.Timestamp
                    })
                });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to retrieve logs after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Failed to retrieve prediction logs",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("GetPredictionLogs")
        .WithTags("Monitoring")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization();

        // Get feedback endpoint (secured)
        app.MapGet("/feedback", async (
            [FromQuery] string? name,
            [FromQuery] string? from,
            [FromQuery] string? to,
            [FromQuery] int? limit,
            IFeedbackCollector feedbackCollector,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Resolve(name);

            try
            {
                logger.LogInformation("Retrieving feedback for model '{ModelName}'", modelName);

                if (!TryParseDateRange(from, to, out var fromDate, out var toDate, out var dateError))
                    return Results.BadRequest(new { error = dateError });
                var maxEntries = limit ?? 1000;

                var feedback = await feedbackCollector.GetFeedbackAsync(modelName, fromDate, toDate, maxEntries, ct);

                logger.LogInformation("Retrieved {Count} feedback entries in {ElapsedMs}ms",
                    feedback.Count, stopwatch.ElapsedMilliseconds);

                return Results.Ok(new
                {
                    count = feedback.Count,
                    modelName,
                    feedback = feedback.Select(f => new
                    {
                        predictionId = f.PredictionId,
                        modelName = f.ModelName,
                        predictedValue = f.PredictedValue,
                        actualValue = f.ActualValue,
                        source = f.Source,
                        weight = f.Weight,
                        timestamp = f.Timestamp
                    })
                });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to retrieve feedback after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Failed to retrieve feedback",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("GetFeedback")
        .WithTags("Monitoring")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization();

        // A retraining set from logged predictions and their feedback — `mloop sample create --name` over HTTP.
        // The file it writes is what POST /train takes in additionalDataFiles.
        app.MapPost("/samples", async (
            [FromBody] SampleRequest request,
            ProjectRootPath projectRootPath,
            IFileSystemManager fileSystem,
            IProjectDiscovery discovery,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var modelName = MLoop.Core.Storage.ModelName.Resolve(request.Name);
            var strategy = FeedbackSamples.ParseStrategy(request.Strategy ?? "random");
            if (strategy is null)
                return Results.BadRequest(new { error = $"Unknown strategy '{request.Strategy}'. {FeedbackSamples.StrategyGuidance}." });
            if (request.Rows is <= 0)
                return Results.BadRequest(new { error = "rows is a positive number of rows." });

            var columns = await FeedbackSamples.ColumnsAsync(fileSystem, discovery, modelName, ct);
            if (columns is null)
                return Results.BadRequest(new { error = FeedbackSamples.NoLabelError(modelName) });

            var projectRoot = projectRootPath.Value;
            var outputPath = FeedbackSamples.DefaultPath(projectRoot, modelName);
            try
            {
                var result = await new FileDataSampler(projectRoot).SampleAsync(
                    modelName, request.Rows ?? 1000, strategy.Value, outputPath, columns, ct);
                logger.LogInformation("Sampled {Count} rows ({Labeled} with feedback) for model '{ModelName}'",
                    result.SampledCount, result.LabeledCount, modelName);
                return Results.Ok(new
                {
                    modelName,
                    path = Path.GetRelativePath(projectRoot, result.OutputPath),
                    sampledCount = result.SampledCount,
                    labeledCount = result.LabeledCount,
                    totalAvailable = result.TotalAvailable,
                    strategy = result.StrategyUsed.ToString(),
                    labelColumn = result.LabelColumn,
                    weightColumn = result.WeightColumn,
                });
            }
            catch (InvalidOperationException ex)
            {
                // No logged predictions for the model: nothing to sample, said as the CLI says it.
                return Results.NotFound(new { error = ex.Message });
            }
        })
        .WithName("CreateSample")
        .WithTags("Monitoring")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<object>(StatusCodes.Status400BadRequest)
        .Produces<object>(StatusCodes.Status404NotFound)
        .RequireAuthorization("Admin");

        // Submit feedback endpoint (secured)
        app.MapPost("/feedback", async (
            [FromBody] FeedbackRequest request,
            IFeedbackCollector feedbackCollector,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();

            try
            {
                logger.LogInformation("Recording feedback for prediction '{PredictionId}'", request.PredictionId);

                await feedbackCollector.RecordFeedbackAsync(
                    request.PredictionId, request.ActualValue, request.Source, request.Weight, ct);

                logger.LogInformation("Feedback recorded in {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);

                return Results.Ok(new
                {
                    predictionId = request.PredictionId,
                    recorded = true,
                    timestamp = DateTime.UtcNow
                });
            }
            catch (ArgumentException ex)
            {
                return Results.Problem(title: "Invalid feedback", detail: ex.Message,
                    statusCode: StatusCodes.Status400BadRequest);
            }
            catch (PredictionNotFoundException ex)
            {
                return Results.Problem(title: "Prediction not found", detail: ex.Message,
                    statusCode: StatusCodes.Status404NotFound);
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to record feedback after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Failed to record feedback",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("SubmitFeedback")
        .WithTags("Monitoring")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<ProblemDetails>(StatusCodes.Status400BadRequest)
        .Produces<ProblemDetails>(StatusCodes.Status404NotFound)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization();

        // Input drift: the production model's recent inputs against its training profile (secured)
        app.MapGet("/drift", async (
            [FromQuery] string? name,
            [FromQuery] int? rows,
            [FromQuery] string? sliceBy,
            [FromServices] ProjectRootPath root,
            IPredictionLogger predictionLogs,
            CancellationToken ct) =>
        {
            var modelName = MLoop.Core.Storage.ModelName.Resolve(name);
            var window = rows ?? MLoop.Ops.Services.DriftEvaluator.DefaultWindowRows;
            if (window < MLoop.Ops.Services.DriftEvaluator.MinimumRows)
                return Results.Problem(
                    title: "Invalid window",
                    detail: $"rows must be at least {MLoop.Ops.Services.DriftEvaluator.MinimumRows}.",
                    statusCode: StatusCodes.Status400BadRequest);
            var check = await MLoop.Ops.Services.DriftEvaluator.CheckAsync(root.Value, predictionLogs, modelName, window, sliceBy, ct);
            return Results.Ok(new
            {
                modelName,
                status = check.Status.ToString(),
                experimentId = check.ExperimentId,
                message = check.Message,
                drift = check.Report is { } r ? MLoop.Ops.Services.DriftEvaluator.Signal(r) : null,
                sliceBy,
                slices = check.Slices?.Select(sl => new { value = sl.Value, drift = MLoop.Ops.Services.DriftEvaluator.Signal(sl.Report) }),
            });
        })
        .WithName("Drift")
        .WithTags("Monitoring")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<ProblemDetails>(StatusCodes.Status400BadRequest)
        .RequireAuthorization();

        // Retraining trigger evaluation endpoint (secured)
        app.MapGet("/trigger", async (
            [FromQuery] string? name,
            IRetrainingTrigger trigger,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Resolve(name);

            try
            {
                logger.LogInformation("Evaluating retraining triggers for model '{ModelName}'", modelName);

                var conditions = await trigger.GetDefaultConditionsAsync(modelName, ct);
                var evaluation = await trigger.EvaluateAsync(modelName, conditions, ct);

                logger.LogInformation("Trigger evaluation completed in {ElapsedMs}ms: shouldRetrain={ShouldRetrain}",
                    stopwatch.ElapsedMilliseconds, evaluation.ShouldRetrain);

                return Results.Ok(new
                {
                    modelName,
                    shouldRetrain = evaluation.ShouldRetrain,
                    recommendedAction = evaluation.RecommendedAction,
                    evaluatedAt = evaluation.EvaluatedAt,
                    conditions = evaluation.ConditionResults.Select(cr => new
                    {
                        type = cr.Condition.Type.ToString(),
                        name = cr.Condition.Name,
                        threshold = cr.Condition.Threshold,
                        currentValue = cr.CurrentValue,
                        isMet = cr.IsMet,
                        details = cr.Details,
                        drift = cr.Drift is { } d ? MLoop.Ops.Services.DriftEvaluator.Signal(d) : null
                    })
                });
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Trigger evaluation failed after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return Results.Problem(
                    title: "Trigger evaluation failed",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("EvaluateTrigger")
        .WithTags("Monitoring")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<ProblemDetails>(StatusCodes.Status500InternalServerError)
        .RequireAuthorization();
    }

    static bool TryParseDateRange(string? from, string? to,
        out DateTimeOffset? fromDate, out DateTimeOffset? toDate, out string? error)
    {
        fromDate = null;
        toDate = null;
        error = null;

        if (from != null)
        {
            if (!DateTimeOffset.TryParse(from, System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None, out var parsedFrom))
            {
                error = $"Invalid 'from' date format: {from}";
                return false;
            }
            fromDate = parsedFrom;
        }

        if (to != null)
        {
            if (!DateTimeOffset.TryParse(to, System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None, out var parsedTo))
            {
                error = $"Invalid 'to' date format: {to}";
                return false;
            }
            toDate = parsedTo;
        }

        return true;
    }
}

/// <summary>A <c>POST /samples</c> body — the options of <c>mloop sample create --name</c>.</summary>
public record SampleRequest(string? Name = null, int? Rows = null, string? Strategy = null);

public record FeedbackRequest(
    string PredictionId,
    object ActualValue,
    string? Source = null,
    double? Weight = null);
