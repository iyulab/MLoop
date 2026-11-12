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
/// Monitoring endpoints: prediction logs, feedback, retraining samples, input drift, and the retraining trigger.
/// </summary>
public static class MonitoringEndpoints
{
    public static void MapMonitoringEndpoints(this WebApplication app)
    {
        // Prediction logs endpoint (secured)
        app.MapGet("/logs", async Task<Results<Ok<LogListResponse>, BadRequest<ErrorResponse>, ProblemHttpResult>> (
            [FromQuery] string? name,
            [FromQuery] string? from,
            [FromQuery] string? to,
            [FromQuery] int? limit,
            IPredictionLogger predictionLogger,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Filter(name);

            try
            {
                logger.LogInformation("Retrieving prediction logs (filter: '{ModelName}')", modelName ?? "all");

                if (!TryParseDateRange(from, to, out var fromDate, out var toDate, out var dateError))
                    return TypedResults.BadRequest(new ErrorResponse(dateError));
                var maxEntries = limit ?? 1000;

                var logs = await predictionLogger.GetLogsAsync(modelName, fromDate, toDate, maxEntries, ct);

                logger.LogInformation("Retrieved {Count} log entries in {ElapsedMs}ms",
                    logs.Count, stopwatch.ElapsedMilliseconds);

                return TypedResults.Ok(new LogListResponse(
                    logs.Count,
                    modelName,
                    [.. logs.Select(l => new LogEntryView(l.Id, l.ModelName, l.ExperimentId, l.Input, l.Output, l.Confidence, l.Timestamp))]));
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to retrieve logs after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Failed to retrieve prediction logs",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("GetPredictionLogs")
        .WithTags("Monitoring")
        .RequireAuthorization();

        // Get feedback endpoint (secured)
        app.MapGet("/feedback", async Task<Results<Ok<FeedbackListResponse>, BadRequest<ErrorResponse>, ProblemHttpResult>> (
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
                    return TypedResults.BadRequest(new ErrorResponse(dateError));
                var maxEntries = limit ?? 1000;

                var feedback = await feedbackCollector.GetFeedbackAsync(modelName, fromDate, toDate, maxEntries, ct);

                logger.LogInformation("Retrieved {Count} feedback entries in {ElapsedMs}ms",
                    feedback.Count, stopwatch.ElapsedMilliseconds);

                return TypedResults.Ok(new FeedbackListResponse(
                    feedback.Count,
                    modelName,
                    [.. feedback.Select(f => new FeedbackEntryView(f.PredictionId, f.ModelName, f.PredictedValue, f.ActualValue, f.Source, f.Weight, f.Timestamp))]));
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to retrieve feedback after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Failed to retrieve feedback",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("GetFeedback")
        .WithTags("Monitoring")
        .RequireAuthorization();

        // A retraining set from logged predictions and their feedback — `mloop sample create --name` over HTTP.
        // The file it writes is what POST /train takes in additionalDataFiles.
        app.MapPost("/samples", async Task<Results<Ok<SampleResponse>, BadRequest<ErrorResponse>, NotFound<ErrorResponse>>> (
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
                return TypedResults.BadRequest(new ErrorResponse($"Unknown strategy '{request.Strategy}'. {FeedbackSamples.StrategyGuidance}."));
            if (request.Rows is <= 0)
                return TypedResults.BadRequest(new ErrorResponse("rows is a positive number of rows."));

            var columns = await FeedbackSamples.ColumnsAsync(fileSystem, discovery, modelName, ct);
            if (columns is null)
                return TypedResults.BadRequest(new ErrorResponse(FeedbackSamples.NoLabelError(modelName)));

            var projectRoot = projectRootPath.Value;
            var outputPath = FeedbackSamples.DefaultPath(projectRoot, modelName);
            try
            {
                var result = await new FileDataSampler(projectRoot).SampleAsync(
                    modelName, request.Rows ?? 1000, strategy.Value, outputPath, columns, ct);
                logger.LogInformation("Sampled {Count} rows ({Labeled} with feedback) for model '{ModelName}'",
                    result.SampledCount, result.LabeledCount, modelName);
                return TypedResults.Ok(new SampleResponse(
                    modelName,
                    Path.GetRelativePath(projectRoot, result.OutputPath),
                    result.SampledCount,
                    result.LabeledCount,
                    result.TotalAvailable,
                    result.StrategyUsed.ToString(),
                    result.LabelColumn,
                    result.WeightColumn));
            }
            catch (InvalidOperationException ex)
            {
                // No logged predictions for the model: nothing to sample, said as the CLI says it.
                return TypedResults.NotFound(new ErrorResponse(ex.Message));
            }
        })
        .WithName("CreateSample")
        .WithTags("Monitoring")
        .RequireAuthorization("Admin");

        // Submit feedback endpoint (secured)
        app.MapPost("/feedback", async Task<Results<Ok<FeedbackRecordedResponse>, ProblemHttpResult>> (
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

                return TypedResults.Ok(new FeedbackRecordedResponse(request.PredictionId, Recorded: true, DateTime.UtcNow));
            }
            catch (ArgumentException ex)
            {
                return TypedResults.Problem(title: "Invalid feedback", detail: ex.Message,
                    statusCode: StatusCodes.Status400BadRequest);
            }
            catch (PredictionNotFoundException ex)
            {
                return TypedResults.Problem(title: "Prediction not found", detail: ex.Message,
                    statusCode: StatusCodes.Status404NotFound);
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to record feedback after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Failed to record feedback",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("SubmitFeedback")
        .WithTags("Monitoring")
        .RequireAuthorization();

        // Input drift: the production model's recent inputs against its training profile (secured)
        app.MapGet("/drift", async Task<Results<Ok<DriftResponse>, ProblemHttpResult>> (
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
                return TypedResults.Problem(
                    title: "Invalid window",
                    detail: $"rows must be at least {MLoop.Ops.Services.DriftEvaluator.MinimumRows}.",
                    statusCode: StatusCodes.Status400BadRequest);
            var check = await MLoop.Ops.Services.DriftEvaluator.CheckAsync(root.Value, predictionLogs, modelName, window, sliceBy, ct);
            return TypedResults.Ok(new DriftResponse(
                modelName,
                check.Status.ToString(),
                check.ExperimentId,
                check.Message,
                check.Report is { } r ? MLoop.Ops.Services.DriftEvaluator.Signal(r) : null,
                sliceBy,
                check.Slices is null ? null : [.. check.Slices.Select(sl => new SliceDriftView(sl.Value, MLoop.Ops.Services.DriftEvaluator.Signal(sl.Report)))]));
        })
        .WithName("Drift")
        .WithTags("Monitoring")
        .RequireAuthorization();

        // Retraining trigger evaluation endpoint (secured)
        app.MapGet("/trigger", async Task<Results<Ok<TriggerResponse>, ProblemHttpResult>> (
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

                return TypedResults.Ok(new TriggerResponse(
                    modelName,
                    evaluation.ShouldRetrain,
                    evaluation.RecommendedAction,
                    evaluation.EvaluatedAt,
                    [.. evaluation.ConditionResults.Select(cr => new TriggerConditionView(
                        cr.Condition.Type.ToString(),
                        cr.Condition.Name,
                        cr.Condition.Threshold,
                        cr.CurrentValue,
                        cr.IsMet,
                        cr.Details,
                        cr.Drift is { } d ? MLoop.Ops.Services.DriftEvaluator.Signal(d) : null))]));
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Trigger evaluation failed after {ElapsedMs}ms", stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Trigger evaluation failed",
                    detail: ex.Message,
                    statusCode: StatusCodes.Status500InternalServerError
                );
            }
        })
        .WithName("EvaluateTrigger")
        .WithTags("Monitoring")
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
[System.Text.Json.Serialization.JsonUnmappedMemberHandling(System.Text.Json.Serialization.JsonUnmappedMemberHandling.Disallow)]
public record SampleRequest(string? Name = null, int? Rows = null, string? Strategy = null);

[System.Text.Json.Serialization.JsonUnmappedMemberHandling(System.Text.Json.Serialization.JsonUnmappedMemberHandling.Disallow)]
public record FeedbackRequest(
    string PredictionId,
    object ActualValue,
    string? Source = null,
    double? Weight = null);
