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
/// Training endpoints: start an asynchronous job, list jobs, read one job's status.
/// </summary>
public static class TrainingEndpoints
{
    public static void MapTrainingEndpoints(this WebApplication app)
    {
        // Start training job (async, secured)
        app.MapPost("/train", async (
            [FromBody] TrainingJobRequest request,
            TrainingJobStore jobStore,
            TrainingJobRunner jobRunner,
            ProjectRootPath projectRootPath,
            ConfigLoader configLoader,
            ILogger<Program> logger) =>
        {
            // Refuse a metric the task cannot optimize now, not minutes later inside the job — the job
            // would otherwise fail (or, before this, quietly optimize the task default).
            var metricRejection = MLoop.Core.Evaluation.MetricNames.Rejection(request.Task, request.Metric);
            if (metricRejection != null)
                return Results.BadRequest(new { error = metricRejection });

            // Validate every data file exists and is within project root. A relative path is the project's — the
            // form POST /samples answers with — not the server process's working directory.
            string InProject(string file) => ProjectPaths.Resolve(file, projectRootPath.Value);
            foreach (var file in new[] { request.DataFile }.Concat(request.AdditionalDataFiles ?? []))
            {
                var resolvedDataFile = InProject(file);
                if (!ProjectPaths.IsWithin(resolvedDataFile, projectRootPath.Value))
                    return Results.BadRequest(new { error = $"Data file must be within the project directory: {file}" });
                if (!File.Exists(resolvedDataFile))
                    return Results.BadRequest(new { error = $"Data file not found: {file}" });
            }
            request = request with
            {
                DataFile = InProject(request.DataFile),
                AdditionalDataFiles = request.AdditionalDataFiles?.Select(InProject).ToList(),
            };

            // Refused now, not inside the job: what the project's mloop.yaml asks for that this run could
            // not honour (a weight on a task it does not reach, prep steps the API does not run).
            var (trainDefinition, definitionError) = await TrainingDefinition.ResolveAsync(configLoader, request.ModelName, request.Task,
                request.LabelColumn, request.TimeLimitSeconds, request.Metric, request.TestSplit, request.Weight);
            if (definitionError != null)
                return Results.BadRequest(new { error = definitionError });
            if (TrainingInput.BalanceConflict(trainDefinition!, request.Balance) is { } balanceError)
                return Results.BadRequest(new { error = balanceError });

            // Create job and enqueue for background processing
            var jobId = jobStore.CreateJob(request);
            jobRunner.EnqueueJob(jobId);

            logger.LogInformation("Training job '{JobId}' enqueued for model '{ModelName}' with task '{Task}'",
                jobId, request.ModelName, request.Task);

            return Results.Accepted($"/jobs/{jobId}", new
            {
                jobId,
                modelName = request.ModelName,
                status = "Queued",
                message = "Training job created. Poll /jobs/{jobId} for status."
            });
        })
        .WithName("StartTraining")
        .WithTags("Training")
        .Produces<object>(StatusCodes.Status202Accepted)
        .Produces<object>(StatusCodes.Status400BadRequest)
        .RequireAuthorization("Admin");

        // List all jobs
        app.MapGet("/jobs", (
            TrainingJobStore jobStore,
            ILogger<Program> logger) =>
        {
            var jobs = jobStore.GetAllJobs();

            return Results.Ok(new
            {
                count = jobs.Count,
                jobs = jobs.Select(j => new
                {
                    jobId = j.JobId,
                    modelName = j.ModelName,
                    task = j.Task,
                    status = j.Status.ToString(),
                    statusMessage = j.StatusMessage,
                    createdAt = j.CreatedAt,
                    completedAt = j.CompletedAt,
                    experimentId = j.ExperimentId,
                    bestTrainer = j.BestTrainer,
                    promoted = j.Promoted
                })
            });
        })
        .WithName("ListJobs")
        .WithTags("Training")
        .Produces<object>(StatusCodes.Status200OK)
        .RequireAuthorization();

        // Get job status
        app.MapGet("/jobs/{id}", (
            string id,
            TrainingJobStore jobStore,
            ILogger<Program> logger) =>
        {
            var job = jobStore.GetJob(id);
            if (job == null)
                return Results.NotFound(new { error = $"Job '{id}' not found." });

            return Results.Ok(new
            {
                jobId = job.JobId,
                modelName = job.ModelName,
                dataFile = job.DataFile,
                additionalDataFiles = job.AdditionalDataFiles,
                labelColumn = job.LabelColumn,
                task = job.Task,
                timeLimitSeconds = job.TimeLimitSeconds,
                metric = job.Metric,
                testSplit = job.TestSplit,
                maxRows = job.MaxRows,
                weight = job.Weight,
                balance = job.Balance,
                promote = job.Promote,
                status = job.Status.ToString(),
                statusMessage = job.StatusMessage,
                createdAt = job.CreatedAt,
                completedAt = job.CompletedAt,
                experimentId = job.ExperimentId,
                metrics = job.Metrics,
                bestTrainer = job.BestTrainer,
                promoted = job.Promoted,
                keptInStagingReason = job.KeptInStagingReason,
                warnings = job.Warnings
            });
        })
        .WithName("GetJobStatus")
        .WithTags("Training")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<object>(StatusCodes.Status404NotFound)
        .RequireAuthorization();
    }
}
