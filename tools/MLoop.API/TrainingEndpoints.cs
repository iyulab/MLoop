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
            HttpContext http,
            TrainingJobStore jobStore,
            TrainingJobRunner jobRunner,
            ProjectRootPath projectRootPath,
            ConfigLoader configLoader,
            ILogger<Program> logger) =>
        {
            // The definition first: the task decides what the data is (a table, or a directory of images), and the
            // request may leave the task and the label to mloop.yaml. Refused now, not inside the job: what the
            // project's mloop.yaml asks for that this run could not honour (a weight on a task it does not reach,
            // prep steps the API does not run), and what no task could train (an unknown task, a missing label).
            var (trainDefinition, definitionError) = await TrainingDefinition.ResolveAsync(configLoader,
                new TrainingDefinition.Request(
                    ModelName: request.ResolveModelName(), Task: request.Task, Label: request.LabelColumn, TimeLimitSeconds: request.TimeLimitSeconds, Metric: request.Metric,
                    TestSplit: request.TestSplit, Weight: request.Weight, MaxTrials: request.MaxTrials, Epochs: request.Epochs, Ensemble: request.Ensemble));
            if (definitionError != null)
                return Results.BadRequest(new { error = definitionError });
            if (TrainingInput.BalanceConflict(trainDefinition!, request.Balance) is { } balanceError)
                return Results.BadRequest(new { error = balanceError });
            // Refuse a metric the task cannot optimize now, not minutes later inside the job — the job
            // would otherwise fail (or, before this, quietly optimize the task default).
            var metricRejection = MLoop.Core.Evaluation.MetricNames.Rejection(trainDefinition!.Task, request.Metric);
            if (metricRejection != null)
                return Results.BadRequest(new { error = metricRejection });

            // Validate every data file exists and is within project root. A relative path is the project's — the
            // form POST /samples answers with — not the server process's working directory.
            string InProject(string file) => ProjectPaths.Resolve(file, projectRootPath.Value);
            string dataFile;
            if (DataLoaderFactory.IsDirectoryBased(trainDefinition!.Task))
            {
                // Images: a directory, as `mloop train` reads it (class subfolders; a COCO annotations file plus
                // its images) — named, or the task's place under datasets/ when the request names none.
                var tableOnly = new List<string>();
                if (request.AdditionalDataFiles is { Count: > 0 }) tableOnly.Add("additionalDataFiles");
                if (request.MaxRows is not null) tableOnly.Add("maxRows");
                if (TrainingInput.TableOnlyOptionsRefusal(trainDefinition.Task, tableOnly) is { } tableOnlyRefusal)
                    return Results.BadRequest(new { error = tableOnlyRefusal });
                var directory = TrainingInput.ResolveDirectoryDataset(trainDefinition.Task, request.DataFile, null, projectRootPath.Value);
                if (directory is null)
                {
                    var missing = TrainingInput.DirectoryDatasetNotFound(trainDefinition.Task);
                    var named = string.IsNullOrWhiteSpace(request.DataFile)
                        ? ""
                        : File.Exists(InProject(request.DataFile))
                            ? $" '{request.DataFile}' is a file; {trainDefinition.Task} trains on a directory (not its annotations file)."
                            : $" Not a directory: {request.DataFile}.";
                    return Results.BadRequest(new { error = $"{missing.Message}{named} {missing.Hint}" });
                }
                if (!ProjectPaths.IsWithinReal(directory, projectRootPath.Value))
                    return Results.BadRequest(new { error = $"Data directory must be within the project directory: {request.DataFile}" });
                if (TrainingInput.DirectoryDatasetRefusal(trainDefinition.Task, directory) is { } wrongLayout)
                    return Results.BadRequest(new { error = $"{ProjectRelativeText.In(wrongLayout.Message, projectRootPath.Value)} {wrongLayout.Hint}" });
                dataFile = directory;
            }
            else
            {
                if (string.IsNullOrWhiteSpace(request.DataFile))
                    return Results.BadRequest(new { error = "dataFile is required: the training data, within the project." });
                foreach (var file in new[] { request.DataFile }.Concat(request.AdditionalDataFiles ?? []))
                {
                    var resolvedDataFile = InProject(file);
                    if (!ProjectPaths.IsWithinReal(resolvedDataFile, projectRootPath.Value))
                        return Results.BadRequest(new { error = $"Data file must be within the project directory: {file}" });
                    if (Directory.Exists(resolvedDataFile))
                        return Results.BadRequest(new { error = $"Data file is a directory: {file}. '{trainDefinition.Task}' trains on a table (a file)." });
                    if (!File.Exists(resolvedDataFile))
                        return Results.BadRequest(new { error = $"Data file not found: {file}" });
                }
                dataFile = InProject(request.DataFile);
            }
            // The job records what it trains with — the resolved task, label and data — not the blanks the request left.
            request = request with
            {
                DataFile = dataFile,
                AdditionalDataFiles = request.AdditionalDataFiles?.Select(InProject).ToList(),
                Task = trainDefinition.Task,
                LabelColumn = trainDefinition.Label,
            };

            // Create job and enqueue for background processing
            var jobId = jobStore.CreateJob(request, ModelEndpoints.CallerOrigin(http).By);
            jobRunner.EnqueueJob(jobId);

            logger.LogInformation("Training job '{JobId}' enqueued for model '{ModelName}' with task '{Task}'",
                jobId, request.ResolveModelName(), request.Task);

            return Results.Accepted($"/jobs/{jobId}", new
            {
                jobId,
                modelName = request.ResolveModelName(),
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
            ProjectRootPath projectRootPath,
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
                    statusMessage = ProjectRelativeText.In(j.StatusMessage, projectRootPath.Value),
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
            ProjectRootPath projectRootPath,
            ILogger<Program> logger) =>
        {
            var job = jobStore.GetJob(id);
            if (job == null)
                return Results.NotFound(new { error = $"Job '{id}' not found." });

            return Results.Ok(new
            {
                jobId = job.JobId,
                modelName = job.ModelName,
                // The project's paths, as the request wrote them — not the host's directory layout.
                dataFile = ProjectRelativeText.In(job.DataFile, projectRootPath.Value),
                additionalDataFiles = ProjectRelativeText.In(job.AdditionalDataFiles, projectRootPath.Value),
                labelColumn = job.LabelColumn,
                task = job.Task,
                timeLimitSeconds = job.TimeLimitSeconds,
                maxTrials = job.MaxTrials,
                epochs = job.Epochs,
                ensemble = job.Ensemble,
                metric = job.Metric,
                testSplit = job.TestSplit,
                maxRows = job.MaxRows,
                weight = job.Weight,
                balance = job.Balance,
                promote = job.Promote,
                status = job.Status.ToString(),
                statusMessage = ProjectRelativeText.In(job.StatusMessage, projectRootPath.Value),
                createdAt = job.CreatedAt,
                completedAt = job.CompletedAt,
                experimentId = job.ExperimentId,
                metrics = job.Metrics,
                bestTrainer = job.BestTrainer,
                promoted = job.Promoted,
                keptInStagingReason = ProjectRelativeText.In(job.KeptInStagingReason, projectRootPath.Value),
                warnings = ProjectRelativeText.In(job.Warnings, projectRootPath.Value)
            });
        })
        .WithName("GetJobStatus")
        .WithTags("Training")
        .Produces<object>(StatusCodes.Status200OK)
        .Produces<object>(StatusCodes.Status404NotFound)
        .RequireAuthorization();
    }
}
