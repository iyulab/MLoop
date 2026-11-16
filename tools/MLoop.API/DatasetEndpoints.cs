using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Storage;

namespace MLoop.API;

/// <summary>
/// Training data over HTTP: <c>POST /datasets</c> takes a table file, or a zip of a directory dataset, as the request body and
/// keeps it under the project's <c>datasets/uploads/&lt;id&gt;/</c>; the answer's <c>dataFile</c> is the project-relative path
/// <c>POST /train</c> takes. A client that trains through the server no longer has to share the server's disk to hand it the
/// data. The upload and the training request are two calls on purpose: the same data trains several experiments, a failed
/// upload takes no place in the job queue, and a retry repeats only the call that failed. The rules of what lands on the disk
/// are <see cref="DatasetStore"/>'s.
/// </summary>
public static class DatasetEndpoints
{
    public static void MapDatasetEndpoints(this WebApplication app)
    {
        // The body is the file, streamed to disk: not JSON, not a form buffered in memory.
        app.MapPost("/datasets", async Task<Results<Created<DatasetResponse>, BadRequest<ErrorResponse>, JsonHttpResult<ErrorResponse>>> (
            HttpRequest request,
            [FromQuery] string? fileName,
            [FromQuery] string? task,
            ProjectRootPath projectRootPath,
            IConfiguration configuration,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var limits = DatasetLimits.From(configuration);
            if (request.ContentLength > limits.MaxUploadBytes)
                return TypedResults.Json(new ErrorResponse(TooLarge(limits)), statusCode: StatusCodes.Status413PayloadTooLarge);
            // Kestrel's default body limit is 30 MB — below any dataset; this endpoint's is the configured one (the counting
            // copy below enforces it too, where the server cannot raise it).
            var sizeLimit = request.HttpContext.Features.Get<IHttpMaxRequestBodySizeFeature>();
            if (sizeLimit is { IsReadOnly: false })
                sizeLimit.MaxRequestBodySize = limits.MaxUploadBytes;

            try
            {
                var store = new DatasetStore(projectRootPath.Value);
                var dataset = await store.SaveAsync(request.Body, fileName, limits, path => LayoutRefusal(task, path, projectRootPath.Value), ct, request.ContentLength);
                logger.LogInformation("Dataset '{Id}' stored: {Kind}, {Files} file(s), {Bytes} bytes", dataset.Id, dataset.Kind, dataset.Files, dataset.Bytes);
                return TypedResults.Created($"/datasets/{dataset.Id}", DatasetResponse.Of(dataset));
            }
            catch (DatasetRefusedException ex)
            {
                return ex.StatusCode == StatusCodes.Status413PayloadTooLarge
                    ? TypedResults.Json(new ErrorResponse(ex.Message), statusCode: StatusCodes.Status413PayloadTooLarge)
                    : TypedResults.BadRequest(new ErrorResponse(ex.Message));
            }
            catch (BadHttpRequestException ex) when (ex.StatusCode == StatusCodes.Status413PayloadTooLarge)
            {
                return TypedResults.Json(new ErrorResponse(TooLarge(limits)), statusCode: StatusCodes.Status413PayloadTooLarge);
            }
        })
        .WithName("UploadDataset")
        .WithTags("Datasets")
        // The body is the file, whatever content type the client labels it with (curl sends form-urlencoded unless told otherwise): the
        // name decides table or zip, so no type is turned away with a 415.
        .Accepts<Stream>("application/octet-stream", "application/zip", "*/*")
        .DisableAntiforgery()
        .RequireAuthorization("Admin");

        app.MapGet("/datasets", (ProjectRootPath projectRootPath) =>
        {
            var datasets = new DatasetStore(projectRootPath.Value).List();
            return TypedResults.Ok(new DatasetListResponse(datasets.Count, [.. datasets.Select(DatasetResponse.Of)]));
        })
        .WithName("ListDatasets")
        .WithTags("Datasets")
        .RequireAuthorization();

        app.MapDelete("/datasets/{id}", Results<NoContent, NotFound<ErrorResponse>, Conflict<ErrorResponse>> (
            string id,
            ProjectRootPath projectRootPath,
            TrainingJobStore jobStore) =>
        {
            var store = new DatasetStore(projectRootPath.Value);
            if (store.Find(id) is null)
                return TypedResults.NotFound(new ErrorResponse($"No dataset '{id}' (GET /datasets lists them)."));
            // A queued or running job reads the data by path: removing it would fail the job minutes in.
            var reading = store.JobReading(id, jobStore);
            if (reading is not null)
                return TypedResults.Conflict(new ErrorResponse($"Job '{reading.JobId}' is training on dataset '{id}'; remove it when the job has finished."));
            store.Delete(id);
            return TypedResults.NoContent();
        })
        .WithName("DeleteDataset")
        .WithTags("Datasets")
        .RequireAuthorization("Admin");
    }

    private static string TooLarge(DatasetLimits limits) =>
        $"The upload is larger than the server takes ({limits.MaxUploadBytes:N0} bytes; Serve:Datasets:MaxUploadBytes).";

    /// <summary>When the upload names the task it is for: whether its shape is the one the task trains on — a table for a table task, a
    /// directory dataset in the layout of the task for an image one — asked of the same functions <c>POST /train</c> asks.</summary>
    private static string? LayoutRefusal(string? task, string path, string projectRoot)
    {
        if (string.IsNullOrWhiteSpace(task))
            return null;
        var wantsDirectory = DataLoaderFactory.IsDirectoryBased(task);
        var isDirectory = Directory.Exists(path);
        if (wantsDirectory && !isDirectory)
            return $"'{task}' trains on a directory dataset: upload a .zip of it, not a table file.";
        if (!wantsDirectory && isDirectory)
            return $"'{task}' trains on a table: upload one table file, not a .zip.";
        if (wantsDirectory && TrainingInput.DirectoryDatasetRefusal(task, path) is { } wrongLayout)
            return $"{ProjectRelativeText.In(wrongLayout.Message, projectRoot)} {wrongLayout.Hint}";
        return null;
    }
}
