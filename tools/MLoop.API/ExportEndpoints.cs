using System.Diagnostics;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Export;
using MLoop.Core.Storage;

namespace MLoop.API;

/// <summary>
/// Export endpoints: an experiment's model as an ONNX graph plus a manifest — what <c>mloop export onnx</c> writes, for a
/// consumer that reaches the server over HTTP and has no command line beside it — and the two files back.
/// </summary>
public static class ExportEndpoints
{
    public static void MapExportEndpoints(this WebApplication app)
    {
        app.MapPost("/experiments/{id}/export/onnx", async Task<Results<Ok<OnnxExportResponse>, NotFound<ErrorResponse>, BadRequest<ErrorResponse>, Conflict<ErrorResponse>, UnprocessableEntity<ErrorResponse>, ProblemHttpResult>> (
            string id,
            [FromQuery] string? name,
            [FromBody] OnnxExportBody? body,
            IExperimentStore experimentStore,
            ProjectRootPath projectRootPath,
            ILogger<Program> logger,
            CancellationToken ct) =>
        {
            var stopwatch = Stopwatch.StartNew();
            var modelName = MLoop.Core.Storage.ModelName.Resolve(name);
            var root = projectRootPath.Value;
            string Said(string text) => ProjectRelativeText.In(text, root) ?? text;

            try
            {
                // Written beside the experiment and nowhere else: a caller names an experiment, not a place on the server.
                var outcome = await new OnnxExporter(experimentStore).ExportAsync(
                    new OnnxExportRequest(root, modelName, id, body?.Data, OutputDirectory: null, body?.Verify ?? false, DataWithinProject: true, Annotations: body?.Annotations), ct);

                if (outcome.Refusal is { } refusal)
                {
                    logger.LogInformation("Export of '{ExperimentId}' ({ModelName}) refused: {Kind}", id, modelName, refusal.Kind);
                    var message = Said(refusal.Message);
                    return refusal.Kind switch
                    {
                        OnnxExportRefusalKind.ExperimentNotFound => TypedResults.NotFound(new ErrorResponse(
                            $"{message} GET /experiments lists the experiments of '{modelName}'.")),
                        OnnxExportRefusalKind.NoDataFile => TypedResults.BadRequest(new ErrorResponse(
                            $"{message} Pass \"data\": a file within the project with the model's input columns.")),
                        OnnxExportRefusalKind.DataOutsideProject or OnnxExportRefusalKind.InvalidAnnotations
                            => TypedResults.BadRequest(new ErrorResponse(message)),
                        OnnxExportRefusalKind.VerifyUnavailable => TypedResults.Conflict(new ErrorResponse(
                            $"\"verify\" needs ONNX Runtime on this server. {message}")),
                        OnnxExportRefusalKind.GraphDisagrees => TypedResults.UnprocessableEntity(new ErrorResponse(
                            $"{message} The comparison is in the manifest: GET /experiments/{id}/export/onnx/{ExperimentLayout.OnnxManifestFileName}.")),
                        _ => TypedResults.UnprocessableEntity(new ErrorResponse(message)),
                    };
                }

                var export = outcome.Output!;
                var query = modelName == MLoop.Core.Storage.ModelName.Resolve(null) ? "" : $"?name={Uri.EscapeDataString(modelName)}";
                string Url(string file) => $"/experiments/{Uri.EscapeDataString(export.ExperimentId)}/export/onnx/{file}{query}";
                logger.LogInformation("Exported '{ExperimentId}' ({ModelName}) to ONNX in {ElapsedMs}ms", export.ExperimentId, modelName, stopwatch.ElapsedMilliseconds);
                return TypedResults.Ok(new OnnxExportResponse(
                    export.ModelName,
                    export.ExperimentId,
                    export.Task,
                    OnnxSha256: OnnxManifest.Sha256Of(export.OnnxPath),
                    ManifestSha256: OnnxManifest.Sha256Of(export.ManifestPath),
                    SourceModelSha256: export.Manifest.SourceModelSha256,
                    OnnxUrl: Url(ExperimentLayout.OnnxModelFileName),
                    ManifestUrl: Url(ExperimentLayout.OnnxManifestFileName),
                    Portable: export.Manifest.Portable,
                    Verification: export.Verification));
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                logger.LogError(ex, "Failed to export experiment '{ExperimentId}' after {ElapsedMs}ms", id, stopwatch.ElapsedMilliseconds);
                return TypedResults.Problem(
                    title: "Failed to export experiment",
                    detail: Said(ex.Message),
                    statusCode: StatusCodes.Status500InternalServerError);
            }
        })
        .WithName("ExportOnnx")
        .WithTags("Experiments")
        .RequireAuthorization("Admin");

        app.MapGet("/experiments/{id}/export/onnx/{file}", Results<PhysicalFileHttpResult, NotFound<ErrorResponse>> (
            string id,
            string file,
            [FromQuery] string? name,
            IExperimentStore experimentStore) =>
        {
            var modelName = MLoop.Core.Storage.ModelName.Resolve(name);
            // Two names, not a path: the files the export writes, and nothing else under the experiment.
            var contentType = file switch
            {
                ExperimentLayout.OnnxModelFileName => "application/octet-stream",
                ExperimentLayout.OnnxManifestFileName => "application/json",
                _ => null,
            };
            if (contentType is null)
                return TypedResults.NotFound(new ErrorResponse(
                    $"'{file}' is not an export file; they are {ExperimentLayout.OnnxModelFileName} and {ExperimentLayout.OnnxManifestFileName}."));
            if (!experimentStore.ExperimentExists(modelName, id))
                return TypedResults.NotFound(new ErrorResponse($"Experiment '{id}' not found for model '{modelName}'."));
            var path = Path.GetFullPath(Path.Combine(experimentStore.GetExperimentPath(modelName, id), file));
            if (!File.Exists(path))
                return TypedResults.NotFound(new ErrorResponse(
                    $"Experiment '{id}' has no {file}: POST /experiments/{id}/export/onnx writes it."));
            return TypedResults.PhysicalFile(path, contentType, fileDownloadName: file);
        })
        .WithName("GetOnnxExportFile")
        .WithTags("Experiments")
        .RequireAuthorization();
    }
}
