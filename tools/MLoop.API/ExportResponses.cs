using MLoop.Core.Export;

namespace MLoop.API;

// The bodies of the export endpoints (ExportEndpoints) — see ExperimentResponses for why they are named types.

/// <summary>
/// The body of <c>POST /experiments/{id}/export/onnx</c>; the whole body may be left out.
/// </summary>
/// <param name="Data">
/// A data file within the project with the model's input columns, read for the graph's input schema; absent: the file the
/// experiment was trained on (an upload stays until it is deleted).
/// </param>
/// <param name="Verify">
/// Run the graph in ONNX Runtime on up to <see cref="OnnxVerification.DefaultRows"/> rows of the data and compare it with the model
/// (and the manifest's decision rule with what <c>/predict</c> answers); a graph that disagrees is not kept. Needs ONNX Runtime on the server.
/// </param>
[System.Text.Json.Serialization.JsonUnmappedMemberHandling(System.Text.Json.Serialization.JsonUnmappedMemberHandling.Disallow)]
public sealed record OnnxExportBody(string? Data = null, bool Verify = false);

/// <summary>
/// <c>POST /experiments/{id}/export/onnx</c>: the graph and its manifest are written beside the experiment. The two
/// SHA-256 values are of the files' bytes — what <c>GET</c> on the two URLs returns; the manifest carries the graph's
/// (<c>onnxSha256</c>) and the model's (<c>sourceModelSha256</c>) too.
/// </summary>
public sealed record OnnxExportResponse(
    string ModelName,
    string ExperimentId,
    string Task,
    string OnnxSha256,
    string ManifestSha256,
    string SourceModelSha256,
    string OnnxUrl,
    string ManifestUrl,
    bool Portable,
    OnnxVerificationResult? Verification);
