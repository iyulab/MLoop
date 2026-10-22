using System.Security.Cryptography;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLoop.Core.Export;

/// <summary>
/// The machine-readable description written beside an exported ONNX graph: what to feed it, what it
/// answers, how to read the answer, and which trained model it came from — enough for a runtime
/// without ML.NET to use the graph and for a reader to check it is the model they think it is.
/// </summary>
public sealed record OnnxManifest
{
    /// <summary>Version of this manifest's shape. Raised when a field changes meaning.</summary>
    public int ManifestVersion { get; init; } = 1;

    /// <summary>The graph file, relative to the manifest.</summary>
    public required string OnnxFile { get; init; }

    /// <summary>SHA-256 of the graph file, lowercase hex.</summary>
    public required string OnnxSha256 { get; init; }

    /// <summary>SHA-256 of the ML.NET model the graph was converted from, lowercase hex.</summary>
    public required string SourceModelSha256 { get; init; }

    public required string ModelName { get; init; }
    public required string ExperimentId { get; init; }

    /// <summary>The task the model was trained for (canonical name).</summary>
    public required string Task { get; init; }

    /// <summary>The MLoop version that wrote the export.</summary>
    public string? MloopVersion { get; init; }

    /// <summary>When the export was written (UTC).</summary>
    public DateTimeOffset ExportedAt { get; init; }

    public required IReadOnlyList<OnnxTensorSpec> Inputs { get; init; }

    /// <summary>Inputs the graph declares that no output depends on; feed any value of the type.</summary>
    public IReadOnlyList<OnnxTensorSpec> UnusedInputs { get; init; } = [];

    public required IReadOnlyList<OnnxOutputSpec> Outputs { get; init; }

    /// <summary>How to read the answer: label column, classes and decision rule.</summary>
    public required OnnxLabelInfo Label { get; init; }

    /// <summary>The experiment's recorded test metrics.</summary>
    public IReadOnlyDictionary<string, double>? Metrics { get; init; }

    /// <summary>The column that weighted the rows in training and in the metrics, if any.</summary>
    public string? WeightColumn { get; init; }

    /// <summary>
    /// Whether the model was trained portable — with text featurized by character n-gram counts,
    /// which are case-sensitive and take the text as it is (no normalization).
    /// </summary>
    public bool Portable { get; init; }

    /// <summary>
    /// The binary model's decision threshold, when one was chosen (<c>mloop threshold</c>): the graph does
    /// not apply it — its <c>PredictedLabel</c> is the trainer's own — so a consumer answering as the model
    /// does compares the probability with it, after <see cref="Calibration"/> when there is one
    /// (<see cref="OnnxDecisionRecipe"/>). Null when the model answers at its trainer's threshold.
    /// </summary>
    public Models.DecisionThreshold? DecisionThreshold { get; init; }

    /// <summary>
    /// The probability calibration beside the model, when one was fit — not in the graph, which carries
    /// the trained model's own probability. Apply it (then <see cref="DecisionThreshold"/>) to answer as
    /// <c>mloop predict</c> does.
    /// </summary>
    public Models.ProbabilityCalibration? Calibration { get; init; }

    /// <summary>How the graph compared with the model on data rows (<c>--verify</c>); null when not checked.</summary>
    public OnnxVerificationResult? Verification { get; init; }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        // A file people read too: keep "Score > 0" as written rather than \u003E.
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public string ToJson() => JsonSerializer.Serialize(this, JsonOptions);

    public static OnnxManifest FromJson(string json) =>
        JsonSerializer.Deserialize<OnnxManifest>(json, JsonOptions)
        ?? throw new InvalidDataException("The ONNX manifest is empty.");

    /// <summary>SHA-256 of a file, lowercase hex.</summary>
    public static string Sha256Of(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}

/// <summary>How an exported model's answer reads.</summary>
public sealed record OnnxLabelInfo
{
    /// <summary>The label column the model was trained on; null for unsupervised tasks.</summary>
    public string? Column { get; init; }

    /// <summary>
    /// Class names. Binary: <c>[negative, positive]</c> — <c>PredictedLabel</c> true means the second.
    /// Multiclass: in <c>Score</c> slot order. Null when the model carries no names.
    /// </summary>
    public IReadOnlyList<string>? Classes { get; init; }

    /// <summary>The rule that turns the outputs into the answer <c>mloop predict</c> gives, in words (<see cref="OnnxDecisionRecipe.Describe"/>).</summary>
    public string? Decision { get; init; }
}
