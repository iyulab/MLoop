using System.Collections.Concurrent;
using Microsoft.ML;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Data;
using MLoop.Core.Export;
using MLoop.Core.Models;
using MLoop.Core.Runtime;
using MLoop.Core.Storage;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>What an export is asked to do.</summary>
/// <param name="ProjectRoot">The project the experiment belongs to; relative paths are the project's.</param>
/// <param name="ModelName">The model whose experiment is exported.</param>
/// <param name="ExperimentId">The experiment to export.</param>
/// <param name="DataFile">A data file with the model's input columns (read for the input schema); null: the file the experiment trained on.</param>
/// <param name="OutputDirectory">Where <c>model.onnx</c> and its manifest are written; null: the experiment's own directory.</param>
/// <param name="Verify">Run the graph in ONNX Runtime and compare it with the model; a graph that disagrees is not kept.</param>
/// <param name="Annotations">Opaque pairs written into the manifest (see <see cref="OnnxAnnotations"/>); null: none.</param>
/// <param name="DataWithinProject">
/// Both data files — the named one and the one the experiment recorded — must lie inside the project (a server reads
/// what its caller names; the command line reads what its user can already read).
/// </param>
public sealed record OnnxExportRequest(
    string ProjectRoot,
    string ModelName,
    string ExperimentId,
    string? DataFile = null,
    string? OutputDirectory = null,
    bool Verify = false,
    bool DataWithinProject = false,
    IReadOnlyDictionary<string, string>? Annotations = null);

/// <summary>Why an export was refused. A refusal writes no graph (<see cref="GraphDisagrees"/> keeps a manifest recording why).</summary>
public enum OnnxExportRefusalKind
{
    /// <summary>The experiment does not exist or has no model.</summary>
    ExperimentNotFound,
    /// <summary>The pairs to write into the manifest are not acceptable (too many, too long, an empty key).</summary>
    InvalidAnnotations,
    /// <summary>The experiment's task has no verified export.</summary>
    UnsupportedTask,
    /// <summary>The model reads a column through a text encoder, which a graph does not carry.</summary>
    TextEncoder,
    /// <summary>There is no data file to read the model's input columns from.</summary>
    NoDataFile,
    /// <summary>The data file lies outside the project and the request keeps data inside it.</summary>
    DataOutsideProject,
    /// <summary>Verification was asked for and ONNX Runtime cannot run here; the message is why, and how to make it run.</summary>
    VerifyUnavailable,
    /// <summary>The model has a step with no ONNX form.</summary>
    NotConvertible,
    /// <summary>The graph answers differently from the model on the checked rows; it was not kept.</summary>
    GraphDisagrees,
}

/// <summary>A refusal, said without the words of any one surface (an option name, a command): each surface adds its own next step.</summary>
public sealed record OnnxExportRefusal(OnnxExportRefusalKind Kind, string Message, string? ManifestPath = null);

/// <summary>A written graph and its manifest.</summary>
public sealed record OnnxExportOutput(
    string ModelName,
    string ExperimentId,
    string Task,
    string OnnxPath,
    string ManifestPath,
    OnnxManifest Manifest,
    OnnxExportResult Result,
    OnnxVerificationResult? Verification);

/// <summary>An export's answer: the output, or the refusal.</summary>
public sealed record OnnxExportOutcome(OnnxExportOutput? Output, OnnxExportRefusal? Refusal);

/// <summary>
/// Writes a trained model as an ONNX graph plus a manifest, for runtimes that cannot load ML.NET (Native AOT
/// executables, edge agents, other languages). The one place the export is done: <c>mloop export onnx</c> and
/// <c>POST /experiments/{id}/export/onnx</c> both call it, so a graph is the same whichever surface asked for it.
/// </summary>
public sealed class OnnxExporter
{
    /// <summary>Tasks whose models the export has been verified for.</summary>
    public static readonly IReadOnlySet<string> SupportedTasks =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "binary-classification", "multiclass-classification", "regression",
        };

    // One export at a time per output directory: two writers of the same model.onnx and manifest would leave
    // a graph from one and a manifest from the other. A server is one process, so a process-wide lock is the whole scope.
    private static readonly ConcurrentDictionary<string, SemaphoreSlim> DirectoryLocks = new(StringComparer.Ordinal);

    private readonly IExperimentStore _experiments;

    public OnnxExporter(IExperimentStore experiments)
    {
        _experiments = experiments ?? throw new ArgumentNullException(nameof(experiments));
    }

    public async Task<OnnxExportOutcome> ExportAsync(OnnxExportRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var modelName = request.ModelName;
        var experimentId = request.ExperimentId.Trim();

        if (OnnxAnnotations.Refusal(request.Annotations) is { } annotationRefusal)
            return Refused(OnnxExportRefusalKind.InvalidAnnotations, annotationRefusal);

        var (modelPath, notFound) = ExperimentModels.Find(_experiments, modelName, experimentId);
        if (modelPath is null)
            return Refused(OnnxExportRefusalKind.ExperimentNotFound, notFound!);

        // Checked before any work: a verification that cannot run is a refusal, not a skipped step.
        if (request.Verify && OnnxRuntimeNative.Unavailable() is { } unavailable)
            return Refused(OnnxExportRefusalKind.VerifyUnavailable, unavailable);

        var experiment = await _experiments.LoadAsync(modelName, experimentId, cancellationToken);
        var task = TaskTypes.Canonical(experiment.Task) ?? experiment.Task ?? "unknown";
        if (!SupportedTasks.Contains(task))
            return Refused(OnnxExportRefusalKind.UnsupportedTask,
                $"ONNX export supports {string.Join(", ", SupportedTasks.Order(StringComparer.Ordinal))} models; '{experimentId}' is {task}.");

        var experimentPath = _experiments.GetExperimentPath(modelName, experimentId);
        if (TextEncodings.ReadBeside(modelPath) is { Columns.Count: > 0 } encodings)
        {
            // The graph reads the columns' sentence vectors; the encoders that make them are not in it.
            return Refused(OnnxExportRefusalKind.TextEncoder,
                $"'{experimentId}' reads {string.Join(", ", encodings.Columns.Select(c => $"'{c.Column}'"))} through a text encoder, " +
                "which an ONNX graph of the model does not carry; it cannot be exported.");
        }

        // The input schema comes from data, because the model was saved without one; the training file is the
        // natural default — it has exactly the columns the model was fit on.
        var named = request.DataFile is { Length: > 0 };
        var dataPath = named
            ? ProjectPaths.Resolve(request.DataFile!, request.ProjectRoot)
            : experiment.Config?.DataFile is { Length: > 0 } recorded
                ? ProjectPaths.Resolve(recorded, request.ProjectRoot)
                : null;
        if (dataPath is not null && request.DataWithinProject && !ProjectPaths.IsWithinReal(dataPath, request.ProjectRoot))
        {
            return named
                ? Refused(OnnxExportRefusalKind.DataOutsideProject, $"The data file must be within the project directory: {request.DataFile}")
                : Refused(OnnxExportRefusalKind.NoDataFile,
                    $"Data file not found: {dataPath}. It lies outside the project directory, so it is not read for the model's input columns.");
        }
        if (dataPath is null || !TabularDataFile.Exists(dataPath))
        {
            return Refused(OnnxExportRefusalKind.NoDataFile, dataPath is null
                ? "No data file to read the model's input columns from."
                : $"Data file not found: {dataPath}.");
        }

        var outputDirectory = request.OutputDirectory is { Length: > 0 }
            ? ProjectPaths.Resolve(request.OutputDirectory, request.ProjectRoot)
            : experimentPath;

        var gate = DirectoryLocks.GetOrAdd(Path.GetFullPath(outputDirectory), _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancellationToken);
        var tempFiles = new List<string>();
        try
        {
            dataPath = await TabularDataFile.AsCsvAsync(dataPath);

            var labelColumn = string.IsNullOrEmpty(experiment.Config?.LabelColumn) ? null : experiment.Config.LabelColumn;
            var engine = new PredictionEngine();
            var input = engine.LoadInferenceInput(
                modelPath, dataPath, experiment.Config?.InputSchema, labelColumn,
                PreserveColumns(experiment.Config), tempFiles, layers: ScoringLayers.None);

            Directory.CreateDirectory(outputDirectory);
            var onnxPath = Path.Combine(outputDirectory, ExperimentLayout.OnnxModelFileName);
            var manifestPath = Path.Combine(outputDirectory, ExperimentLayout.OnnxManifestFileName);

            OnnxExportResult result;
            var partial = $"{onnxPath}.{Guid.NewGuid():N}.partial";
            try
            {
                using (var stream = File.Create(partial))
                {
                    result = OnnxExport.Write(new MLContext(seed: 42), input.Model, input.Data,
                        input.LoaderOptions, input.Header, input.LabelColumn, stream);
                }
                File.Move(partial, onnxPath, overwrite: true);
            }
            catch (OnnxExportException ex)
            {
                return Refused(OnnxExportRefusalKind.NotConvertible, ex.Message);
            }
            finally
            {
                if (File.Exists(partial))
                    File.Delete(partial);
            }

            var recipe = OnnxDecisionRecipe.Beside(task, modelPath);
            OnnxVerificationResult? verification = null;
            if (request.Verify)
            {
                // The rule is checked too: replayed on the graph's outputs, it has to give what
                // mloop predict gives — the model with the layers beside it.
                var layered = recipe.HasLayers ? ScoringModel.Load(new MLContext(seed: 42), modelPath, task) : null;
                try
                {
                    verification = OnnxVerification.Compare(input.Model, input.Data, result, onnxPath, recipe: recipe, layered: layered);
                }
                catch (OnnxExportException ex)
                {
                    return Refused(OnnxExportRefusalKind.NotConvertible, ex.Message);
                }
                if (!verification.Passed)
                {
                    // A graph that answers differently from the model is not left where a consumer
                    // would pick it up; the manifest stays, recording why.
                    File.Delete(onnxPath);
                }
            }

            var manifest = new OnnxManifest
            {
                OnnxFile = Path.GetFileName(onnxPath),
                OnnxSha256 = File.Exists(onnxPath) ? OnnxManifest.Sha256Of(onnxPath) : "",
                SourceModelSha256 = OnnxManifest.Sha256Of(modelPath),
                ModelName = modelName,
                ExperimentId = experimentId,
                Task = task,
                MloopVersion = ErrorSuggestions.GetVersion(),
                ExportedAt = DateTimeOffset.UtcNow,
                Inputs = result.Inputs,
                UnusedInputs = result.UnusedInputs,
                Outputs = result.Outputs,
                Label = LabelInfo(task, labelColumn, experiment, input, result, recipe),
                DecisionThreshold = recipe.Threshold,
                Calibration = recipe.Calibration,
                Metrics = experiment.Metrics,
                WeightColumn = experiment.Config?.WeightColumn,
                Portable = experiment.Config?.Portable == true,
                Verification = verification,
                Annotations = OnnxAnnotations.Normalize(request.Annotations),
            };
            await File.WriteAllTextAsync(manifestPath, manifest.ToJson(), cancellationToken);

            if (verification is { Passed: false })
            {
                return new OnnxExportOutcome(null, new OnnxExportRefusal(OnnxExportRefusalKind.GraphDisagrees,
                    $"The exported graph disagrees with the model on {verification.Rows} rows checked ({Describe(verification)}); it was not kept.",
                    manifestPath));
            }

            return new OnnxExportOutcome(
                new OnnxExportOutput(modelName, experimentId, task, onnxPath, manifestPath, manifest, result, verification), null);
        }
        finally
        {
            foreach (var file in tempFiles)
            {
                try { if (File.Exists(file)) File.Delete(file); } catch { /* best effort */ }
            }
            gate.Release();
        }
    }

    private static OnnxExportOutcome Refused(OnnxExportRefusalKind kind, string message)
        => new(null, new OnnxExportRefusal(kind, message));

    private static OnnxLabelInfo LabelInfo(
        string task, string? labelColumn, ExperimentData experiment, PredictionEngine.InferenceInput input, OnnxExportResult result,
        OnnxDecisionRecipe recipe)
    {
        switch (task)
        {
            case "binary-classification":
                {
                    var vocabulary = BinaryLabelVocabulary.Of(experiment.Config?.InputSchema);
                    return new OnnxLabelInfo
                    {
                        Column = labelColumn,
                        Classes = vocabulary is { } v ? [v.Negative, v.Positive] : null,
                        Decision = recipe.Describe(),
                    };
                }
            case "multiclass-classification":
                {
                    IReadOnlyList<string>? classes = null;
                    try
                    {
                        var scored = input.Model.GetOutputSchema(input.Data.Schema);
                        if (scored.GetColumnOrNull("Score") is { } score && score.Type is Microsoft.ML.Data.VectorDataViewType vector)
                            classes = MulticlassScoreVocabulary.Of(score, vector.Size);
                    }
                    catch (Exception)
                    {
                        // No names is an honest answer; the manifest says so by leaving them out.
                    }
                    // The pipeline usually maps the key back to the class name; say what the graph gives.
                    var predicted = result.Outputs.FirstOrDefault(o => o.Column == "PredictedLabel");
                    var predictedReads = predicted?.ElementType == "string"
                        ? "PredictedLabel is the name of the highest-scoring class"
                        : "PredictedLabel is the 1-based index of the highest-scoring class in Classes (0 = none)";
                    return new OnnxLabelInfo
                    {
                        Column = labelColumn,
                        Classes = classes,
                        Decision = recipe.Describe(predictedReads),
                    };
                }
            default:
                return new OnnxLabelInfo { Column = labelColumn, Decision = recipe.Describe() };
        }
    }

    /// <summary>The comparison of a graph with its model, in a line.</summary>
    public static string Describe(OnnxVerificationResult verification)
    {
        var parts = verification.MaxAbsDifference
            .Select(kv => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{kv.Key} max |difference| {kv.Value:G3}"))
            .ToList();
        parts.Add($"{verification.LabelMismatches} label mismatch(es)");
        if (verification.Layered is { } layered)
        {
            // The manifest's decision rule (calibration, threshold) on the graph's outputs, against mloop predict.
            var rule = layered.MaxAbsDifference
                .Select(kv => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{kv.Key} max |difference| {kv.Value:G3}"))
                .Append($"{layered.LabelMismatches} answer mismatch(es)");
            parts.Add($"decision rule vs mloop predict: {string.Join(", ", rule)}");
        }
        return string.Join(", ", parts);
    }

    private static List<string>? PreserveColumns(ExperimentConfig? config)
    {
        if (config == null) return null;
        var columns = new[] { config.GroupColumn, config.UserColumn, config.ItemColumn }
            .Where(c => !string.IsNullOrWhiteSpace(c))
            .Select(c => c!)
            .ToList();
        return columns.Count > 0 ? columns : null;
    }
}
