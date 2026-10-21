using System.CommandLine;
using System.Text.Json;
using Microsoft.ML;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.Display;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Export;
using MLoop.Core.Models;
using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop export onnx - Writes a trained model as an ONNX graph plus a manifest, for runtimes
/// that cannot load ML.NET (Native AOT executables, edge agents, other languages).
/// </summary>
public static class ExportCommand
{
    /// <summary>Tasks whose models the export has been verified for.</summary>
    internal static readonly IReadOnlySet<string> SupportedTasks =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "binary-classification", "multiclass-classification", "regression",
        };

    public static Command Create()
    {
        var command = new Command("export", "Export a trained model to a portable format");
        command.Subcommands.Add(CreateOnnx());
        return command;
    }

    private static Command CreateOnnx()
    {
        var experimentIdArg = new Argument<string?>("experiment-id")
        {
            Description = "Experiment to export (default: the production model)",
            Arity = ArgumentArity.ZeroOrOne
        };

        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')"
        };

        var dataOption = new Option<string?>("--data")
        {
            Description = "A data file with the model's input columns, read for the input schema (default: the experiment's training file)"
        };

        var outputOption = new Option<string?>("--output", "-o")
        {
            Description = "Directory to write model.onnx and its manifest to (default: the experiment's directory)"
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output the result as JSON (machine-readable, non-interactive)"
        };

        var verifyOption = new Option<bool>("--verify")
        {
            Description = $"Run the graph in ONNX Runtime on up to {OnnxVerification.DefaultRows} rows of the data and compare it with the model; a graph that disagrees is not kept (needs: mloop runtime install onnxruntime)"
        };

        var command = new Command("onnx", "Export a model as ONNX, with a manifest describing its inputs, outputs and classes");
        command.Arguments.Add(experimentIdArg);
        command.Options.Add(nameOption);
        command.Options.Add(dataOption);
        command.Options.Add(outputOption);
        command.Options.Add(jsonOption);
        command.Options.Add(verifyOption);

        command.SetAction(parseResult => ExecuteAsync(
            parseResult.GetValue(experimentIdArg),
            parseResult.GetValue(nameOption),
            parseResult.GetValue(dataOption),
            parseResult.GetValue(outputOption),
            parseResult.GetValue(jsonOption),
            parseResult.GetValue(verifyOption)));

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? experimentId, string? modelName, string? dataFile, string? outputDir, bool jsonOutput, bool verify = false)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;
        var tempFiles = new List<string>();

        try
        {
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);

            string projectRoot;
            try
            {
                projectRoot = projectDiscovery.FindRoot();
            }
            catch (InvalidOperationException)
            {
                WriteError("Not inside a MLoop project. Run 'mloop init' to create a new project.", jsonOutput);
                return 1;
            }

            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? ConfigDefaults.DefaultModelName
                : modelName.Trim().ToLowerInvariant();

            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

            if (string.IsNullOrWhiteSpace(experimentId))
            {
                var production = await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None);
                if (production is null)
                {
                    WriteError($"Model '{resolvedModelName}' has no production model. Name an experiment: mloop export onnx <experiment-id>", jsonOutput);
                    return 1;
                }
                experimentId = production.ExperimentId;
            }
            experimentId = experimentId.Trim();

            if (!experimentStore.ExperimentExists(resolvedModelName, experimentId))
            {
                WriteError($"Experiment '{experimentId}' not found for model '{resolvedModelName}'. "
                           + $"Run 'mloop list --name {resolvedModelName}' to see all experiments.", jsonOutput);
                return 1;
            }

            // Checked before any work: a verification that cannot run is a refusal, not a skipped step.
            if (verify && OnnxVerification.Unavailable() is { } unavailable)
            {
                WriteError($"--verify needs ONNX Runtime. {unavailable}", jsonOutput);
                return 1;
            }

            var experiment = await experimentStore.LoadAsync(resolvedModelName, experimentId, CancellationToken.None);
            var task = TaskTypes.Canonical(experiment.Task) ?? experiment.Task ?? "unknown";
            if (!SupportedTasks.Contains(task))
            {
                WriteError($"ONNX export supports {string.Join(", ", SupportedTasks.Order())} models; '{experimentId}' is {task}.", jsonOutput);
                return 1;
            }

            var experimentPath = experimentStore.GetExperimentPath(resolvedModelName, experimentId);
            var modelPath = Path.Combine(experimentPath, ExperimentLayout.ModelFileName);
            if (!File.Exists(modelPath))
            {
                WriteError($"'{experimentId}' has no trained model ({ExperimentLayout.ModelFileName}).", jsonOutput);
                return 1;
            }

            // The input schema comes from data, because the model was saved without one; the
            // training file is the natural default — it has exactly the columns the model was fit on.
            var dataPath = dataFile is { Length: > 0 }
                ? Path.GetFullPath(Path.IsPathRooted(dataFile) ? dataFile : Path.Combine(projectRoot, dataFile))
                : experiment.Config?.DataFile is { Length: > 0 } recorded
                    ? Path.GetFullPath(Path.IsPathRooted(recorded) ? recorded : Path.Combine(projectRoot, recorded))
                    : null;
            if (dataPath is null || !TabularDataFile.Exists(dataPath))
            {
                WriteError(dataPath is null
                    ? "No data file to read the model's input columns from. Pass one: --data <file>"
                    : $"Data file not found: {dataPath}. Pass one with the model's input columns: --data <file>", jsonOutput);
                return 1;
            }
            dataPath = await TabularDataFile.AsCsvAsync(dataPath);

            var labelColumn = string.IsNullOrEmpty(experiment.Config?.LabelColumn) ? null : experiment.Config.LabelColumn;
            var engine = new PredictionEngine();
            var input = engine.LoadInferenceInput(
                modelPath, dataPath, experiment.Config?.InputSchema, labelColumn,
                PreserveColumns(experiment.Config), tempFiles, withDecisionThreshold: false);

            var outputDirectory = outputDir is { Length: > 0 }
                ? Path.GetFullPath(Path.IsPathRooted(outputDir) ? outputDir : Path.Combine(projectRoot, outputDir))
                : experimentPath;
            Directory.CreateDirectory(outputDirectory);
            var onnxPath = Path.Combine(outputDirectory, ExperimentLayout.OnnxModelFileName);
            var manifestPath = Path.Combine(outputDirectory, ExperimentLayout.OnnxManifestFileName);

            OnnxExportResult result;
            var partial = onnxPath + ".partial";
            try
            {
                using (var stream = File.Create(partial))
                {
                    result = OnnxExport.Write(new MLContext(seed: 42), input.Model, input.Data,
                        input.LoaderOptions, input.Header, input.LabelColumn, stream);
                }
                File.Move(partial, onnxPath, overwrite: true);
            }
            finally
            {
                if (File.Exists(partial))
                    File.Delete(partial);
            }

            OnnxVerificationResult? verification = null;
            if (verify)
            {
                verification = OnnxVerification.Compare(input.Model, input.Data, result, onnxPath);
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
                ModelName = resolvedModelName,
                ExperimentId = experimentId,
                Task = task,
                MloopVersion = ErrorSuggestions.GetVersion(),
                ExportedAt = DateTimeOffset.UtcNow,
                Inputs = result.Inputs,
                UnusedInputs = result.UnusedInputs,
                Outputs = result.Outputs,
                Label = LabelInfo(task, labelColumn, experiment, input, result, DecisionThreshold.ReadBeside(modelPath)),
                DecisionThreshold = DecisionThreshold.ReadBeside(modelPath),
                Metrics = experiment.Metrics,
                WeightColumn = experiment.Config?.WeightColumn,
                Portable = experiment.Config?.Portable == true,
                Verification = verification,
            };
            await File.WriteAllTextAsync(manifestPath, manifest.ToJson());

            if (verification is { Passed: false })
            {
                WriteError(
                    $"The exported graph disagrees with the model on {verification.Rows} rows checked "
                    + $"({Describe(verification)}); it was not kept. The comparison is in {manifestPath}.", jsonOutput);
                return 1;
            }

            if (jsonOutput)
            {
                Console.WriteLine(JsonSerializer.Serialize(new
                {
                    Model = resolvedModelName,
                    ExperimentId = experimentId,
                    OnnxPath = onnxPath,
                    ManifestPath = manifestPath,
                    manifest.OnnxSha256,
                    Inputs = result.Inputs,
                    Outputs = result.Outputs,
                    Verification = verification,
                }, new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));
            }
            else
            {
                AnsiConsole.WriteLine();
                AnsiConsole.MarkupLineInterpolated($"[green]>[/] Exported [cyan]{resolvedModelName}[/] / [cyan]{experimentId}[/] ({task}) to ONNX");
                ValueLine.Write("[green]>[/] Model:    ", Path.GetRelativePath(projectRoot, onnxPath));
                ValueLine.Write("[green]>[/] Manifest: ", Path.GetRelativePath(projectRoot, manifestPath));
                foreach (var tensor in result.Inputs)
                    AnsiConsole.MarkupLineInterpolated(
                        $"  input  [cyan]{tensor.Name}[/] {tensor.ElementType}[[{tensor.Width}]] ← {string.Join(", ", tensor.SourceColumns)}");
                foreach (var tensor in result.Outputs)
                    AnsiConsole.MarkupLineInterpolated($"  output [cyan]{tensor.Name}[/] {tensor.ElementType}[[{tensor.Width}]]");
                if (verification is not null)
                    AnsiConsole.MarkupLineInterpolated(
                        $"[green]>[/] Verified on {verification.Rows} rows with {verification.Runtime ?? "ONNX Runtime"}: {Describe(verification)}");
                foreach (var tensor in result.UnusedInputs)
                    AnsiConsole.MarkupLineInterpolated(
                        $"  [grey]input  {tensor.Name} {tensor.ElementType}[[{tensor.Width}]] is not used by any output; feed any value[/]");
                AnsiConsole.WriteLine();
            }

            return 0;
        }
        catch (OnnxExportException ex)
        {
            WriteError(ex.Message, jsonOutput);
            return 1;
        }
        catch (Exception ex)
        {
            if (jsonOutput)
            {
                WriteError(ex.Message, jsonOutput: true);
                return 1;
            }
            ErrorSuggestions.DisplayError(ex, "export");
            return 1;
        }
        finally
        {
            foreach (var file in tempFiles)
            {
                try { if (File.Exists(file)) File.Delete(file); } catch { /* best effort */ }
            }
        }
    }

    private static OnnxLabelInfo LabelInfo(
        string task, string? labelColumn, ExperimentData experiment, PredictionEngine.InferenceInput input, OnnxExportResult result,
        DecisionThreshold? threshold = null)
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
                    // The graph answers at the trainer's own threshold; a chosen one is the consumer's to apply.
                    Decision = threshold is { } t
                        ? string.Create(System.Globalization.CultureInfo.InvariantCulture,
                            $"The model answers the second class when Probability >= {t.Threshold:R} (its decision threshold — apply it to Probability; the graph's PredictedLabel is the trainer's own, Score > 0); Probability is the calibrated chance of the second class")
                        : "PredictedLabel is true (the second class) when Score > 0; Probability is the calibrated chance of the second class",
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
                    Decision = $"Score holds one value per class, in Classes order; {predictedReads}",
                };
            }
            default:
                return new OnnxLabelInfo { Column = labelColumn, Decision = "Score is the predicted value" };
        }
    }

    private static string Describe(OnnxVerificationResult verification)
    {
        var parts = verification.MaxAbsDifference
            .Select(kv => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{kv.Key} max |difference| {kv.Value:G3}"))
            .ToList();
        parts.Add($"{verification.LabelMismatches} label mismatch(es)");
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

    private static void WriteError(string message, bool jsonOutput)
    {
        if (jsonOutput)
        {
            JsonError.Emit(message);
            Console.Error.WriteLine(message);
        }
        else
        {
            ErrorConsole.Out.MarkupLineInterpolated($"[red]Error:[/] {message}");
        }
    }
}
