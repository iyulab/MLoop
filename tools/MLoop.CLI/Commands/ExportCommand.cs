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
using MLoop.Core.Runtime;
using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop export onnx - Writes a trained model as an ONNX graph plus a manifest, for runtimes
/// that cannot load ML.NET (Native AOT executables, edge agents, other languages).
/// </summary>
public static class ExportCommand
{
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
            Description = $"Run the graph in ONNX Runtime on up to {OnnxVerification.DefaultRows} rows of the data and compare it with the model — and the manifest's decision rule (calibration, threshold) with mloop predict's answer; a graph that disagrees is not kept (needs: mloop runtime install onnxruntime)"
        };

        var annotationOption = new Option<string[]>("--annotation")
        {
            Description = $"Write key=value into the manifest's annotations (repeatable; up to {OnnxAnnotations.MaxPairs} pairs) — opaque text mloop does not read, such as the version of the preprocessing that feeds the graph. The manifest's SHA-256 covers them",
            AllowMultipleArgumentsPerToken = false,
        };

        var command = new Command("onnx", "Export a model as ONNX, with a manifest describing its inputs, outputs and classes");
        command.Arguments.Add(experimentIdArg);
        command.Options.Add(nameOption);
        command.Options.Add(dataOption);
        command.Options.Add(outputOption);
        command.Options.Add(jsonOption);
        command.Options.Add(verifyOption);
        command.Options.Add(annotationOption);

        command.SetAction(parseResult => ExecuteAsync(
            parseResult.GetValue(experimentIdArg),
            parseResult.GetValue(nameOption),
            parseResult.GetValue(dataOption),
            parseResult.GetValue(outputOption),
            parseResult.GetValue(jsonOption),
            parseResult.GetValue(verifyOption),
            parseResult.GetValue(annotationOption)));

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? experimentId, string? modelName, string? dataFile, string? outputDir, bool jsonOutput, bool verify = false, string[]? annotation = null)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;
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
                ErrorConsole.Error(Markup.Escape("Not inside a MLoop project. Run 'mloop init' to create a new project."));
                return 1;
            }

            var (annotations, annotationError) = ParseAnnotations(annotation);
            if (annotationError is not null)
            {
                ErrorConsole.Error(Markup.Escape(annotationError));
                return 1;
            }

            var resolvedModelName = MLoop.Core.Storage.ModelName.Resolve(modelName);

            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

            if (string.IsNullOrWhiteSpace(experimentId))
            {
                var production = await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None);
                if (production is null)
                {
                    ErrorConsole.Error(Markup.Escape($"Model '{resolvedModelName}' has no production model. Name an experiment: mloop export onnx <experiment-id>"));
                    return 1;
                }
                experimentId = production.ExperimentId;
            }
            experimentId = experimentId.Trim();

            var outcome = await new OnnxExporter(experimentStore).ExportAsync(
                new OnnxExportRequest(projectRoot, resolvedModelName, experimentId, dataFile, outputDir, verify, Annotations: annotations),
                CancellationToken.None);
            if (outcome.Refusal is { } refused)
            {
                ErrorConsole.Error(Markup.Escape(Refusal(refused, resolvedModelName)));
                return 1;
            }

            var export = outcome.Output!;
            var (task, result, verification, onnxPath, manifestPath) = (export.Task, export.Result, export.Verification, export.OnnxPath, export.ManifestPath);

            if (jsonOutput)
            {
                Console.WriteLine(JsonSerializer.Serialize(new
                {
                    Model = resolvedModelName,
                    ExperimentId = experimentId,
                    OnnxPath = onnxPath,
                    ManifestPath = manifestPath,
                    export.Manifest.OnnxSha256,
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
                        $"[green]>[/] Verified on {verification.Rows} rows with {verification.Runtime ?? "ONNX Runtime"}: {OnnxExporter.Describe(verification)}");
                foreach (var tensor in result.UnusedInputs)
                    AnsiConsole.MarkupLineInterpolated(
                        $"  [grey]input  {tensor.Name} {tensor.ElementType}[[{tensor.Width}]] is not used by any output; feed any value[/]");
                AnsiConsole.WriteLine();
            }

            return 0;
        }
        catch (OnnxExportException ex)
        {
            ErrorConsole.Error(Markup.Escape(ex.Message));
            return 1;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "export");
            return 1;
        }
    }

    /// <summary>The <c>--annotation key=value</c> pairs, or what is wrong with one: the key is up to the first '=', the value is the rest.</summary>
    private static (Dictionary<string, string>? Pairs, string? Error) ParseAnnotations(string[]? given)
    {
        if (given is not { Length: > 0 })
            return (null, null);
        var pairs = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var item in given)
        {
            var at = item.IndexOf('=');
            if (at <= 0)
                return (null, $"--annotation '{item}' is not key=value.");
            var key = item[..at];
            if (!pairs.TryAdd(key, item[(at + 1)..]))
                return (null, $"--annotation names '{key}' twice.");
        }
        return (pairs, null);
    }

    /// <summary>The refusal in this command's words: what went wrong, then the option or command that changes it.</summary>
    private static string Refusal(OnnxExportRefusal refusal, string modelName) => refusal.Kind switch
    {
        OnnxExportRefusalKind.ExperimentNotFound => $"{refusal.Message} Run 'mloop list --name {modelName}' to see all experiments.",
        OnnxExportRefusalKind.VerifyUnavailable => $"--verify needs ONNX Runtime. {refusal.Message}",
        OnnxExportRefusalKind.NoDataFile when refusal.Message.StartsWith("Data file not found", StringComparison.Ordinal)
            => $"{refusal.Message} Pass one with the model's input columns: --data <file>",
        OnnxExportRefusalKind.NoDataFile => $"{refusal.Message} Pass one: --data <file>",
        OnnxExportRefusalKind.GraphDisagrees => $"{refusal.Message} The comparison is in {refusal.ManifestPath}.",
        _ => refusal.Message,
    };
}
