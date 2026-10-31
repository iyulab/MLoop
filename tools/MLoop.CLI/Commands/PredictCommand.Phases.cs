using System.CommandLine;
using System.Globalization;
using System.Text.Json;
using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Preprocessing;
using MLoop.Core.Storage;
using MLoop.Core.Training;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using MLoop.Extensibility.Preprocessing;
using Spectre.Console;
using static MLoop.Core.Prediction.CategoricalMapper;

namespace MLoop.CLI.Commands;

// The phases of `mloop predict`, in the order ExecuteAsync runs them. Each shows its own error
// and returns null (or an exit code), so ExecuteAsync reads as the sequence.
public static partial class PredictCommand
{
    /// <summary>The model a prediction runs on and what its experiment record says it was trained on.</summary>
    private sealed record PredictModel(
        string ModelPath,
        string? ExperimentId,
        InputSchemaInfo? TrainedSchema,
        string? LabelColumn,
        string? TaskType,
        List<string>? PreserveColumns,
        RegressionInterval? Interval);

    /// <summary>The model named by <c>--experiment</c>, <c>--model</c> or production — null after an error was shown.</summary>
    private static async Task<PredictModel?> ResolveModelAsync(
        string? modelPath, string? experiment, string resolvedModelName, string projectRoot,
        FileSystemManager fileSystem, ProjectDiscovery projectDiscovery)
    {
        // Initialize stores
        var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
        var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

        // Resolve model path (Convention: use production model if not specified)
        string resolvedModelPath;
        string? experimentId = null;
        InputSchemaInfo? trainedSchema = null;
        string? configLabelColumn = null;
        string? taskType = null;
        // group/user/item columns must stay individually addressable at predict time so the
        // ranking/recommendation model's key transforms can find them (see ApplyColumnPreservation).
        List<string>? preserveColumns = null;
        // ② regression wave: the stored conformal band, surfaced as [ScoreLowerBound, ScoreUpperBound]
        // in the CSV for regression models (same single source the serve /predict path uses).
        RegressionInterval? interval = null;

        // What the model was trained on (schema · label · task · band) comes from its experiment
        // record, whichever way the model was named. A record that does not load leaves these
        // null and the predict path continues without categorical preprocessing.
        async Task<bool> ReadExperimentAsync(string id)
        {
            try
            {
                var experimentData = await experimentStore.LoadAsync(resolvedModelName, id, CancellationToken.None);
                trainedSchema = experimentData?.Config?.InputSchema;
                configLabelColumn = string.IsNullOrEmpty(experimentData?.Config?.LabelColumn) ? null : experimentData.Config.LabelColumn;
                taskType = experimentData?.Task;
                preserveColumns = BuildPreserveColumns(experimentData?.Config);
                interval = RegressionInterval.FromMetrics(experimentData?.Task, experimentData?.Metrics);
                return true;
            }
            catch
            {
                return false;
            }
        }

        if (!string.IsNullOrEmpty(experiment) && !string.IsNullOrEmpty(modelPath))
        {
            ErrorConsole.Error(
                "--experiment and --model both name the model to predict with.",
                "Give one: --experiment <exp-id> for an experiment of this project, --model <path> for a model file.");
            return null;
        }

        if (!string.IsNullOrEmpty(experiment))
        {
            // A shadow prediction: the candidate answers the same rows production would, and
            // production stays where it is. Logged rows carry this experiment's id, so drift
            // (which reads production's rows only) and feedback keep the two apart.
            experimentId = experiment.Trim();
            var (candidatePath, refusal) = ExperimentModels.Find(experimentStore, resolvedModelName, experimentId);
            if (candidatePath is null)
            {
                ErrorConsole.Error(
                    Markup.Escape(refusal!),
                    $"Run [blue]mloop list --name {resolvedModelName}[/] to see all experiments.");
                return null;
            }
            resolvedModelPath = candidatePath;

            var production = await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None);
            var isProduction = string.Equals(production?.ExperimentId, experimentId, StringComparison.OrdinalIgnoreCase);

            AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
            AnsiConsole.MarkupLine(isProduction
                ? $"[green]>[/] Using experiment: [cyan]{Markup.Escape(experimentId)}[/] (the production model)"
                : $"[green]>[/] Using experiment: [cyan]{Markup.Escape(experimentId)}[/] (not production — production is {(production is null ? "not set" : $"[cyan]{production.ExperimentId}[/]")})");

            await ReadExperimentAsync(experimentId);
        }
        else if (string.IsNullOrEmpty(modelPath))
        {
            // Auto-load production model for the specified model name
            var productionModel = await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None);

            if (productionModel == null)
            {
                ErrorConsole.Error(
                    $"No production model found for '[cyan]{resolvedModelName}[/]'.",
                    $"Train and promote a model first: [blue]mloop train --name {resolvedModelName}[/] — or predict with a candidate: [blue]--experiment <exp-id>[/]");
                return null;
            }

            resolvedModelPath = fileSystem.CombinePath(productionModel.ModelPath, ExperimentLayout.ModelFileName);
            experimentId = productionModel.ExperimentId;

            AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
            AnsiConsole.MarkupLine($"[green]>[/] Using production model: [cyan]{productionModel.ExperimentId}[/]");

            await ReadExperimentAsync(experimentId);
        }
        else
        {
            // Use specified model path
            resolvedModelPath = ProjectPaths.Resolve(modelPath, projectRoot);

            if (!File.Exists(resolvedModelPath))
            {
                ErrorConsole.Error($"Model file not found: {resolvedModelPath}");
                return null;
            }

            AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
            ValueLine.Write("[green]>[/] Using model file: ", Path.GetRelativePath(projectRoot, resolvedModelPath));

            // Try to infer experiment ID from path
            var modelDir = Path.GetDirectoryName(resolvedModelPath);
            if (modelDir != null)
            {
                var possibleExpId = Path.GetFileName(modelDir);

                if (experimentStore.ExperimentExists(resolvedModelName, possibleExpId)
                    && await ReadExperimentAsync(possibleExpId))
                {
                    experimentId = possibleExpId;
                }
            }

            // A model file carries no record of the columns it was trained on - that sits in its experiment folder. Without it the
            // data is scored as it reads, and a table whose columns read differently from training fails with a message about the data.
            if (experimentId is null)
                WarningConsole.Warn($"'{Path.GetFileName(resolvedModelPath)}' is not in an experiment folder of model '{resolvedModelName}', so the " +
                    "columns it was trained on are unknown and the data is scored as it reads (a categorical column or a changed column type can fail or " +
                    "score wrongly). Predict with --experiment <exp-id> or keep the model in its experiment folder.");
        }

        return new PredictModel(resolvedModelPath, experimentId, trainedSchema, configLabelColumn, taskType, preserveColumns, interval);
    }

    /// <summary>The file to predict over: the argument, or datasets/predict.csv; a folder of images becomes its table. Null after an error was shown.</summary>
    private static string? ResolveDataFile(string? dataFile, string projectRoot, string? taskType, FileSystemManager fileSystem)
    {
        // Resolve data file path (Convention: datasets/predict.csv)
        string resolvedDataFile;

        if (string.IsNullOrEmpty(dataFile))
        {
            // Auto-discover datasets/predict.csv
            var datasetDiscovery = new DatasetDiscovery(fileSystem);
            var datasets = datasetDiscovery.FindDatasets(projectRoot);

            if (datasets?.PredictPath == null)
            {
                ErrorConsole.Error(
                    "No data file specified and datasets/predict.csv not found.",
                    "Create datasets/predict.csv or specify a file: mloop predict <data-file>");
                return null;
            }

            resolvedDataFile = datasets.PredictPath;
            ValueLine.Write("[green]>[/] Auto-detected: ", Path.GetRelativePath(projectRoot, resolvedDataFile));
        }
        else
        {
            // Validate explicit data file
            resolvedDataFile = ProjectPaths.Resolve(dataFile, projectRoot);

            if (Directory.Exists(resolvedDataFile)
                && string.Equals(taskType, "image-classification", StringComparison.OrdinalIgnoreCase))
            {
                // A folder of pictures is what an image model is asked about; the table it scores
                // (one ImagePath per row) is built here rather than by the user.
                var images = ImageDirectoryLoader.ListImages(resolvedDataFile);
                if (images.Count == 0)
                {
                    ErrorConsole.Error(Markup.Escape(
                        $"No images found under {resolvedDataFile} (looked for {ImageDirectoryLoader.SupportedExtensionList}, in subfolders too)."));
                    return null;
                }
                ValueLine.Write("[green]>[/] Images: ", $"{images.Count:N0} under {Path.GetFileName(Path.TrimEndingDirectorySeparator(resolvedDataFile))}");
                resolvedDataFile = ImageDirectoryLoader.WriteImageTable(images);
            }
            else if (!TabularDataFile.Exists(resolvedDataFile))
            {
                var cause = Markup.Escape(BuildMissingDataFileMessage(resolvedDataFile));
                if (Directory.Exists(resolvedDataFile))
                {
                    // A directory at this path is a wrong-type mismatch, not a wrong-location
                    // one — the path resolution rule below doesn't explain what to do about it.
                    ErrorConsole.Error(cause);
                }
                else
                {
                    ErrorConsole.Error(cause, ErrorConsole.PathNotFoundTip(projectRoot));
                }
                return null;
            }
        }

        return resolvedDataFile;
    }

    /// <summary>Where the predictions are written; the directory is created.</summary>
    private static async Task<string> ResolveOutputPathAsync(
        string? output, string projectRoot, string resolvedModelName, FileSystemManager fileSystem)
    {
        // Resolve output path (Convention: predictions/<model>-predictions-<timestamp>.csv)
        string resolvedOutputPath;

        if (string.IsNullOrEmpty(output))
        {
            var predictionsDir = fileSystem.CombinePath(projectRoot, ProjectLayout.PredictionsDirectory);
            await fileSystem.CreateDirectoryAsync(predictionsDir, CancellationToken.None);

            resolvedOutputPath = fileSystem.CombinePath(
                predictionsDir,
                ProjectLayout.PredictionFileName(
                    resolvedModelName, ProjectLayout.PredictionKind.Rows, DateTimeOffset.Now));
        }
        else
        {
            resolvedOutputPath = ProjectPaths.Resolve(output, projectRoot);

            // Create output directory if needed
            var outputDir = Path.GetDirectoryName(resolvedOutputPath);
            if (!string.IsNullOrEmpty(outputDir) && !Directory.Exists(outputDir))
            {
                await fileSystem.CreateDirectoryAsync(outputDir, CancellationToken.None);
            }
        }

        return resolvedOutputPath;
    }

    /// <summary>
    /// The rows the model reads: any format as a table, UTF-8, multi-line fields flattened, the schema validated against
    /// the training data, and the project's preprocessing steps applied. Null after an error was shown.
    /// </summary>
    private static async Task<string?> PrepareRowsAsync(
        string resolvedDataFile, string? records, string resolvedModelName, string? experimentId,
        FileSystemManager fileSystem, ProjectDiscovery projectDiscovery)
    {
        // A data file in another format (Parquet, Excel, JSON, TSV) is read as a table from here on.
        resolvedDataFile = await TabularDataFile.AsCsvAsync(resolvedDataFile, records);

        // Ensure UTF-8 encoding for prediction data (same as training)
        var (convertedPath, detection) = EncodingDetector.ConvertToUtf8WithBom(resolvedDataFile);
        if (detection.WasConverted && detection.EncodingName != "UTF-8")
        {
            AnsiConsole.MarkupLineInterpolated($"[green]>[/] Converted {detection.EncodingName} → UTF-8: [cyan]{Path.GetFileName(resolvedDataFile)}[/]");
            resolvedDataFile = convertedPath;
        }

        // Flatten multi-line quoted fields in data rows (RFC 4180 multiline support)
        resolvedDataFile = CsvDataLoader.FlattenMultiLineQuotedFields(resolvedDataFile, CoreNarration.Sink);

        // Flatten multi-line quoted headers (same as training)
        resolvedDataFile = CsvDataLoader.FlattenMultiLineHeaders(resolvedDataFile, CoreNarration.Sink);

        AnsiConsole.WriteLine();

        // Validate schema before prediction
        var validator = new SchemaValidator(fileSystem, projectDiscovery);
        var validationResult = await validator.ValidateAsync(resolvedDataFile, resolvedModelName, experimentId);

        if (!validationResult.IsValid)
        {
            SchemaValidationReport.Refuse(validationResult, "prediction data");
            return null;
        }

        // Warnings survive a passing validation, deliberately: a column that changed type is
        // not a missing column, and the failure it causes downstream names a feature vector's
        // width rather than the column. WarningConsole so a --json consumer gets the same
        // finding as a `warning` event, not only the human at the terminal.
        foreach (var warning in validationResult.Warnings)
            WarningConsole.Warn(Markup.Escape(warning));

        AnsiConsole.MarkupLine("[green]>[/] Schema validation passed");
        AnsiConsole.WriteLine();

        // Apply YAML-defined preprocessing pipeline (same as training)
        try
        {
            var configLoader = new ConfigLoader(fileSystem, projectDiscovery);
            var config = await configLoader.LoadUserConfigAsync();

            if (config.Models.TryGetValue(resolvedModelName, out var modelDef) &&
                modelDef.Prep is { Count: > 0 })
            {
                AnsiConsole.MarkupLine($"[green]>[/] Applying {modelDef.Prep.Count} preprocessing step(s)...");

                var prepLogger = new ConsoleLogger();
                var pipelineExecutor = new DataPipelineExecutor(prepLogger);
                var prepOutputPath = Path.Combine(
                    Path.GetDirectoryName(resolvedDataFile)!,
                    $"{Path.GetFileNameWithoutExtension(resolvedDataFile)}_prep{Path.GetExtension(resolvedDataFile)}");

                resolvedDataFile = await pipelineExecutor.ExecuteAsync(
                    resolvedDataFile, modelDef.Prep, prepOutputPath);

                AnsiConsole.MarkupLine("[green]>[/] Preprocessing complete");
                AnsiConsole.WriteLine();
            }
        }
        catch (InvalidOperationException)
        {
            // Not in a project or no config — skip preprocessing
        }

        return resolvedDataFile;
    }

    /// <summary>
    /// <c>--json</c>: structured predictions through the shared PredictionService (the same path as the /predict API),
    /// written to stdout with the conformal band and normalized confidence. Bypasses the CSV output.
    /// </summary>
    private static async Task<int> PredictAsJsonAsync(
        string resolvedDataFile, PredictModel model, bool logPredictions, string projectRoot, string resolvedModelName)
    {
        var (resolvedModelPath, experimentId, trainedSchema, configLabelColumn, taskType, _, interval) = model;

        if (trainedSchema is null || string.IsNullOrEmpty(taskType))
        {
            ErrorConsole.Error("--json requires a trained schema/task (train and promote a model first).");
            return 1;
        }

        var jsonRows = PredictionEngine.LoadCsvAsRows(
            resolvedDataFile, trainedSchema, LabelColumnToExcludeFromRows(taskType, configLabelColumn));
        var jsonResult = new PredictionEngine().PredictWithService(
            jsonRows, trainedSchema, resolvedModelPath, taskType, configLabelColumn, interval);

        // --log under --json logs as `POST /predict` does, and each row carries the id it was logged
        // under — the id `mloop feedback add` takes. Before, this branch returned before logging.
        var jsonPredictions = jsonResult.Rows;
        if (logPredictions && jsonPredictions.Count == jsonRows.Length)
        {
            jsonPredictions = [.. await ServedPredictionLog.LogAsync(
                new FilePredictionLogger(projectRoot), resolvedModelName, experimentId ?? "unknown",
                jsonRows, jsonPredictions)];
        }
        else if (logPredictions)
        {
            WarningConsole.Warn($"These predictions were not logged: {jsonPredictions.Count} answers for {jsonRows.Length} rows.");
        }

        var payload = new
        {
            model = resolvedModelName,
            // Which model answered: the consumer comparing a candidate with production keys
            // the two documents by it.
            experimentId,
            // The same model beyond this project: an experiment id is unique only inside it.
            modelSha256 = ModelFingerprint.Sha256(resolvedModelPath),
            task = jsonResult.TaskType,
            count = jsonPredictions.Count,
            predictions = jsonPredictions,
            warnings = JsonOutputScope.Warnings is { } reported ? [.. reported, .. jsonResult.Warnings ?? []] : jsonResult.Warnings,
        };
        Console.Out.WriteLine(JsonSerializer.Serialize(payload, PredictJsonOptions));
        return 0;
    }
}
