using System.CommandLine;
using System.Globalization;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Evaluation;
using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.CLI.Commands;

// The phases of `mloop evaluate`, in the order ExecuteAsync runs them. Each shows its own error and
// returns null or an exit code, so ExecuteAsync reads as the sequence.
public static partial class EvaluateCommand
{
    /// <summary>The model an evaluation runs on; <see cref="ExitCode"/> is set when the command ends there (skipped, or an error already shown).</summary>
    private sealed record ModelResolution(string ModelPath, string ExperimentId, ExperimentData? Experiment, int? ExitCode)
    {
        public static ModelResolution Exit(int code) => new("", "", null, code);
    }

    /// <summary>The model named by the experiment id, or production — with its experiment record.</summary>
    private static async Task<ModelResolution> ResolveModelAsync(
        string? experimentId, string resolvedModelName, bool jsonOutput,
        FileSystemManager fileSystem, ExperimentStore experimentStore, ModelRegistry modelRegistry)
    {
        string resolvedModelPath;
        string resolvedExperimentId;
        ExperimentData? experimentData = null;

        if (string.IsNullOrEmpty(experimentId))
        {
            // Use production model for the specified model name
            var productionModel = await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None);

            if (productionModel == null)
            {
                AnsiConsole.MarkupLine($"[yellow]>[/] No production model found for '[cyan]{resolvedModelName}[/]'. Skipping evaluation.");
                AnsiConsole.MarkupLine($"[grey]Tip:[/] Train and promote a model first: [blue]mloop train --name {resolvedModelName}[/]");

                // Skipping is a reported outcome, not a failure — the exit code stays 0 — but it is
                // still an exit, so a --json consumer gets the same document shape with nothing
                // measured rather than an empty stdout it cannot parse. The document has to say
                // that it skipped: parseable is not the same as informative, and "no production
                // model" and "evaluated, metrics unavailable" are different answers.
                if (jsonOutput)
                    EmitJson(resolvedModelName, null, null, null, null, null,
                        skipped: "no-production-model");
                return ModelResolution.Exit(0);
            }

            resolvedExperimentId = productionModel.ExperimentId;
            resolvedModelPath = fileSystem.CombinePath(productionModel.ModelPath, ExperimentLayout.ModelFileName);
            experimentData = await experimentStore.LoadAsync(resolvedModelName, resolvedExperimentId, CancellationToken.None);

            AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
            AnsiConsole.MarkupLine($"[green]>[/] Using production model: [cyan]{resolvedExperimentId}[/]");
        }
        else
        {
            // Use specified experiment
            resolvedExperimentId = experimentId;

            var (candidatePath, refusal) = ExperimentModels.Find(experimentStore, resolvedModelName, resolvedExperimentId);
            if (candidatePath is null)
            {
                ErrorConsole.Error(
                    Markup.Escape(refusal!),
                    $"Run [blue]mloop list --name {resolvedModelName}[/] to see all experiments.");
                return ModelResolution.Exit(1);
            }

            experimentData = await experimentStore.LoadAsync(resolvedModelName, resolvedExperimentId, CancellationToken.None);
            resolvedModelPath = candidatePath;

            AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
            AnsiConsole.MarkupLine($"[green]>[/] Using experiment: [cyan]{resolvedExperimentId}[/]");
        }

        if (!File.Exists(resolvedModelPath))
        {
            ErrorConsole.Error(Markup.Escape(ExperimentModels.NoTrainedModel(resolvedExperimentId ?? "production")));
            return ModelResolution.Exit(1);
        }

        return new ModelResolution(resolvedModelPath, resolvedExperimentId, experimentData, null);
    }

    /// <summary>The test data: the argument, or the project's test set; a directory for directory-based tasks. Null after an error was shown.</summary>
    private static string? ResolveTestData(
        string? testDataFile, ExperimentData? experimentData, string projectRoot, FileSystemManager fileSystem)
    {
        bool isDirectoryBased = DataLoaderFactory.IsDirectoryBased(experimentData?.Task);

        // Resolve test data path (a CSV file, or an image directory for directory-based tasks)
        string resolvedTestDataFile;

        if (string.IsNullOrEmpty(testDataFile))
        {
            if (isDirectoryBased)
            {
                var dir = DatasetDiscovery.FindDirectoryDataset(projectRoot, experimentData?.Task);
                if (dir == null)
                {
                    ErrorConsole.Error(
                        "No test data specified and no image dataset found (datasets/images, datasets/coco, datasets/yolo, or datasets/).",
                        "Pass a directory: mloop evaluate <experiment-id> <dir>");
                    return null;
                }
                resolvedTestDataFile = dir;
                ValueLine.Write("[green]>[/] Auto-detected: ", Path.GetRelativePath(projectRoot, resolvedTestDataFile));
            }
            else
            {
                // Auto-discover datasets/test.csv
                var datasetDiscovery = new DatasetDiscovery(fileSystem);
                var datasets = datasetDiscovery.FindDatasets(projectRoot);

                if (datasets?.TestPath == null)
                {
                    ErrorConsole.Error(
                        "No test data specified and datasets/test.csv not found.",
                        "Create datasets/test.csv or specify a file: mloop evaluate <experiment-id> <test-file>");
                    return null;
                }

                resolvedTestDataFile = datasets.TestPath;
                ValueLine.Write("[green]>[/] Auto-detected: ", Path.GetRelativePath(projectRoot, resolvedTestDataFile));
            }
        }
        else
        {
            // GetFullPath normalizes separators and dot segments: the resolved path is echoed
            // to the user and into --json, where `…\demo\datasets/train.csv` read as two paths.
            resolvedTestDataFile = ProjectPaths.Resolve(testDataFile, projectRoot);

            // Directory-based tasks accept a directory (or a direct COCO .json); CSV tasks need a file.
            bool exists = isDirectoryBased
                ? (Directory.Exists(resolvedTestDataFile) || File.Exists(resolvedTestDataFile))
                : TabularDataFile.Exists(resolvedTestDataFile);
            if (!exists)
            {
                ErrorConsole.Error($"Test data not found: {resolvedTestDataFile}", ErrorConsole.PathNotFoundTip(projectRoot));
                return null;
            }

            ValueLine.Write("[green]>[/] Using test data: ", Path.GetRelativePath(projectRoot, resolvedTestDataFile));
        }

        AnsiConsole.WriteLine();
        return resolvedTestDataFile;
    }

    /// <summary>
    /// A table test set read as a CSV and checked against the schema the model was trained on (a directory of images
    /// is not). Returns the file to read — null after an error was shown.
    /// </summary>
    private static async Task<string?> ValidateTestDataAsync(
        string resolvedTestDataFile, string? records, ExperimentData experimentData, string resolvedModelName,
        string resolvedExperimentId, FileSystemManager fileSystem, ProjectDiscovery projectDiscovery)
    {
        // Before the file becomes a CSV copy: the experiment fingerprinted the file the user gave.
        if (await ModelComparison.TrainingDataWarningAsync(experimentData!, resolvedTestDataFile) is { } leak)
            WarningConsole.Warn(Markup.Escape(leak));

        // A data file in another format (Parquet, Excel, JSON, TSV) is read as a table from here on.
        resolvedTestDataFile = await TabularDataFile.AsCsvAsync(resolvedTestDataFile, records);

        var validator = new SchemaValidator(fileSystem, projectDiscovery);
        var validationResult = await validator.ValidateAsync(resolvedTestDataFile, resolvedModelName, resolvedExperimentId);

        if (!validationResult.IsValid)
        {
            SchemaValidationReport.Refuse(validationResult, "test data");
            return null;
        }

        // Same reason as predict: a column that changed type passes validation, and the
        // measurement it produces is computed on values the loader turned into missing.
        foreach (var warning in validationResult.Warnings)
            WarningConsole.Warn(Markup.Escape(warning));

        AnsiConsole.MarkupLine("[green]>[/] Schema validation passed");
        AnsiConsole.WriteLine();

        return resolvedTestDataFile;
    }

    /// <summary>The metrics table against training, the confusion matrix, and the overfitting finding.</summary>
    private static (bool Overfitting, Dictionary<string, double> TrainingMetrics) ShowEvaluation(
        ExperimentData experimentData, Dictionary<string, double> testMetrics, ConfusionCounts? confusion,
        string? weightColumn, string resolvedModelName, string resolvedExperimentId)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[green]Evaluation Results[/]").LeftJustified());
        AnsiConsole.WriteLine();

        // Display results in table
        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Metric");
        table.AddColumn("Training");
        table.AddColumn("Test");
        table.AddColumn("Difference");

        // Get training metrics for comparison
        var trainingMetrics = experimentData.Metrics ?? new Dictionary<string, double>();

        foreach (var metricName in testMetrics.Keys)
        {
            var testValue = testMetrics[metricName];
            var trainingValue = trainingMetrics.ContainsKey(metricName) ? trainingMetrics[metricName] : 0;
            var difference = testValue - trainingValue;

            // A metric recorded as undefined is the worst value of its direction: shown in words, with no difference to read.
            var undefined = MetricSanitizer.IsUndefinedSentinel(metricName, testValue) || MetricSanitizer.IsUndefinedSentinel(metricName, trainingValue);
            var differenceText = undefined ? "-" : FormatMetricDifference(metricName, difference);

            table.AddRow(
                metricName,
                MetricSanitizer.Display(metricName, trainingValue),
                MetricSanitizer.Display(metricName, testValue),
                differenceText);
        }

        AnsiConsole.Write(table);
        if (weightColumn is not null)
            AnsiConsole.MarkupLineInterpolated($"[grey]Each test row counts by its {weightColumn}.[/]");
        AnsiConsole.WriteLine();
        if (confusion is not null)
            ShowConfusion(confusion, weightColumn is not null);

        // Only what was measured says anything about overfitting; an undefined metric's worst value would read as a collapse.
        var overfitting = DetectOverfitting(experimentData.Task ?? "",
            trainingMetrics.Where(m => !MetricSanitizer.IsUndefinedSentinel(m.Key, m.Value)).ToDictionary(m => m.Key, m => m.Value),
            testMetrics.Where(m => !MetricSanitizer.IsUndefinedSentinel(m.Key, m.Value)).ToDictionary(m => m.Key, m => m.Value));

        // Overfitting warning
        if (overfitting)
        {
            // Names the direction, because that is what the finding is. The previous sentence
            // said "large difference", which was also what the check measured — a model doing
            // better on test than on train got the overfitting warning.
            AnsiConsole.MarkupLine("[yellow]Warning:[/] The model scored notably better on its training data than on this test data — it may be overfitting. The table above has both values.");
            AnsiConsole.WriteLine();
        }

        AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Experiment: [cyan]{resolvedExperimentId}[/]");
        AnsiConsole.MarkupLine("[green]>[/] Evaluation complete!");
        AnsiConsole.WriteLine();

        return (overfitting, trainingMetrics);
    }
}
