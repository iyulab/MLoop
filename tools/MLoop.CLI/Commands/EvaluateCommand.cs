using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop evaluate - Evaluates model performance on test data
/// </summary>
public static class EvaluateCommand
{
    public static Command Create()
    {
        var experimentArg = new Argument<string?>("experiment-id")
        {
            Description = "Experiment ID to evaluate (defaults to production model if omitted)",
            Arity = ArgumentArity.ZeroOrOne
        };

        var testDataArg = new Argument<string?>("test-data")
        {
            Description = "Path to test data file (defaults to datasets/test.csv if omitted)",
            Arity = ArgumentArity.ZeroOrOne
        };

        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')"
        };

        var command = new Command("evaluate", "Evaluate model performance on test data");
        command.Arguments.Add(experimentArg);
        command.Arguments.Add(testDataArg);
        command.Options.Add(nameOption);

        command.SetAction((parseResult) =>
        {
            var experimentId = parseResult.GetValue(experimentArg);
            var testDataFile = parseResult.GetValue(testDataArg);
            var name = parseResult.GetValue(nameOption);
            return ExecuteAsync(experimentId, testDataFile, name);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? experimentId,
        string? testDataFile,
        string? modelName)
    {
        try
        {
            // Initialize components
            var fileSystem = new FileSystemManager();
            var projectDiscovery = new ProjectDiscovery(fileSystem);

            // Find project root
            string projectRoot;
            try
            {
                projectRoot = projectDiscovery.FindRoot();
            }
            catch (InvalidOperationException)
            {
                ErrorConsole.Error("Not inside a MLoop project.");
                AnsiConsole.MarkupLine("Run [blue]mloop init[/] to create a new project.");
                return 1;
            }

            // Resolve model name (defaults to "default")
            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? ConfigDefaults.DefaultModelName
                : modelName.Trim().ToLowerInvariant();

            // Resolve experiment and model path
            string resolvedModelPath;
            string resolvedExperimentId;
            ExperimentData? experimentData = null;

            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

            if (string.IsNullOrEmpty(experimentId))
            {
                // Use production model for the specified model name
                var productionModel = await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None);

                if (productionModel == null)
                {
                    AnsiConsole.MarkupLine($"[yellow]>[/] No production model found for '[cyan]{resolvedModelName}[/]'. Skipping evaluation.");
                    AnsiConsole.MarkupLine($"[grey]Tip:[/] Train and promote a model first: [blue]mloop train --name {resolvedModelName}[/]");
                    return 0;
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

                if (!experimentStore.ExperimentExists(resolvedModelName, resolvedExperimentId))
                {
                    ErrorConsole.Error($"Experiment not found: {resolvedExperimentId} for model '{resolvedModelName}'");
                    ErrorConsole.Tip($"Run [blue]mloop list --name {resolvedModelName}[/] to see all experiments.");
                    return 1;
                }

                experimentData = await experimentStore.LoadAsync(resolvedModelName, resolvedExperimentId, CancellationToken.None);
                var experimentPath = experimentStore.GetExperimentPath(resolvedModelName, resolvedExperimentId);
                resolvedModelPath = fileSystem.CombinePath(experimentPath, ExperimentLayout.ModelFileName);

                AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
                AnsiConsole.MarkupLine($"[green]>[/] Using experiment: [cyan]{resolvedExperimentId}[/]");
            }

            if (!File.Exists(resolvedModelPath))
            {
                ErrorConsole.Error($"Model file not found: {resolvedModelPath}");
                return 1;
            }

            // Directory-based tasks (image classification, object detection) evaluate over an image
            // directory — object detection scored by mAP, image classification by multiclass accuracy
            // — so they bypass the CSV-file test-data resolution and CSV schema validation below.
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
                        ErrorConsole.Error("No test data specified and no image dataset found (datasets/images, datasets/coco, datasets/yolo, or datasets/).");
                        ErrorConsole.Tip("Pass a directory: mloop evaluate <experiment-id> <dir>");
                        return 1;
                    }
                    resolvedTestDataFile = dir;
                    AnsiConsole.MarkupLine($"[green]>[/] Auto-detected: [cyan]{Path.GetRelativePath(projectRoot, resolvedTestDataFile)}[/]");
                }
                else
                {
                    // Auto-discover datasets/test.csv
                    var datasetDiscovery = new DatasetDiscovery(fileSystem);
                    var datasets = datasetDiscovery.FindDatasets(projectRoot);

                    if (datasets?.TestPath == null)
                    {
                        ErrorConsole.Error("No test data specified and datasets/test.csv not found.");
                        ErrorConsole.Tip("Create datasets/test.csv or specify a file: mloop evaluate <experiment-id> <test-file>");
                        return 1;
                    }

                    resolvedTestDataFile = datasets.TestPath;
                    AnsiConsole.MarkupLine($"[green]>[/] Auto-detected: [cyan]{Path.GetRelativePath(projectRoot, resolvedTestDataFile)}[/]");
                }
            }
            else
            {
                resolvedTestDataFile = Path.IsPathRooted(testDataFile)
                    ? testDataFile
                    : Path.Combine(projectRoot, testDataFile);

                // Directory-based tasks accept a directory (or a direct COCO .json); CSV tasks need a file.
                bool exists = isDirectoryBased
                    ? (Directory.Exists(resolvedTestDataFile) || File.Exists(resolvedTestDataFile))
                    : File.Exists(resolvedTestDataFile);
                if (!exists)
                {
                    ErrorConsole.Error($"Test data not found: {resolvedTestDataFile}");
                    return 1;
                }

                AnsiConsole.MarkupLine($"[green]>[/] Using test data: [cyan]{Path.GetRelativePath(projectRoot, resolvedTestDataFile)}[/]");
            }

            AnsiConsole.WriteLine();

            // Validate schema before evaluation (CSV-column validation; not applicable to image directories)
            if (!isDirectoryBased)
            {
                var validator = new SchemaValidator(fileSystem, projectDiscovery);
                var validationResult = await validator.ValidateAsync(resolvedModelPath, resolvedTestDataFile, resolvedModelName, resolvedExperimentId);

                if (!validationResult.IsValid)
                {
                    AnsiConsole.MarkupLine("[red]Schema Validation Failed:[/]");
                    AnsiConsole.WriteLine();

                    if (!string.IsNullOrEmpty(validationResult.ErrorMessage))
                    {
                        AnsiConsole.MarkupLine($"[yellow]Warning: {validationResult.ErrorMessage}[/]");
                        AnsiConsole.WriteLine();
                    }

                    if (validationResult.MissingColumns.Any())
                    {
                        AnsiConsole.MarkupLine("[red]Missing columns in test data:[/]");
                        foreach (var col in validationResult.MissingColumns)
                        {
                            AnsiConsole.MarkupLine($"  [grey]-[/] {col}");
                        }
                        AnsiConsole.WriteLine();
                    }

                    return 1;
                }

                AnsiConsole.MarkupLine("[green]>[/] Schema validation passed");
                AnsiConsole.WriteLine();
            }

            // Perform evaluation with progress indicator
            Dictionary<string, double>? testMetrics = null;

            await AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .StartAsync("[yellow]Evaluating model...[/]", async ctx =>
                {
                    var evaluationEngine = new EvaluationEngine();

                    testMetrics = await evaluationEngine.EvaluateAsync(
                        resolvedModelPath,
                        resolvedTestDataFile,
                        experimentData!.Config.LabelColumn,
                        experimentData.Task,
                        CancellationToken.None,
                        experimentData.Config.InputSchema,
                        experimentData.Config.GroupColumn,
                        experimentData.Config.UserColumn,
                        experimentData.Config.ItemColumn);

                    ctx.Status("[green]Evaluation complete![/]");
                });

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
            var trainingMetrics = experimentData!.Metrics ?? new Dictionary<string, double>();

            foreach (var metricName in testMetrics!.Keys)
            {
                var testValue = testMetrics[metricName];
                var trainingValue = trainingMetrics.ContainsKey(metricName) ? trainingMetrics[metricName] : 0;
                var difference = testValue - trainingValue;

                var differenceText = FormatMetricDifference(metricName, difference);

                table.AddRow(
                    metricName,
                    trainingValue.ToString("F4"),
                    testValue.ToString("F4"),
                    differenceText);
            }

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();

            // Overfitting warning
            if (DetectOverfitting(experimentData.Task ?? "", trainingMetrics, testMetrics!))
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] Large metric difference detected between training and test. Model may be overfitting.");
                AnsiConsole.WriteLine();
            }

            AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
            AnsiConsole.MarkupLine($"[green]>[/] Experiment: [cyan]{resolvedExperimentId}[/]");
            AnsiConsole.MarkupLine("[green]>[/] Evaluation complete!");
            AnsiConsole.WriteLine();

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "evaluate");
            return 1;
        }
    }

    internal static bool IsLowerBetterMetric(string metricName)
        => MLoop.Core.Evaluation.MetricDirection.IsLowerBetter(metricName);

    internal static string FormatMetricDifference(string metricName, double difference)
    {
        if (IsLowerBetterMetric(metricName))
        {
            return difference <= 0
                ? $"[green]{difference:F4}[/]"
                : $"[red]+{difference:F4}[/]";
        }

        return difference >= 0
            ? $"[green]+{difference:F4}[/]"
            : $"[red]{difference:F4}[/]";
    }

    internal const double OverfittingThreshold = 0.1;

    internal static bool DetectOverfitting(
        string task,
        Dictionary<string, double> trainingMetrics,
        Dictionary<string, double> testMetrics)
    {
        // F-25: the task is stored as the CLI-canonical string ("binary-classification",
        // "multiclass-classification", "regression"), and each task's primary metric key differs
        // (regression=r_squared, binary=accuracy, multiclass=macro_accuracy). The previous code
        // matched only the literal "classification" with the "accuracy" key, so overfitting
        // detection was silently dead for every real classification model — doubly so for multiclass
        // (wrong task string AND wrong metric key). "classification" is kept as a legacy binary alias.
        var metricKey = task.ToLowerInvariant() switch
        {
            "regression" => "r_squared",
            "binary-classification" or "classification" => "accuracy",
            "multiclass-classification" => "macro_accuracy",
            _ => null
        };

        if (metricKey != null
            && trainingMetrics.TryGetValue(metricKey, out var trainValue)
            && testMetrics.TryGetValue(metricKey, out var testValue))
        {
            return Math.Abs(trainValue - testValue) > OverfittingThreshold;
        }

        return false;
    }
}
