using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Evaluation;
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

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Emit the evaluation result as JSON to stdout instead of the human table. " +
                          "Progress and narration go to stderr."
        };

        var command = new Command("evaluate", "Evaluate model performance on test data");
        command.Arguments.Add(experimentArg);
        command.Arguments.Add(testDataArg);
        command.Options.Add(nameOption);
        command.Options.Add(jsonOption);

        command.SetAction((parseResult) =>
        {
            var experimentId = parseResult.GetValue(experimentArg);
            var testDataFile = parseResult.GetValue(testDataArg);
            var name = parseResult.GetValue(nameOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteAsync(experimentId, testDataFile, name, json);
        });

        return command;
    }

    /// <summary>
    /// The one place this command's <c>--json</c> document is shaped, so the evaluated and the
    /// nothing-to-evaluate exits cannot describe themselves with different keys.
    /// </summary>
    /// <param name="skipped">
    /// Why no evaluation was performed, or <see langword="null"/> when one was. Without it the two
    /// outcomes are the same document: a run that measured nothing and a run that never started both
    /// serialize as all-null metrics, and the exit code is 0 in both cases because skipping is a
    /// reported outcome rather than a failure. The human is told which happened; this is the field
    /// that tells a consumer the same thing.
    /// </param>
    private static void EmitJson(
        string model,
        string? experimentId,
        string? testDataFile,
        object? trainingMetrics,
        object? testMetrics,
        bool? possibleOverfitting,
        string? skipped = null)
    {
        var payload = new
        {
            Model = model,
            Skipped = skipped,
            ExperimentId = experimentId,
            TestDataFile = testDataFile,
            TrainingMetrics = trainingMetrics,
            TestMetrics = testMetrics,
            PossibleOverfitting = possibleOverfitting
        };

        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(payload,
            new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
            }));
    }

    private static async Task<int> ExecuteAsync(
        string? experimentId,
        string? testDataFile,
        string? modelName,
        bool jsonOutput = false)
    {
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

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
                ErrorConsole.Error(
                    ProjectDiscovery.NotInsideProjectCause,
                    ProjectDiscovery.NotInsideProjectGuidance);
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

                    // Skipping is a reported outcome, not a failure — the exit code stays 0 — but it is
                    // still an exit, so a --json consumer gets the same document shape with nothing
                    // measured rather than an empty stdout it cannot parse. The document has to say
                    // that it skipped: parseable is not the same as informative, and "no production
                    // model" and "evaluated, metrics unavailable" are different answers.
                    if (jsonOutput)
                        EmitJson(resolvedModelName, null, null, null, null, null,
                            skipped: "no-production-model");
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
                    ErrorConsole.Error(
                        $"Experiment not found: {resolvedExperimentId} for model '{resolvedModelName}'",
                        $"Run [blue]mloop list --name {resolvedModelName}[/] to see all experiments.");
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
                        ErrorConsole.Error(
                            "No test data specified and no image dataset found (datasets/images, datasets/coco, datasets/yolo, or datasets/).",
                            "Pass a directory: mloop evaluate <experiment-id> <dir>");
                        return 1;
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
                        return 1;
                    }

                    resolvedTestDataFile = datasets.TestPath;
                    ValueLine.Write("[green]>[/] Auto-detected: ", Path.GetRelativePath(projectRoot, resolvedTestDataFile));
                }
            }
            else
            {
                // GetFullPath normalizes separators and dot segments: the resolved path is echoed
                // to the user and into --json, where `…\demo\datasets/train.csv` read as two paths.
                resolvedTestDataFile = Path.GetFullPath(Path.IsPathRooted(testDataFile)
                    ? testDataFile
                    : Path.Combine(projectRoot, testDataFile));

                // Directory-based tasks accept a directory (or a direct COCO .json); CSV tasks need a file.
                bool exists = isDirectoryBased
                    ? (Directory.Exists(resolvedTestDataFile) || File.Exists(resolvedTestDataFile))
                    : File.Exists(resolvedTestDataFile);
                if (!exists)
                {
                    ErrorConsole.Error($"Test data not found: {resolvedTestDataFile}", ErrorConsole.PathNotFoundTip(projectRoot));
                    return 1;
                }

                ValueLine.Write("[green]>[/] Using test data: ", Path.GetRelativePath(projectRoot, resolvedTestDataFile));
            }

            AnsiConsole.WriteLine();

            // Validate schema before evaluation (CSV-column validation; not applicable to image directories)
            if (!isDirectoryBased)
            {
                var validator = new SchemaValidator(fileSystem, projectDiscovery);
                var validationResult = await validator.ValidateAsync(resolvedTestDataFile, resolvedModelName, resolvedExperimentId);

                if (!validationResult.IsValid)
                {
                    AnsiConsole.MarkupLine("[red]Schema Validation Failed:[/]");
                    AnsiConsole.WriteLine();

                    if (!string.IsNullOrEmpty(validationResult.ErrorMessage))
                    {
                        AnsiConsole.MarkupLine($"[red]Error: {validationResult.ErrorMessage}[/]");
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

                    // A --json consumer owes a parseable document at every exit, this one
                    // included. The human channel above renders the same finding in parts; the
                    // document carries it as one sentence.
                    if (jsonOutput)
                    {
                        var parts = new List<string>();
                        if (!string.IsNullOrEmpty(validationResult.ErrorMessage))
                            parts.Add(validationResult.ErrorMessage);
                        // Only when the message did not already name them: it usually does, and the
                        // human channel above lists them separately because it has the room to.
                        if (parts.Count == 0 && validationResult.MissingColumns.Any())
                            parts.Add("Missing columns: " + string.Join(", ", validationResult.MissingColumns));
                        parts.AddRange(validationResult.Suggestions);

                        JsonError.Emit("Schema validation failed. " + string.Join(" ", parts));
                    }

                    return 1;
                }

                // Same reason as predict: a column that changed type passes validation, and the
                // measurement it produces is computed on values the loader turned into missing.
                foreach (var warning in validationResult.Warnings)
                    WarningConsole.Warn(Markup.Escape(warning));

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

            var overfitting = DetectOverfitting(experimentData.Task ?? "", trainingMetrics, testMetrics!);

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

            if (jsonOutput)
                EmitJson(resolvedModelName, resolvedExperimentId, resolvedTestDataFile,
                    trainingMetrics, testMetrics, overfitting);

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

    /// <summary>The gap that counts as suspicious on a metric that lives on a fixed 0..1 scale.</summary>
    internal const double OverfittingThreshold = 0.1;

    /// <summary>
    /// The same suspicion for a metric carrying the label's own units: a proportion of the training
    /// score rather than an absolute amount. An rmse of 4.0 against 4.4 is the same story as an rmse
    /// of 0.004 against 0.0044, and only a relative test says so.
    /// </summary>
    internal const double RelativeOverfittingThreshold = 0.10;

    /// <summary>
    /// True when the model scored enough better on its training data than on its test data to be
    /// worth mentioning.
    /// </summary>
    /// <remarks>
    /// <para>Two things used to make this narrower than it reads. It resolved the metric from its
    /// own three-task switch, so for the eight remaining tasks — image and text classification,
    /// ranking, recommendation, clustering, anomaly detection, forecasting, time-series anomaly —
    /// it found no metric and answered "no" every time, which is indistinguishable from a clean
    /// model. And it compared with <c>Math.Abs</c>, so a model that scored *better* on test than on
    /// train was reported as possibly overfitting, which is the one thing that result cannot mean.
    /// </para>
    /// <para>Widening it is not a substitution, because the threshold was only ever valid for the
    /// three tasks it covered: their metrics sit on a fixed scale. A task whose primary metric is
    /// <c>rmse</c>, <c>mae</c> or <c>average_distance</c> carries the label's units, and a fixed
    /// 0.1 there is a statement about the unit, not the model (see <see cref="MetricScale"/>).
    /// Those get a relative test instead.</para>
    /// </remarks>
    internal static bool DetectOverfitting(
        string task,
        Dictionary<string, double> trainingMetrics,
        Dictionary<string, double> testMetrics)
    {
        // "classification" is a legacy alias for binary that predates the CLI-canonical task
        // strings; TaskMetadata does not carry it, and dropping it here would quietly stop
        // checking models configured before the rename.
        var metricKey = task.Trim().Equals("classification", StringComparison.OrdinalIgnoreCase)
            ? TaskMetadata.PrimaryMetric("binary-classification")
            : TaskMetadata.PrimaryMetric(task);

        if (metricKey == null
            || !trainingMetrics.TryGetValue(metricKey, out var trainValue)
            || !testMetrics.TryGetValue(metricKey, out var testValue))
        {
            return false;
        }

        // How much better the training score is. Negative means test did as well or better, which
        // is not overfitting whatever its size.
        var advantage = MetricDirection.IsLowerBetter(metricKey)
            ? testValue - trainValue
            : trainValue - testValue;

        if (advantage <= 0)
            return false;

        if (MetricScale.IsUnitScaled(metricKey))
            return advantage > OverfittingThreshold;

        // Relative to the training score's own magnitude. A training score of zero is a perfect fit
        // on the training data, so any test error at all is the gap this check exists to find.
        var magnitude = Math.Abs(trainValue);
        return magnitude < double.Epsilon
            ? advantage > 0
            : advantage / magnitude > RelativeOverfittingThreshold;
    }
}
