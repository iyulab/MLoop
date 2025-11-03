using System.CommandLine;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
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

        var command = new Command("evaluate", "Evaluate model performance on test data")
        {
            experimentArg,
            testDataArg
        };

        command.SetHandler(ExecuteAsync, experimentArg, testDataArg);

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? experimentId,
        string? testDataFile)
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
                AnsiConsole.MarkupLine("[red]Error:[/] Not inside a MLoop project.");
                AnsiConsole.MarkupLine("Run [blue]mloop init[/] to create a new project.");
                return 1;
            }

            // Resolve experiment and model path
            string resolvedModelPath;
            string resolvedExperimentId;
            ExperimentData? experimentData = null;

            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

            if (string.IsNullOrEmpty(experimentId))
            {
                // Use production model
                var productionModel = await modelRegistry.GetAsync(ModelStage.Production, CancellationToken.None);

                if (productionModel == null)
                {
                    AnsiConsole.MarkupLine("[red]Error:[/] No production model found.");
                    AnsiConsole.MarkupLine("[yellow]Tip:[/] Train a model first: [blue]mloop train[/]");
                    return 1;
                }

                resolvedExperimentId = productionModel.ExperimentId;
                resolvedModelPath = fileSystem.CombinePath(productionModel.ModelPath, "model.zip");
                experimentData = await experimentStore.LoadAsync(resolvedExperimentId, CancellationToken.None);

                AnsiConsole.MarkupLine($"[green]✓[/] Using production model: [cyan]{resolvedExperimentId}[/]");
            }
            else
            {
                // Use specified experiment
                resolvedExperimentId = experimentId;

                if (!experimentStore.ExperimentExists(resolvedExperimentId))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] Experiment not found: {resolvedExperimentId}");
                    return 1;
                }

                experimentData = await experimentStore.LoadAsync(resolvedExperimentId, CancellationToken.None);
                var experimentPath = experimentStore.GetExperimentPath(resolvedExperimentId);
                resolvedModelPath = fileSystem.CombinePath(experimentPath, "model.zip");

                AnsiConsole.MarkupLine($"[green]✓[/] Using experiment: [cyan]{resolvedExperimentId}[/]");
            }

            if (!File.Exists(resolvedModelPath))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Model file not found: {resolvedModelPath}");
                return 1;
            }

            // Resolve test data file path
            string resolvedTestDataFile;

            if (string.IsNullOrEmpty(testDataFile))
            {
                // Auto-discover datasets/test.csv
                var datasetDiscovery = new DatasetDiscovery(fileSystem);
                var datasets = datasetDiscovery.FindDatasets(projectRoot);

                if (datasets?.TestPath == null)
                {
                    AnsiConsole.MarkupLine("[red]Error:[/] No test data specified and datasets/test.csv not found.");
                    AnsiConsole.MarkupLine("[yellow]Tip:[/] Create datasets/test.csv or specify a file: mloop evaluate <experiment-id> <test-file>");
                    return 1;
                }

                resolvedTestDataFile = datasets.TestPath;
                AnsiConsole.MarkupLine($"[green]✓[/] Auto-detected: [cyan]{Path.GetRelativePath(projectRoot, resolvedTestDataFile)}[/]");
            }
            else
            {
                // Validate explicit test data file
                resolvedTestDataFile = Path.IsPathRooted(testDataFile)
                    ? testDataFile
                    : Path.Combine(projectRoot, testDataFile);

                if (!File.Exists(resolvedTestDataFile))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] Test data file not found: {resolvedTestDataFile}");
                    return 1;
                }

                AnsiConsole.MarkupLine($"[green]✓[/] Using test data: [cyan]{Path.GetRelativePath(projectRoot, resolvedTestDataFile)}[/]");
            }

            AnsiConsole.WriteLine();

            // Validate schema before evaluation
            var validator = new SchemaValidator(fileSystem, projectDiscovery);
            var validationResult = await validator.ValidateAsync(resolvedModelPath, resolvedTestDataFile, resolvedExperimentId);

            if (!validationResult.IsValid)
            {
                AnsiConsole.MarkupLine("[red]Schema Validation Failed:[/]");
                AnsiConsole.WriteLine();

                if (!string.IsNullOrEmpty(validationResult.ErrorMessage))
                {
                    AnsiConsole.MarkupLine($"[yellow]⚠ {validationResult.ErrorMessage}[/]");
                    AnsiConsole.WriteLine();
                }

                if (validationResult.MissingColumns.Any())
                {
                    AnsiConsole.MarkupLine("[red]Missing columns in test data:[/]");
                    foreach (var col in validationResult.MissingColumns)
                    {
                        AnsiConsole.MarkupLine($"  [grey]•[/] {col}");
                    }
                    AnsiConsole.WriteLine();
                }

                return 1;
            }

            AnsiConsole.MarkupLine("[green]✓[/] Schema validation passed");
            AnsiConsole.WriteLine();

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
                        CancellationToken.None);

                    ctx.Status("[green]Evaluation complete![/]");
                });

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

                var differenceText = difference >= 0
                    ? $"[green]+{difference:F4}[/]"
                    : $"[red]{difference:F4}[/]";

                // For some metrics, lower is better (RMSE, MAE, MSE), so invert the color
                if (metricName.ToLower().Contains("rmse") ||
                    metricName.ToLower().Contains("mae") ||
                    metricName.ToLower().Contains("mse") ||
                    metricName.ToLower().Contains("loss"))
                {
                    differenceText = difference <= 0
                        ? $"[green]{difference:F4}[/]"
                        : $"[red]+{difference:F4}[/]";
                }

                table.AddRow(
                    metricName,
                    trainingValue.ToString("F4"),
                    testValue.ToString("F4"),
                    differenceText);
            }

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();

            // Overfitting warning
            if (experimentData.Task == "regression")
            {
                if (trainingMetrics.ContainsKey("r_squared") && testMetrics.ContainsKey("r_squared"))
                {
                    var r2Diff = Math.Abs(trainingMetrics["r_squared"] - testMetrics["r_squared"]);
                    if (r2Diff > 0.1)
                    {
                        AnsiConsole.MarkupLine("[yellow]⚠ Warning:[/] Large R² difference detected. Model may be overfitting.");
                        AnsiConsole.WriteLine();
                    }
                }
            }
            else if (experimentData.Task == "classification")
            {
                if (trainingMetrics.ContainsKey("accuracy") && testMetrics.ContainsKey("accuracy"))
                {
                    var accDiff = Math.Abs(trainingMetrics["accuracy"] - testMetrics["accuracy"]);
                    if (accDiff > 0.1)
                    {
                        AnsiConsole.MarkupLine("[yellow]⚠ Warning:[/] Large accuracy difference detected. Model may be overfitting.");
                        AnsiConsole.WriteLine();
                    }
                }
            }

            AnsiConsole.MarkupLine("[green]✓[/] Evaluation complete!");
            AnsiConsole.WriteLine();

            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.Markup("[red]Error:[/] ");
            AnsiConsole.WriteLine(ex.Message);

            if (ex.InnerException != null)
            {
                AnsiConsole.Markup("[grey]Details:[/] ");
                AnsiConsole.WriteLine(ex.InnerException.Message);
            }

            return 1;
        }
    }
}
