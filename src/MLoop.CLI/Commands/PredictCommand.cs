using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop predict - Makes predictions using production model or specified model
/// </summary>
public static class PredictCommand
{
    public static Command Create()
    {
        var modelArg = new Argument<string?>("model-path")
        {
            Description = "Path to model file (defaults to production model if omitted)",
            Arity = ArgumentArity.ZeroOrOne
        };

        var dataFileArg = new Argument<string?>("data-file")
        {
            Description = "Path to prediction data (defaults to datasets/predict.csv if omitted)",
            Arity = ArgumentArity.ZeroOrOne
        };

        var outputOption = new Option<string?>("--output", "-o")
        {
            Description = "Output path for predictions (defaults to predictions/predictions.csv)"
        };

        var command = new Command("predict", "Make predictions with a trained model")
        {
            modelArg,
            dataFileArg,
            outputOption
        };

        command.SetHandler(ExecuteAsync, modelArg, dataFileArg, outputOption);

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? modelPath,
        string? dataFile,
        string? output)
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

            // Resolve model path (Convention: use production model if not specified)
            string resolvedModelPath;

            if (string.IsNullOrEmpty(modelPath))
            {
                // Auto-load production model
                var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
                var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

                var productionModel = await modelRegistry.GetAsync(ModelStage.Production, CancellationToken.None);

                if (productionModel == null)
                {
                    AnsiConsole.MarkupLine("[red]Error:[/] No production model found.");
                    AnsiConsole.MarkupLine("[yellow]Tip:[/] Train a model first: [blue]mloop train[/]");
                    return 1;
                }

                resolvedModelPath = fileSystem.CombinePath(productionModel.ModelPath, "model.zip");
                AnsiConsole.MarkupLine($"[green]✓[/] Using production model: [cyan]{productionModel.ExperimentId}[/]");
            }
            else
            {
                // Use specified model path
                resolvedModelPath = Path.IsPathRooted(modelPath)
                    ? modelPath
                    : Path.Combine(projectRoot, modelPath);

                if (!File.Exists(resolvedModelPath))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] Model file not found: {resolvedModelPath}");
                    return 1;
                }

                AnsiConsole.MarkupLine($"[green]✓[/] Using model: [cyan]{Path.GetRelativePath(projectRoot, resolvedModelPath)}[/]");
            }

            // Resolve data file path (Convention: datasets/predict.csv)
            string resolvedDataFile;

            if (string.IsNullOrEmpty(dataFile))
            {
                // Auto-discover datasets/predict.csv
                var datasetDiscovery = new DatasetDiscovery(fileSystem);
                var datasets = datasetDiscovery.FindDatasets(projectRoot);

                if (datasets?.PredictPath == null)
                {
                    AnsiConsole.MarkupLine("[red]Error:[/] No data file specified and datasets/predict.csv not found.");
                    AnsiConsole.MarkupLine("[yellow]Tip:[/] Create datasets/predict.csv or specify a file: mloop predict <model> <data-file>");
                    return 1;
                }

                resolvedDataFile = datasets.PredictPath;
                AnsiConsole.MarkupLine($"[green]✓[/] Auto-detected: [cyan]{Path.GetRelativePath(projectRoot, resolvedDataFile)}[/]");
            }
            else
            {
                // Validate explicit data file
                resolvedDataFile = Path.IsPathRooted(dataFile)
                    ? dataFile
                    : Path.Combine(projectRoot, dataFile);

                if (!File.Exists(resolvedDataFile))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] Data file not found: {resolvedDataFile}");
                    return 1;
                }
            }

            // Resolve output path (Convention: predictions/predictions.csv)
            string resolvedOutputPath;

            if (string.IsNullOrEmpty(output))
            {
                var predictionsDir = fileSystem.CombinePath(projectRoot, "predictions");
                await fileSystem.CreateDirectoryAsync(predictionsDir, CancellationToken.None);

                var timestamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
                resolvedOutputPath = fileSystem.CombinePath(predictionsDir, $"predictions-{timestamp}.csv");
            }
            else
            {
                resolvedOutputPath = Path.IsPathRooted(output)
                    ? output
                    : Path.Combine(projectRoot, output);

                // Create output directory if needed
                var outputDir = Path.GetDirectoryName(resolvedOutputPath);
                if (!string.IsNullOrEmpty(outputDir) && !Directory.Exists(outputDir))
                {
                    await fileSystem.CreateDirectoryAsync(outputDir, CancellationToken.None);
                }
            }

            AnsiConsole.WriteLine();

            // Make predictions with progress
            int predictedCount = 0;

            await AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .StartAsync("[yellow]Making predictions...[/]", async ctx =>
                {
                    var predictionEngine = new PredictionEngine();

                    predictedCount = await predictionEngine.PredictAsync(
                        resolvedModelPath,
                        resolvedDataFile,
                        resolvedOutputPath,
                        CancellationToken.None);

                    ctx.Status("[green]Predictions complete![/]");
                });

            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[green]✓[/] Predictions complete!");
            AnsiConsole.MarkupLine($"[green]✓[/] Predicted: [yellow]{predictedCount}[/] rows");
            AnsiConsole.MarkupLine($"[green]✓[/] Output saved to: [cyan]{Path.GetRelativePath(projectRoot, resolvedOutputPath)}[/]");
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
