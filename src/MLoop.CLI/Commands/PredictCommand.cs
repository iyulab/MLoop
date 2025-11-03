using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using Spectre.Console;
using static MLoop.CLI.Infrastructure.ML.CategoricalMapper;

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

        var unknownStrategyOption = new Option<string>("--unknown-strategy")
        {
            Description = "Strategy for handling unknown categorical values: auto (default), error, use-most-frequent, use-missing"
        };
        unknownStrategyOption.SetDefaultValue("auto");

        var command = new Command("predict", "Make predictions with a trained model")
        {
            modelArg,
            dataFileArg,
            outputOption,
            unknownStrategyOption
        };

        command.SetHandler(ExecuteAsync, modelArg, dataFileArg, outputOption, unknownStrategyOption);

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? modelPath,
        string? dataFile,
        string? output,
        string unknownStrategy)
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

            // Load schema for validation and preprocessing
            string? experimentId = null;
            InputSchemaInfo? trainedSchema = null;

            if (string.IsNullOrEmpty(modelPath))
            {
                // We're using production model, get the experiment ID and schema
                var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
                var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);
                var productionModel = await modelRegistry.GetAsync(ModelStage.Production, CancellationToken.None);
                experimentId = productionModel?.ExperimentId;

                // Load experiment data to get schema
                if (experimentId != null)
                {
                    try
                    {
                        var experimentData = await experimentStore.LoadAsync(experimentId, CancellationToken.None);
                        trainedSchema = experimentData?.Config?.InputSchema;
                    }
                    catch
                    {
                        // Continue without schema - will skip categorical preprocessing
                    }
                }
            }
            else
            {
                // Using explicit model path - try to infer experiment ID from path
                var modelDir = Path.GetDirectoryName(resolvedModelPath);
                if (modelDir != null)
                {
                    var possibleExpId = Path.GetFileName(modelDir);
                    var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);

                    if (experimentStore.ExperimentExists(possibleExpId))
                    {
                        try
                        {
                            var experimentData = await experimentStore.LoadAsync(possibleExpId, CancellationToken.None);
                            trainedSchema = experimentData?.Config?.InputSchema;
                            experimentId = possibleExpId;
                        }
                        catch
                        {
                            // Continue without schema
                        }
                    }
                }
            }

            // Validate schema before prediction
            var validator = new SchemaValidator(fileSystem, projectDiscovery);
            var validationResult = await validator.ValidateAsync(resolvedModelPath, resolvedDataFile, experimentId);

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
                    AnsiConsole.MarkupLine("[red]Missing columns in prediction data:[/]");
                    foreach (var col in validationResult.MissingColumns)
                    {
                        AnsiConsole.MarkupLine($"  [grey]•[/] {col}");
                    }
                    AnsiConsole.WriteLine();
                }

                if (validationResult.TypeMismatchColumns.Any())
                {
                    AnsiConsole.MarkupLine("[red]Column type mismatches:[/]");
                    foreach (var (name, expected, actual) in validationResult.TypeMismatchColumns)
                    {
                        AnsiConsole.MarkupLine($"  [grey]•[/] {name}: expected {expected}, got {actual}");
                    }
                    AnsiConsole.WriteLine();
                }

                if (validationResult.Suggestions.Any())
                {
                    AnsiConsole.MarkupLine("[yellow]Suggestions:[/]");
                    foreach (var suggestion in validationResult.Suggestions)
                    {
                        AnsiConsole.MarkupLine($"  [grey]•[/] {suggestion}");
                    }
                    AnsiConsole.WriteLine();
                }

                return 1;
            }

            AnsiConsole.MarkupLine("[green]✓[/] Schema validation passed");
            AnsiConsole.WriteLine();

            // Parse unknown value strategy
            var strategy = unknownStrategy.ToLowerInvariant() switch
            {
                "auto" => UnknownValueStrategy.Auto,
                "error" => UnknownValueStrategy.Error,
                "use-most-frequent" => UnknownValueStrategy.UseMostFrequent,
                "use-missing" => UnknownValueStrategy.UseMissing,
                _ => UnknownValueStrategy.Auto
            };

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
                        trainedSchema,
                        strategy,
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
