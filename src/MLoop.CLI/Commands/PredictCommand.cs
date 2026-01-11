using System.CommandLine;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
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
        var dataFileArg = new Argument<string?>("data-file")
        {
            Description = "Path to prediction data (defaults to datasets/predict.csv if omitted)",
            Arity = ArgumentArity.ZeroOrOne
        };

        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')"
        };

        var modelPathOption = new Option<string?>("--model", "-m")
        {
            Description = "Path to model file (overrides production model)"
        };

        var outputOption = new Option<string?>("--output", "-o")
        {
            Description = "Output path for predictions (defaults to predictions/predictions.csv)"
        };

        var unknownStrategyOption = new Option<string>("--unknown-strategy")
        {
            Description = "Strategy for handling unknown categorical values: auto (default), error, use-most-frequent, use-missing",
            DefaultValueFactory = _ => "auto"
        };

        var command = new Command("predict", "Make predictions with a trained model");
        command.Arguments.Add(dataFileArg);
        command.Options.Add(nameOption);
        command.Options.Add(modelPathOption);
        command.Options.Add(outputOption);
        command.Options.Add(unknownStrategyOption);

        command.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg);
            var name = parseResult.GetValue(nameOption);
            var modelPath = parseResult.GetValue(modelPathOption);
            var output = parseResult.GetValue(outputOption);
            var unknownStrategy = parseResult.GetValue(unknownStrategyOption)!;
            return ExecuteAsync(dataFile, name, modelPath, output, unknownStrategy);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? dataFile,
        string? modelName,
        string? modelPath,
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

            // Resolve model name (defaults to "default")
            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? ConfigDefaults.DefaultModelName
                : modelName.Trim().ToLowerInvariant();

            // Initialize stores
            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

            // Resolve model path (Convention: use production model if not specified)
            string resolvedModelPath;
            string? experimentId = null;
            InputSchemaInfo? trainedSchema = null;

            if (string.IsNullOrEmpty(modelPath))
            {
                // Auto-load production model for the specified model name
                var productionModel = await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None);

                if (productionModel == null)
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] No production model found for '[cyan]{resolvedModelName}[/]'.");
                    AnsiConsole.MarkupLine($"[yellow]Tip:[/] Train and promote a model first: [blue]mloop train --name {resolvedModelName}[/]");
                    return 1;
                }

                resolvedModelPath = fileSystem.CombinePath(productionModel.ModelPath, "model.zip");
                experimentId = productionModel.ExperimentId;

                AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
                AnsiConsole.MarkupLine($"[green]>[/] Using production model: [cyan]{productionModel.ExperimentId}[/]");

                // Load experiment data to get schema
                try
                {
                    var experimentData = await experimentStore.LoadAsync(resolvedModelName, experimentId, CancellationToken.None);
                    trainedSchema = experimentData?.Config?.InputSchema;
                }
                catch
                {
                    // Continue without schema - will skip categorical preprocessing
                }
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

                AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
                AnsiConsole.MarkupLine($"[green]>[/] Using model file: [cyan]{Path.GetRelativePath(projectRoot, resolvedModelPath)}[/]");

                // Try to infer experiment ID from path
                var modelDir = Path.GetDirectoryName(resolvedModelPath);
                if (modelDir != null)
                {
                    var possibleExpId = Path.GetFileName(modelDir);

                    if (experimentStore.ExperimentExists(resolvedModelName, possibleExpId))
                    {
                        try
                        {
                            var experimentData = await experimentStore.LoadAsync(resolvedModelName, possibleExpId, CancellationToken.None);
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
                    AnsiConsole.MarkupLine("[yellow]Tip:[/] Create datasets/predict.csv or specify a file: mloop predict <data-file>");
                    return 1;
                }

                resolvedDataFile = datasets.PredictPath;
                AnsiConsole.MarkupLine($"[green]>[/] Auto-detected: [cyan]{Path.GetRelativePath(projectRoot, resolvedDataFile)}[/]");
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
                resolvedOutputPath = fileSystem.CombinePath(predictionsDir, $"{resolvedModelName}-predictions-{timestamp}.csv");
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

            // Validate schema before prediction
            var validator = new SchemaValidator(fileSystem, projectDiscovery);
            var validationResult = await validator.ValidateAsync(resolvedModelPath, resolvedDataFile, resolvedModelName, experimentId);

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
                    AnsiConsole.MarkupLine("[red]Missing columns in prediction data:[/]");
                    foreach (var col in validationResult.MissingColumns)
                    {
                        AnsiConsole.MarkupLine($"  [grey]-[/] {col}");
                    }
                    AnsiConsole.WriteLine();
                }

                if (validationResult.TypeMismatchColumns.Any())
                {
                    AnsiConsole.MarkupLine("[red]Column type mismatches:[/]");
                    foreach (var (name, expected, actual) in validationResult.TypeMismatchColumns)
                    {
                        AnsiConsole.MarkupLine($"  [grey]-[/] {name}: expected {expected}, got {actual}");
                    }
                    AnsiConsole.WriteLine();
                }

                if (validationResult.Suggestions.Any())
                {
                    AnsiConsole.MarkupLine("[yellow]Suggestions:[/]");
                    foreach (var suggestion in validationResult.Suggestions)
                    {
                        AnsiConsole.MarkupLine($"  [grey]-[/] {suggestion}");
                    }
                    AnsiConsole.WriteLine();
                }

                return 1;
            }

            AnsiConsole.MarkupLine("[green]>[/] Schema validation passed");
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
            AnsiConsole.Write(new Rule("[green]Predictions Complete![/]").LeftJustified());
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
            AnsiConsole.MarkupLine($"[green]>[/] Predicted: [yellow]{predictedCount}[/] rows");
            AnsiConsole.MarkupLine($"[green]>[/] Output saved to: [cyan]{Path.GetRelativePath(projectRoot, resolvedOutputPath)}[/]");
            AnsiConsole.WriteLine();

            return 0;
        }
        catch (Exception ex)
        {
            // T8.3: Enhanced error messaging with actionable suggestions
            ErrorSuggestions.DisplayError(ex, "prediction");
            return 1;
        }
    }
}
