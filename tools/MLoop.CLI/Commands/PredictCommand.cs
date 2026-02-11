using System.CommandLine;
using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Preprocessing;
using MLoop.DataStore.Interfaces;
using MLoop.DataStore.Services;
using MLoop.Extensibility.Preprocessing;
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

        var logOption = new Option<bool>("--log", "-l")
        {
            Description = "Log predictions to .mloop/logs/ for monitoring and analysis"
        };

        var includeFeaturesOption = new Option<bool>("--include-features")
        {
            Description = "Include original feature columns in prediction output"
        };

        var command = new Command("predict", "Make predictions with a trained model");
        command.Arguments.Add(dataFileArg);
        command.Options.Add(nameOption);
        command.Options.Add(modelPathOption);
        command.Options.Add(outputOption);
        command.Options.Add(unknownStrategyOption);
        command.Options.Add(logOption);
        command.Options.Add(includeFeaturesOption);

        command.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg);
            var name = parseResult.GetValue(nameOption);
            var modelPath = parseResult.GetValue(modelPathOption);
            var output = parseResult.GetValue(outputOption);
            var unknownStrategy = parseResult.GetValue(unknownStrategyOption)!;
            var logPredictions = parseResult.GetValue(logOption);
            var includeFeatures = parseResult.GetValue(includeFeaturesOption);
            return ExecuteAsync(dataFile, name, modelPath, output, unknownStrategy, logPredictions, includeFeatures);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? dataFile,
        string? modelName,
        string? modelPath,
        string? output,
        string unknownStrategy,
        bool logPredictions,
        bool includeFeatures)
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

            // Apply YAML-defined preprocessing pipeline (same as training)
            try
            {
                var configLoader = new ConfigLoader(fileSystem, projectDiscovery);
                var config = await configLoader.LoadUserConfigAsync();

                if (config.Models.TryGetValue(resolvedModelName, out var modelDef) &&
                    modelDef.Prep is { Count: > 0 })
                {
                    AnsiConsole.MarkupLine($"[green]>[/] Applying {modelDef.Prep.Count} preprocessing step(s)...");

                    var prepLogger = new PredictPrepLogger();
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

            // Merge original features with predictions if requested
            if (includeFeatures)
            {
                MergeInputWithPredictions(resolvedDataFile, resolvedOutputPath, trainedSchema);
                AnsiConsole.MarkupLine("[green]>[/] Original features included in output");
            }

            // Log predictions if requested
            if (logPredictions)
            {
                await LogPredictionsAsync(
                    projectRoot,
                    resolvedModelName,
                    experimentId ?? "unknown",
                    resolvedDataFile,
                    resolvedOutputPath);

                AnsiConsole.MarkupLine($"[green]>[/] Predictions logged to: [cyan].mloop/logs/{resolvedModelName}/[/]");
            }

            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[green]Predictions Complete![/]").LeftJustified());
            AnsiConsole.WriteLine();

            AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{resolvedModelName}[/]");
            AnsiConsole.MarkupLine($"[green]>[/] Predicted: [yellow]{predictedCount}[/] rows");
            AnsiConsole.MarkupLine($"[green]>[/] Output saved to: [cyan]{Path.GetRelativePath(projectRoot, resolvedOutputPath)}[/]");
            if (logPredictions)
            {
                AnsiConsole.MarkupLine($"[green]>[/] Logged to: [cyan].mloop/logs/{resolvedModelName}/[/]");
            }
            AnsiConsole.WriteLine();

            // Show prediction preview and distribution
            DisplayPredictionPreview(resolvedOutputPath);
            DisplayPredictionDistribution(resolvedOutputPath);

            return 0;
        }
        catch (Exception ex)
        {
            // T8.3: Enhanced error messaging with actionable suggestions
            ErrorSuggestions.DisplayError(ex, "prediction");
            return 1;
        }
    }

    private static void MergeInputWithPredictions(string inputPath, string outputPath, InputSchemaInfo? schema)
    {
        string? labelColumn = null;
        if (schema != null)
        {
            labelColumn = schema.Columns.FirstOrDefault(c => c.Purpose == "Label")?.Name;
        }

        var inputLines = File.ReadAllLines(inputPath, System.Text.Encoding.UTF8);
        var predLines = File.ReadAllLines(outputPath, System.Text.Encoding.UTF8);

        if (inputLines.Length < 2 || predLines.Length < 2) return;

        var inputHeaders = CsvFieldParser.ParseFields(inputLines[0]);
        int labelIdx = labelColumn != null ? Array.IndexOf(inputHeaders, labelColumn) : -1;

        // Column indices to keep (exclude label)
        var keepIndices = Enumerable.Range(0, inputHeaders.Length)
            .Where(i => i != labelIdx)
            .ToList();

        using var writer = new StreamWriter(outputPath, false, new System.Text.UTF8Encoding(true));

        // Header: features + predictions
        var featureHeaders = keepIndices.Select(i => QuoteCsvField(inputHeaders[i]));
        writer.WriteLine(string.Join(",", featureHeaders) + "," + predLines[0]);

        // Data rows
        for (int r = 1; r < Math.Min(inputLines.Length, predLines.Length); r++)
        {
            var fields = CsvFieldParser.ParseFields(inputLines[r]);
            var featureValues = keepIndices.Select(i => i < fields.Length ? QuoteCsvField(fields[i]) : "");
            writer.WriteLine(string.Join(",", featureValues) + "," + predLines[r]);
        }
    }

    private static string QuoteCsvField(string value)
    {
        if (value.Contains(',') || value.Contains('"') || value.Contains('\n'))
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        return value;
    }

    private static async Task LogPredictionsAsync(
        string projectRoot,
        string modelName,
        string experimentId,
        string inputFile,
        string outputFile)
    {
        var logger = new FilePredictionLogger(projectRoot);

        // Read input data
        var inputRecords = new List<Dictionary<string, object>>();
        using (var inputReader = new StreamReader(inputFile))
        using (var inputCsv = new CsvReader(inputReader, new CsvConfiguration(CultureInfo.InvariantCulture)))
        {
            await foreach (var record in inputCsv.GetRecordsAsync<dynamic>())
            {
                var dict = new Dictionary<string, object>();
                foreach (var property in (IDictionary<string, object>)record)
                {
                    dict[property.Key] = property.Value;
                }
                inputRecords.Add(dict);
            }
        }

        // Read output data (predictions)
        var predictions = new List<object>();
        using (var outputReader = new StreamReader(outputFile))
        using (var outputCsv = new CsvReader(outputReader, new CsvConfiguration(CultureInfo.InvariantCulture)))
        {
            await foreach (var record in outputCsv.GetRecordsAsync<dynamic>())
            {
                var dict = (IDictionary<string, object>)record;
                // Use the last column as prediction (typically "PredictedLabel" or "Score")
                var lastValue = dict.Values.LastOrDefault() ?? string.Empty;
                predictions.Add(lastValue);
            }
        }

        // Log each prediction
        var entries = new List<PredictionLogEntry>();
        var timestamp = DateTimeOffset.UtcNow;

        for (int i = 0; i < Math.Min(inputRecords.Count, predictions.Count); i++)
        {
            entries.Add(new PredictionLogEntry(
                modelName,
                experimentId,
                inputRecords[i],
                predictions[i],
                null,
                timestamp));
        }

        if (entries.Count > 0)
        {
            await logger.LogBatchAsync(modelName, experimentId, entries);
        }
    }

    private static void DisplayPredictionPreview(string outputPath)
    {
        try
        {
            var lines = File.ReadLines(outputPath).Take(6).ToList(); // header + 5 rows
            if (lines.Count < 2) return;

            var headers = CsvFieldParser.ParseFields(lines[0]);

            var table = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey)
                .Title("[bold]Prediction Preview (first 5 rows)[/]");

            foreach (var header in headers)
            {
                table.AddColumn(new TableColumn($"[bold]{Markup.Escape(header)}[/]"));
            }

            for (int i = 1; i < lines.Count; i++)
            {
                var fields = CsvFieldParser.ParseFields(lines[i]);
                var cells = headers.Select((_, idx) =>
                    idx < fields.Length ? Markup.Escape(fields[idx]) : "[grey]-[/]").ToArray();
                table.AddRow(cells);
            }

            AnsiConsole.Write(table);
            AnsiConsole.WriteLine();
        }
        catch
        {
            // Non-critical — skip preview on error
        }
    }

    private static void DisplayPredictionDistribution(string outputPath)
    {
        try
        {
            var lines = File.ReadLines(outputPath).ToList();
            if (lines.Count < 2) return;

            var headers = CsvFieldParser.ParseFields(lines[0]);

            // Find PredictedLabel column, or use first column
            var predIdx = Array.FindIndex(headers, h =>
                h.Equals("PredictedLabel", StringComparison.OrdinalIgnoreCase));
            if (predIdx < 0) predIdx = 0;

            // Collect values
            var values = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 1; i < lines.Count; i++)
            {
                var fields = CsvFieldParser.ParseFields(lines[i]);
                if (predIdx < fields.Length)
                {
                    var val = fields[predIdx].Trim();
                    values[val] = values.GetValueOrDefault(val) + 1;
                }
            }

            // Only show distribution for categorical predictions (reasonable number of unique values)
            if (values.Count < 2 || values.Count > 50) return;

            AnsiConsole.Write(new Rule($"[blue]Prediction Distribution ({headers[predIdx]})[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var total = values.Values.Sum();
            foreach (var (label, count) in values.OrderByDescending(kv => kv.Value))
            {
                var pct = (double)count / total * 100;
                var barWidth = (int)(pct / 100 * 30);
                var bar = new string('#', Math.Max(barWidth, 1));
                AnsiConsole.MarkupLine($"  [cyan]{Markup.Escape(label),-20}[/] [yellow]{bar}[/] {count} ({pct:F1}%)");
            }

            AnsiConsole.WriteLine();
        }
        catch
        {
            // Non-critical — skip distribution on error
        }
    }

    private class PredictPrepLogger : ILogger
    {
        public void Debug(string message) { }
        public void Info(string message) => AnsiConsole.MarkupLine($"[grey]  {Markup.Escape(message)}[/]");
        public void Warning(string message) => AnsiConsole.MarkupLine($"[yellow]  {Markup.Escape(message)}[/]");
        public void Error(string message) => AnsiConsole.MarkupLine($"[red]  {Markup.Escape(message)}[/]");
        public void Error(string message, Exception ex) => AnsiConsole.MarkupLine($"[red]  {Markup.Escape(message)}: {Markup.Escape(ex.Message)}[/]");
    }
}
