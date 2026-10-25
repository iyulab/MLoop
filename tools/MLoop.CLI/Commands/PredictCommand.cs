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

/// <summary>
/// mloop predict - Makes predictions using production model or specified model
/// </summary>
public static partial class PredictCommand
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

        var experimentOption = new Option<string?>("--experiment", "-e")
        {
            Description = "Predict with this experiment (e.g. a staging candidate) instead of production — a shadow prediction; production is unchanged"
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

        var logOption = new Option<bool>("--log")
        {
            Description = "Log predictions to .mloop/logs/ for monitoring and analysis"
        };

        var includeFeaturesOption = new Option<bool>("--include-features")
        {
            Description = "Include original feature columns in prediction output (default: true)",
            DefaultValueFactory = _ => true
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Emit structured predictions as JSON to stdout (same schema as the /predict API, incl. normalized confidence) instead of writing a CSV. Progress goes to stderr."
        };

        var command = new Command("predict",
            "Make predictions with a trained model. The default output mixes a human-readable " +
            "summary with the results — for machine/automation consumption use --json (pure JSON " +
            "to stdout, progress on stderr) or --output <file> (CSV).");
        command.Arguments.Add(dataFileArg);
        command.Options.Add(nameOption);
        command.Options.Add(modelPathOption);
        command.Options.Add(experimentOption);
        command.Options.Add(outputOption);
        command.Options.Add(unknownStrategyOption);
        command.Options.Add(logOption);
        command.Options.Add(includeFeaturesOption);
        command.Options.Add(jsonOption);
        var recordsOption = DataFileOptions.Records();
        command.Options.Add(recordsOption);

        command.SetAction((parseResult) =>
        {
            var dataFile = parseResult.GetValue(dataFileArg);
            var name = parseResult.GetValue(nameOption);
            var modelPath = parseResult.GetValue(modelPathOption);
            var output = parseResult.GetValue(outputOption);
            var unknownStrategy = parseResult.GetValue(unknownStrategyOption)!;
            var logPredictions = parseResult.GetValue(logOption);
            var includeFeatures = parseResult.GetValue(includeFeaturesOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteAsync(dataFile, name, modelPath, output, unknownStrategy, logPredictions, includeFeatures, json, parseResult.GetValue(recordsOption), parseResult.GetValue(experimentOption));
        });

        return command;
    }

    // camelCase, null fields omitted — matches the /predict API's field names so the two structured
    // outputs share one schema (the single-source goal); consumers key by name and tolerate absent fields.
    private static readonly JsonSerializerOptions PredictJsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };

    private static async Task<int> ExecuteAsync(
        string? dataFile,
        string? modelName,
        string? modelPath,
        string? output,
        string unknownStrategy,
        bool logPredictions,
        bool includeFeatures,
        bool json = false,
        string? records = null,
        string? experiment = null)
    {
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = json ? new JsonOutputScope() : null;

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

            // The model to predict with, and what its experiment record says it was trained on.
            var model = await ResolveModelAsync(modelPath, experiment, resolvedModelName, projectRoot, fileSystem, projectDiscovery);
            if (model is null)
                return 1;
            var (resolvedModelPath, experimentId, trainedSchema, configLabelColumn, taskType, preserveColumns, interval) = model;

            // Forecasting: generate future predictions without input data. The forecast is horizon-based
            // (stateful SSA replaying the training series), so it exits here before the row-oriented
            // paths — including the --json PredictionService route, which would silently no-op.
            if (taskType != null && taskType.Equals("forecasting", StringComparison.OrdinalIgnoreCase))
            {
                if (json)
                    return await PredictForecastingJsonAsync(resolvedModelPath, resolvedModelName, experimentId);

                var forecastOutputPath = fileSystem.CombinePath(
                    projectRoot, ProjectLayout.PredictionsDirectory,
                    ProjectLayout.PredictionFileName(
                        resolvedModelName, ProjectLayout.PredictionKind.Forecast, DateTimeOffset.Now));
                return await PredictForecastingAsync(resolvedModelPath, forecastOutputPath, experimentId);
            }

            // DL tasks (image-classification, object-detection, NLP) need their native runtime
            // loaded before any Model.Load deserializes native parameters. Guard at the CLI boundary
            // so every downstream load (schema validation, PredictionEngine, PredictionService) is
            // covered with one call where the task type is known. No-op for tabular tasks.
            if (taskType != null)
            {
                try
                {
                    MLoop.Core.Runtime.RuntimeManager.EnsureRuntimeForTask(taskType);
                }
                catch (InvalidOperationException ex)
                {
                    ErrorConsole.Error($"{ex.Message}");
                    return 1;
                }
            }

            // Object detection predicts over images (a folder of them, one image, or a COCO/YOLO dataset's
            // pictures) and emits detections (boxes + labels + scores) as JSON, so it bypasses the
            // CSV-oriented resolution, schema validation, preprocessing, and categorical mapping below.
            if (taskType != null && taskType.Equals("object-detection", StringComparison.OrdinalIgnoreCase))
            {
                return await PredictObjectDetectionAsync(
                    projectRoot, resolvedModelPath, dataFile, output, resolvedModelName, configLabelColumn, fileSystem);
            }

            // Resolve data file path (Convention: datasets/predict.csv)
            var resolvedDataFile = ResolveDataFile(dataFile, projectRoot, taskType, fileSystem);
            if (resolvedDataFile is null)
                return 1;

            // Resolve output path (Convention: predictions/predictions.csv)
            var resolvedOutputPath = await ResolveOutputPathAsync(output, projectRoot, resolvedModelName, fileSystem);

            // Any format as a table, UTF-8, flattened, validated against the training schema, preprocessed.
            var preparedRows = await PrepareRowsAsync(
                resolvedDataFile, records, resolvedModelName, experimentId, fileSystem, projectDiscovery);
            if (preparedRows is null)
                return 1;
            resolvedDataFile = preparedRows;

            // Parse unknown value strategy
            var strategy = unknownStrategy.ToLowerInvariant() switch
            {
                "auto" => UnknownValueStrategy.Auto,
                "error" => UnknownValueStrategy.Error,
                "use-most-frequent" => UnknownValueStrategy.UseMostFrequent,
                "use-missing" => UnknownValueStrategy.UseMissing,
                _ => UnknownValueStrategy.Auto
            };

            // Warn for specialized prediction formats
            if (taskType != null && taskType.Equals("time-series-anomaly", StringComparison.OrdinalIgnoreCase))
            {
                AnsiConsole.MarkupLine("[yellow]Note:[/] Time-series anomaly predictions output specialized format.");
                AnsiConsole.WriteLine();
            }

            // --json: route structured predictions through the shared PredictionService (the same path as
            // the /predict API) and emit them to stdout, so mloop-mcp and other consumers get the conformal
            // band and normalized confidence instead of parsing CSV/Spectre text. Bypasses the CSV output.
            if (json)
                return await PredictAsJsonAsync(resolvedDataFile, model, logPredictions, projectRoot, resolvedModelName);

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
                        CancellationToken.None,
                        configLabelColumn,
                        preserveColumns,
                        interval,
                        taskType);

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
            ValueLine.Write("[green]>[/] Output saved to: ", Path.GetRelativePath(projectRoot, resolvedOutputPath));
            if (logPredictions)
            {
                AnsiConsole.MarkupLine($"[green]>[/] Logged to: [cyan].mloop/logs/{resolvedModelName}/[/]");
            }
            AnsiConsole.WriteLine();

            // Show prediction preview and distribution
            DisplayPredictionPreview(resolvedOutputPath, resolvedDataFile);
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

    /// <summary>
    /// Columns the predict-time loader must keep individually addressable (not merged into the
    /// Features vector) because the trained model's per-column transforms reference them by name —
    /// the ranking group column and the recommendation user/item columns. Returns null when none
    /// apply.
    /// </summary>
    private static List<string>? BuildPreserveColumns(ExperimentConfig? config)
    {
        if (config == null) return null;
        var cols = new List<string>();
        if (!string.IsNullOrWhiteSpace(config.GroupColumn)) cols.Add(config.GroupColumn);
        if (!string.IsNullOrWhiteSpace(config.UserColumn)) cols.Add(config.UserColumn);
        if (!string.IsNullOrWhiteSpace(config.ItemColumn)) cols.Add(config.ItemColumn);
        return cols.Count > 0 ? cols : null;
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
        var featureHeaders = keepIndices.Select(i => CsvFieldParser.Quote(inputHeaders[i]));
        writer.WriteLine(string.Join(",", featureHeaders) + "," + predLines[0]);

        // Data rows
        for (int r = 1; r < Math.Min(inputLines.Length, predLines.Length); r++)
        {
            var fields = CsvFieldParser.ParseFields(inputLines[r]);
            var featureValues = keepIndices.Select(i => i < fields.Length ? CsvFieldParser.Quote(fields[i]) : "");
            writer.WriteLine(string.Join(",", featureValues) + "," + predLines[r]);
        }
    }


    /// <summary>
    /// Builds an honest diagnosis when the resolved predict data path is not a readable data file.
    /// A bare "Data file not found" is misleading when the path actually exists as a directory.
    /// (An image-classification model predicts over a folder of images and never reaches here;
    /// object detection has its own directory path.)
    /// </summary>
    internal static string BuildMissingDataFileMessage(string resolvedDataFile)
    {
        if (Directory.Exists(resolvedDataFile))
            return $"Expected a data file but got a directory: {resolvedDataFile}";

        return $"Data file not found: {resolvedDataFile}";
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
                predictions.Add(SelectPredictionValue(dict));
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

    /// <summary>Columns appended by the CSV predict path that enrich, not replace, the prediction — the
    /// conformal band and the additive <c>Confidence</c> column. None of these may ever be logged as the
    /// prediction.</summary>
    private static readonly HashSet<string> EnrichmentColumns =
        new(StringComparer.OrdinalIgnoreCase) { "Confidence", "ScoreLowerBound", "ScoreUpperBound" };

    /// <summary>
    /// Selects the canonical point prediction from a prediction CSV row by column NAME, not position.
    /// ML.NET emits <c>PredictedLabel</c> (classification/clustering/anomaly-detection) or a scalar
    /// <c>Score</c> (regression/ranking) as the actual prediction. A positional "last column" heuristic
    /// silently logs whichever column happens to trail — the appended <c>Confidence</c> (or, pre-change,
    /// <c>ScoreUpperBound</c> for banded regression) — corrupting the prediction logs that feed
    /// feedback/trigger/drift. When no named prediction column exists (time-series-anomaly emits a
    /// <c>Prediction</c> vector split into <c>Prediction.0/1/2</c>), fall back to the last column but skip
    /// <see cref="EnrichmentColumns"/> so an appended enrichment is never mistaken for the prediction.
    /// </summary>
    internal static object SelectPredictionValue(IDictionary<string, object> row)
    {
        foreach (var name in new[] { "PredictedLabel", "Score" })
        {
            if (row.TryGetValue(name, out var value) && value != null)
            {
                return value;
            }
        }
        return row.Where(kv => !EnrichmentColumns.Contains(kv.Key))
                  .Select(kv => kv.Value)
                  .LastOrDefault() ?? string.Empty;
    }

    /// <summary>
    /// The narrowest column this preview is willing to render, borders and padding included. Below
    /// it a cell stops being a value and becomes one character per line.
    /// </summary>
    private const int MinimumReadableColumnWidth = 14;

    /// <summary>
    /// Which of an output file's columns the preview shows, in file order, and how many it leaves out.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A prediction file carries the answer in its <i>trailing</i> columns, because the run appends
    /// them to the row it was given. A table that renders every column therefore spends the width on
    /// the user's own input and squeezes the answer to nothing: measured on this repository's churn
    /// example at 80 columns, 24 columns each got one character, every header was elided to an
    /// ellipsis, and the values ran vertically one letter per line. The output was the input, plus
    /// nothing readable.
    /// </para>
    /// <para>
    /// So the answer columns are kept unconditionally and the input columns fill whatever width is
    /// left, from the front — enough to recognize which row is which. What is dropped is what the
    /// user already has in their own file, and the count is stated rather than silently swallowed.
    /// A column is an "answer" when the input did not have it, which is a fact about these two
    /// files rather than a vocabulary this has to keep in step with the trainers.
    /// </para>
    /// </remarks>
    internal static (string[] Shown, int Omitted) SelectPreviewColumns(
        IReadOnlyList<string> outputHeaders,
        IReadOnlyCollection<string> inputHeaders,
        int consoleWidth)
    {
        var fromInput = new HashSet<string>(inputHeaders.Select(NormalizeHeader), StringComparer.OrdinalIgnoreCase);
        var answers = outputHeaders.Where(h => !fromInput.Contains(NormalizeHeader(h))).ToList();

        // Never fewer than the answers themselves: a preview that hides what was predicted has no
        // reason to exist. Beyond them, as many leading input columns as the width can carry.
        var capacity = Math.Max(consoleWidth / MinimumReadableColumnWidth, answers.Count + 1);
        var context = outputHeaders.Where(h => fromInput.Contains(NormalizeHeader(h)))
                                   .Take(Math.Max(0, capacity - answers.Count))
                                   .ToHashSet(StringComparer.Ordinal);

        var shown = outputHeaders.Where(h => context.Contains(h) || !fromInput.Contains(NormalizeHeader(h)))
                                 .ToArray();

        return (shown, outputHeaders.Count - shown.Length);
    }

    /// <summary>
    /// A header as it compares across two files — a byte-order mark written into one of them is not
    /// part of the column's name, and left in place it would make the first input column read as
    /// something the model predicted.
    /// </summary>
    private static string NormalizeHeader(string header) => header.Trim().TrimStart('﻿');

    private static void DisplayPredictionPreview(string outputPath, string inputPath)
    {
        try
        {
            var lines = File.ReadLines(outputPath).Take(6).ToList(); // header + 5 rows
            if (lines.Count < 2) return;

            var headers = CsvFieldParser.ParseFields(lines[0]);
            var inputHeaders = ReadHeaderOf(inputPath);

            var (shown, omitted) = SelectPreviewColumns(headers, inputHeaders, AnsiConsole.Profile.Width);
            var shownIndexes = shown.Select(h => Array.IndexOf(headers, h)).Where(i => i >= 0).ToArray();

            var table = new Table()
                .Border(TableBorder.Rounded)
                .BorderColor(Color.Grey)
                .Title("[bold]Prediction Preview (first 5 rows)[/]");

            foreach (var index in shownIndexes)
            {
                table.AddColumn(new TableColumn($"[bold]{Markup.Escape(NormalizeHeader(headers[index]))}[/]"));
            }

            for (int i = 1; i < lines.Count; i++)
            {
                var fields = CsvFieldParser.ParseFields(lines[i]);
                var cells = shownIndexes.Select(idx =>
                    idx < fields.Length ? Markup.Escape(fields[idx]) : "[grey]-[/]").ToArray();
                table.AddRow(cells);
            }

            AnsiConsole.Write(table);
            if (omitted > 0)
            {
                AnsiConsole.MarkupLine(
                    $"[grey]{omitted} input column(s) not shown — every column is in the saved file.[/]");
            }
            AnsiConsole.WriteLine();
        }
        catch
        {
            // Non-critical — skip preview on error
        }
    }

    /// <summary>
    /// The column names of a CSV, or none when the file cannot be read — in which case every output
    /// column reads as an answer and the preview shows them all, as it did before.
    /// </summary>
    private static IReadOnlyCollection<string> ReadHeaderOf(string path)
    {
        try
        {
            var header = File.ReadLines(path).FirstOrDefault();
            return header is null ? [] : CsvFieldParser.ParseFields(header);
        }
        catch
        {
            return [];
        }
    }

    /// <summary>
    /// The column whose values the distribution counts: the predicted class, or <c>-1</c> when the
    /// answer is not a class. A score (regression, ranking) has no categories to count — and falling
    /// back to the first column, as this did, counted an input column (a ranking file's query ids)
    /// under the heading "Prediction Distribution".
    /// </summary>
    internal static int DistributionColumn(string[] headers) =>
        Array.FindIndex(headers, h => h.Equals("PredictedLabel", StringComparison.OrdinalIgnoreCase));

    private static void DisplayPredictionDistribution(string outputPath)
    {
        try
        {
            // Stream file instead of loading entirely into memory
            var values = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            string[]? headers = null;
            int predIdx = 0;

            foreach (var line in File.ReadLines(outputPath))
            {
                if (headers == null)
                {
                    headers = CsvFieldParser.ParseFields(line);
                    predIdx = DistributionColumn(headers);
                    if (predIdx < 0) return;
                    continue;
                }

                var fields = CsvFieldParser.ParseFields(line);
                if (predIdx < fields.Length)
                {
                    var val = fields[predIdx].Trim();
                    values[val] = values.GetValueOrDefault(val) + 1;
                }

                // Early exit if too many unique values (regression output)
                if (values.Count > 50) return;
            }

            if (headers == null || values.Count < 2) return;

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

    /// <summary>
    /// Which column to strip from the structured-predict input rows. For tabular tasks the label is
    /// the unknown-to-predict and any stale values in the input CSV must not leak into the features.
    /// For time-series tasks the "label" IS the monitored series value column — stripping it defaulted
    /// the whole series to zero and the detector silently returned an all-normal fabrication
    /// (isAnomaly=false, rawScore=0 on every row) with exit 0 (the silent-GIGO family).
    /// </summary>
    internal static string? LabelColumnToExcludeFromRows(string? taskType, string? configLabelColumn)
        => TrainingRunner.IsTimeSeriesTask(taskType) ? null : configLabelColumn;

    /// <summary>
    /// --json presenter for forecasting: emits the horizon forecast to stdout in the same payload
    /// shape as the tabular --json path, so structured consumers (mloop-mcp) get the forecast and
    /// its bounds instead of a silently ignored flag — the CLI-side half of that gap.
    /// </summary>
    private static async Task<int> PredictForecastingJsonAsync(
        string modelPath, string modelName, string? experimentId)
    {
        var (forecast, error) = await ForecastReplayService.ComputeForecastAsync(new MLContext(), modelPath, experimentId);
        if (forecast is null)
        {
            // AnsiConsole is rerouted to stderr in --json mode, keeping stdout pure.
            ErrorConsole.Error($"{error ?? "Forecast failed."}");
            return 1;
        }

        var payload = new
        {
            model = modelName,
            experimentId,
            modelSha256 = ModelFingerprint.Sha256(modelPath),
            task = "forecasting",
            count = forecast.ForecastedValues.Length,
            predictions = ForecastReplayService.BuildForecastRows(forecast),
            warnings = JsonOutputScope.Warnings ?? [],
        };
        Console.Out.WriteLine(JsonSerializer.Serialize(payload, PredictJsonOptions));
        return 0;
    }

    private static async Task<int> PredictForecastingAsync(string modelPath, string outputPath, string? experimentId)
    {
        var exitCode = 1;
        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .StartAsync("[yellow]Generating forecast...[/]", async ctx =>
            {
                var (forecast, error) = await ForecastReplayService.ComputeForecastAsync(new MLContext(), modelPath, experimentId);
                if (forecast is null)
                {
                    var cause = Markup.Escape(error ?? "Forecast failed.");
                    if (error?.Contains("training data", StringComparison.OrdinalIgnoreCase) == true)
                        ErrorConsole.Error(cause, "Keep the original training data file in place for forecasting predict.");
                    else
                        ErrorConsole.Error(cause);
                    return;
                }

                var forecastValues = forecast.ForecastedValues;
                var lowerValues = forecast.LowerBound;
                var upperValues = forecast.UpperBound;

                // Write forecast CSV
                var dir = Path.GetDirectoryName(outputPath);
                if (dir != null) Directory.CreateDirectory(dir);

                await using var writer = new StreamWriter(outputPath, false, new System.Text.UTF8Encoding(true));
                await writer.WriteLineAsync("Step,ForecastedValue,LowerBound,UpperBound").ConfigureAwait(false);
                for (int i = 0; i < forecastValues.Length; i++)
                {
                    var lower = i < lowerValues.Length ? lowerValues[i] : float.NaN;
                    var upper = i < upperValues.Length ? upperValues[i] : float.NaN;
                    await writer.WriteLineAsync($"{i + 1},{forecastValues[i]},{lower},{upper}").ConfigureAwait(false);
                }

                ctx.Status("[green]Forecast complete![/]");

                AnsiConsole.MarkupLine($"[green]>[/] Forecasted: [yellow]{forecastValues.Length}[/] steps ahead");
                ValueLine.Write("[green]>[/] Output saved to: ", outputPath);
                AnsiConsole.WriteLine();

                // Show summary table
                var table = new Table().Border(TableBorder.Rounded).BorderColor(Color.Grey);
                var confidencePct = (int)Math.Round(ForecastOutput.ConfidenceLevel * 100);
                table.AddColumn("Step");
                table.AddColumn("Forecast");
                table.AddColumn($"Lower {confidencePct}%");
                table.AddColumn($"Upper {confidencePct}%");

                var showCount = Math.Min(forecastValues.Length, 10);
                for (int i = 0; i < showCount; i++)
                {
                    var lo = i < lowerValues.Length ? lowerValues[i].ToString("F4") : "-";
                    var up = i < upperValues.Length ? upperValues[i].ToString("F4") : "-";
                    table.AddRow((i + 1).ToString(), forecastValues[i].ToString("F4"), lo, up);
                }
                if (forecastValues.Length > 10)
                    table.AddRow("...", "...", "...", "...");

                AnsiConsole.Write(table);
                exitCode = 0;
            });
        return exitCode;
    }

    /// <summary>
    /// Predicts object detections over images and writes them as JSON — labels, scores, and boxes.
    /// The input is pictures alone: a folder of them (any depth — a COCO or YOLO dataset's images
    /// included, its annotations unread), one image, or a COCO annotations file's folder. The native
    /// runtime is already loaded by the EnsureRuntimeForTask guard before this method is reached.
    /// </summary>
    private static async Task<int> PredictObjectDetectionAsync(
        string projectRoot,
        string modelPath,
        string? dataFile,
        string? output,
        string modelName,
        string? labelColumn,
        FileSystemManager fileSystem)
    {
        // Resolve the input dataset directory (or a direct COCO .json).
        string resolvedDataDir;
        if (string.IsNullOrEmpty(dataFile))
        {
            var odDir = DatasetDiscovery.FindDirectoryDataset(projectRoot, "object-detection");
            if (odDir == null)
            {
                ErrorConsole.Error(
                    "No data specified and no object-detection dataset found (datasets/coco, datasets/yolo, or datasets/).",
                    "Pass a directory: mloop predict <dir>");
                return 1;
            }
            resolvedDataDir = odDir;
            ValueLine.Write("[green]>[/] Auto-detected: ", Path.GetRelativePath(projectRoot, resolvedDataDir));
        }
        else
        {
            resolvedDataDir = ProjectPaths.Resolve(dataFile, projectRoot);
            if (!Directory.Exists(resolvedDataDir) && !File.Exists(resolvedDataDir))
            {
                ErrorConsole.Error($"Data not found: {resolvedDataDir}", ErrorConsole.PathNotFoundTip(projectRoot));
                return 1;
            }
        }

        // The pictures: an image file is itself, a COCO annotations file stands for its folder.
        IReadOnlyList<string> images = File.Exists(resolvedDataDir)
            ? (ImageDirectoryLoader.IsSupportedImage(resolvedDataDir)
                ? [Path.GetFullPath(resolvedDataDir)]
                : ImageDirectoryLoader.ListImages(Path.GetDirectoryName(Path.GetFullPath(resolvedDataDir))!))
            : ImageDirectoryLoader.ListImages(resolvedDataDir);
        if (images.Count == 0)
        {
            ErrorConsole.Error(
                $"No images found in {Path.GetRelativePath(projectRoot, resolvedDataDir)}.",
                $"Object detection predicts over image files ({ImageDirectoryLoader.SupportedExtensionList}).");
            return 1;
        }

        // Resolve output path — the convention itself lives in ProjectLayout, because `status`
        // reads these names back and used to know only the tabular one.
        string resolvedOutputPath;
        if (string.IsNullOrEmpty(output))
        {
            var predictionsDir = fileSystem.CombinePath(projectRoot, ProjectLayout.PredictionsDirectory);
            await fileSystem.CreateDirectoryAsync(predictionsDir, CancellationToken.None);
            resolvedOutputPath = fileSystem.CombinePath(
                predictionsDir,
                ProjectLayout.PredictionFileName(
                    modelName, ProjectLayout.PredictionKind.Detections, DateTimeOffset.Now));
        }
        else
        {
            resolvedOutputPath = ProjectPaths.Resolve(output, projectRoot);
            var outputDir = Path.GetDirectoryName(resolvedOutputPath);
            if (!string.IsNullOrEmpty(outputDir) && !Directory.Exists(outputDir))
                await fileSystem.CreateDirectoryAsync(outputDir, CancellationToken.None);
        }

        AnsiConsole.WriteLine();

        IReadOnlyList<ImageDetectionResult> detections = Array.Empty<ImageDetectionResult>();

        await AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .StartAsync("[yellow]Detecting objects...[/]", async ctx =>
            {
                await Task.Run(() =>
                {
                    var mlContext = new MLContext(seed: 42);
                    var model = ScoringModel.Load(mlContext, modelPath, "object-detection");

                    detections = ObjectDetectionPredictor.Detect(mlContext, model, images, labelColumn);
                });

                ctx.Status("[green]Detection complete![/]");
            });

        var json = System.Text.Json.JsonSerializer.Serialize(detections, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });
        await File.WriteAllTextAsync(resolvedOutputPath, json, new System.Text.UTF8Encoding(false));

        var totalDetections = detections.Sum(d => d.Detections.Count);

        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[green]Detections Complete![/]").LeftJustified());
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{modelName}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Images: [yellow]{detections.Count}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Detections: [yellow]{totalDetections}[/]");
        ValueLine.Write("[green]>[/] Output saved to: ", Path.GetRelativePath(projectRoot, resolvedOutputPath));
        AnsiConsole.WriteLine();

        DisplayDetectionSummary(detections);

        return 0;
    }

    /// <summary>Shows a per-class detection count and a small per-image preview.</summary>
    private static void DisplayDetectionSummary(IReadOnlyList<ImageDetectionResult> detections)
    {
        if (detections.Count == 0) return;

        var byClass = detections
            .SelectMany(d => d.Detections)
            .GroupBy(b => b.Label)
            .OrderByDescending(g => g.Count())
            .ToList();

        if (byClass.Count > 0)
        {
            AnsiConsole.Write(new Rule("[blue]Detections by class[/]").LeftJustified());
            AnsiConsole.WriteLine();
            var total = byClass.Sum(g => g.Count());
            foreach (var g in byClass)
            {
                var pct = total > 0 ? (double)g.Count() / total * 100 : 0;
                var barWidth = (int)(pct / 100 * 30);
                var bar = new string('#', Math.Max(barWidth, 1));
                AnsiConsole.MarkupLine($"  [cyan]{Markup.Escape(g.Key),-20}[/] [yellow]{bar}[/] {g.Count()} ({pct:F1}%)");
            }
            AnsiConsole.WriteLine();
        }

        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey)
            .Title("[bold]Detection Preview (first 5 images)[/]");
        table.AddColumn("[bold]Image[/]");
        table.AddColumn("[bold]Detections[/]");
        table.AddColumn("[bold]Top label (score)[/]");

        foreach (var image in detections.Take(5))
        {
            var top = image.Detections.OrderByDescending(b => b.Score).FirstOrDefault();
            var topText = top != null ? $"{Markup.Escape(top.Label)} ({top.Score:F2})" : "[grey]none[/]";
            table.AddRow(
                Markup.Escape(Path.GetFileName(image.ImagePath)),
                image.Detections.Count.ToString(),
                topText);
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }
}
