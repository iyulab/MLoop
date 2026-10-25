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

/// <summary>
/// mloop evaluate - Evaluates model performance on test data
/// </summary>
public static partial class EvaluateCommand
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

        var weightOption = new Option<string?>("--weight")
        {
            Description = "Column that weights the test rows (e.g. 1/π of a recheck sample). Defaults to the column the experiment was trained with, when the file has it"
        };
        var againstOption = new Option<string?>("--against")
        {
            Description = "Also evaluate this experiment (or 'production') on the same rows and weights, and compare the two row by row"
        };
        var sliceByOption = new Option<string?>("--slice-by")
        {
            Description = "Also report the metrics separately for each value of this column of the test data"
        };

        var command = new Command("evaluate", "Evaluate model performance on test data");
        command.Arguments.Add(experimentArg);
        command.Arguments.Add(testDataArg);
        command.Options.Add(nameOption);
        command.Options.Add(jsonOption);
        command.Options.Add(sliceByOption);
        command.Options.Add(weightOption);
        command.Options.Add(againstOption);
        var recordsOption = DataFileOptions.Records();
        command.Options.Add(recordsOption);

        command.SetAction((parseResult) =>
        {
            var experimentId = parseResult.GetValue(experimentArg);
            var testDataFile = parseResult.GetValue(testDataArg);
            var name = parseResult.GetValue(nameOption);
            var json = parseResult.GetValue(jsonOption);
            return ExecuteAsync(experimentId, testDataFile, name, json, parseResult.GetValue(recordsOption),
                parseResult.GetValue(sliceByOption), parseResult.GetValue(weightOption), parseResult.GetValue(againstOption));
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
        string? skipped = null,
        string? sliceBy = null,
        IReadOnlyList<EvaluationSlices.Slice>? slices = null,
        string? weightColumn = null,
        Comparison? comparison = null,
        ConfusionCounts? confusion = null,
        IReadOnlyList<string>? warnings = null)
    {
        warnings ??= JsonOutputScope.Warnings;
        var payload = new
        {
            WeightColumn = weightColumn,
            Comparison = comparison,
            Confusion = confusion,
            Model = model,
            Skipped = skipped,
            ExperimentId = experimentId,
            TestDataFile = testDataFile,
            TrainingMetrics = trainingMetrics,
            TestMetrics = testMetrics,
            PossibleOverfitting = possibleOverfitting,
            SliceBy = sliceBy,
            Slices = slices,
            Warnings = warnings is { Count: > 0 } ? warnings : null,
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
        bool jsonOutput = false,
        string? records = null,
        string? sliceBy = null,
        string? weight = null,
        string? against = null)
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
            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

            var model = await ResolveModelAsync(experimentId, resolvedModelName, jsonOutput, fileSystem, experimentStore, modelRegistry);
            if (model.ExitCode is { } exitCode)
                return exitCode;
            var (resolvedModelPath, resolvedExperimentId, experimentData, _) = model;

            // Directory-based tasks (image classification, object detection) evaluate over an image
            // directory — object detection scored by mAP, image classification by multiclass accuracy
            // — so they bypass the CSV-file test-data resolution and CSV schema validation below.
            bool isDirectoryBased = DataLoaderFactory.IsDirectoryBased(experimentData?.Task);

            var resolvedTestDataFile = ResolveTestData(testDataFile, experimentData, projectRoot, fileSystem);
            if (resolvedTestDataFile is null)
                return 1;

            // Validate schema before evaluation (CSV-column validation; not applicable to image directories)
            if (!isDirectoryBased)
            {
                var validated = await ValidateTestDataAsync(
                    resolvedTestDataFile, records, experimentData!, resolvedModelName, resolvedExperimentId,
                    fileSystem, projectDiscovery);
                if (validated is null)
                    return 1;
                resolvedTestDataFile = validated;
            }

            // The weight the test rows count by: the one asked for, else the one the experiment was
            // trained with when this file carries it. Training metrics recorded under a weight and test
            // metrics without one describe different distributions; saying so is the least owed.
            var (weightColumn, weightWarning) = ModelComparison.ResolveWeight(weight, experimentData!, resolvedTestDataFile);
            if (weightWarning is not null)
                WarningConsole.Warn(Markup.Escape(weightWarning));

            // Perform evaluation with progress indicator
            Dictionary<string, double>? testMetrics = null;
            ConfusionCounts? confusion = null;

            await AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .StartAsync("[yellow]Evaluating model...[/]", async ctx =>
                {
                    var evaluationEngine = new EvaluationEngine();

                    var evaluation = await evaluationEngine.EvaluateAsync(
                        resolvedModelPath,
                        resolvedTestDataFile,
                        experimentData!.Config.LabelColumn,
                        experimentData.Task,
                        CancellationToken.None,
                        experimentData.Config.InputSchema,
                        experimentData.Config.GroupColumn,
                        experimentData.Config.UserColumn,
                        experimentData.Config.ItemColumn,
                        weightColumn,
                        new MLoop.Core.Metrics.MetricEngine(projectRoot, new ConsoleLogger()));
                    (testMetrics, confusion) = (evaluation.Metrics, evaluation.Confusion);

                    ctx.Status("[green]Evaluation complete![/]");
                });

            var (overfitting, trainingMetrics) = ShowEvaluation(
                experimentData!, testMetrics!, confusion, weightColumn, resolvedModelName, resolvedExperimentId);

            IReadOnlyList<EvaluationSlices.Slice>? slices = null;
            if (!string.IsNullOrWhiteSpace(sliceBy))
            {
                slices = await EvaluationSlices.EvaluateAsync(new EvaluationEngine(), sliceBy.Trim(), resolvedTestDataFile, resolvedModelPath, experimentData!, weightColumn);
                ShowSlices(sliceBy.Trim(), slices, testMetrics!.Keys.ToList());
            }

            Comparison? comparison = null;
            if (!string.IsNullOrWhiteSpace(against))
            {
                var baseline = await ModelComparison.ResolveBaselineAsync(against.Trim(), resolvedModelName, resolvedExperimentId, experimentData!,
                    experimentStore, modelRegistry);
                if (baseline.Error is { } error)
                {
                    ErrorConsole.Error(error);
                    return 1;
                }
                comparison = await ModelComparison.CompareAsync(new EvaluationEngine(), resolvedExperimentId, resolvedModelPath, experimentData!,
                    baseline, resolvedTestDataFile, weightColumn);
                ShowComparison(comparison, testMetrics!);
            }

            if (jsonOutput)
                EmitJson(resolvedModelName, resolvedExperimentId, resolvedTestDataFile,
                    trainingMetrics, testMetrics, overfitting, sliceBy: sliceBy?.Trim(), slices: slices, weightColumn: weightColumn,
                    comparison: comparison, confusion: confusion);

            return 0;
        }
        catch (Exception ex)
        {
            ErrorSuggestions.DisplayError(ex, "evaluate");
            return 1;
        }
    }

    /// <summary>Above this many classes the matrix stays in <c>--json</c>: a wider table does not fit a terminal.</summary>
    internal const int MaxConfusionClassesShown = 10;

    private static void ShowConfusion(ConfusionCounts confusion, bool weighted)
    {
        if (confusion.Classes.Count > MaxConfusionClassesShown)
        {
            AnsiConsole.MarkupLine($"[grey]{confusion.Classes.Count} classes — the confusion matrix is in --json (confusion).[/]");
            AnsiConsole.WriteLine();
            return;
        }
        // The caption is a line of its own: as a table title it wrapped to the width of a two-class table.
        AnsiConsole.MarkupLine("[bold]Confusion[/] [grey](rows: actual · columns: predicted)[/]");
        var table = new Table().Border(TableBorder.Rounded).AddColumn("");
        foreach (var c in confusion.Classes)
            table.AddColumn(new TableColumn(Markup.Escape(c)).RightAligned());
        for (var i = 0; i < confusion.Classes.Count; i++)
        {
            var cells = new List<string> { Markup.Escape(confusion.Classes[i]) };
            for (var j = 0; j < confusion.Classes.Count; j++)
            {
                var n = confusion.Counts[i][j];
                var text = weighted ? n.ToString("0.##", CultureInfo.InvariantCulture) : n.ToString("0", CultureInfo.InvariantCulture);
                cells.Add(i == j ? $"[green]{text}[/]" : n > 0 ? text : $"[grey]{text}[/]");
            }
            table.AddRow(cells.ToArray());
        }
        AnsiConsole.Write(table);
        if (weighted)
            AnsiConsole.MarkupLine("[grey]Counts are weighted sums.[/]");
        AnsiConsole.WriteLine();
    }

    private static void ShowComparison(Comparison comparison, Dictionary<string, double> candidateMetrics)
    {
        AnsiConsole.Write(new Rule($"[green]Against {Markup.Escape(comparison.Against)}[/]").LeftJustified());
        var table = new Table().Border(TableBorder.Rounded)
            .AddColumn("Metric")
            .AddColumn(new TableColumn("This experiment").RightAligned())
            .AddColumn(new TableColumn(Markup.Escape(comparison.Against)).RightAligned())
            .AddColumn(new TableColumn("Difference").RightAligned());
        foreach (var (name, value) in candidateMetrics)
        {
            if (!comparison.Metrics.TryGetValue(name, out var other))
                continue;
            table.AddRow(Markup.Escape(name), value.ToString("F4", CultureInfo.InvariantCulture),
                other.ToString("F4", CultureInfo.InvariantCulture), FormatMetricDifference(name, value - other));
        }
        AnsiConsole.Write(table);
        if (comparison.Paired is { Rows: > 0 } p)
        {
            var measure = p.Measure == PairedComparison.ErrorRate ? "error rate" : "squared error";
            var verdict = p.Verdict switch
            {
                PairedVerdict.CandidateBetter => "[green]this experiment is better on these rows[/]",
                PairedVerdict.BaselineBetter => $"[red]{Markup.Escape(comparison.Against)} is better on these rows[/]",
                _ => "[yellow]the two cannot be told apart on these rows[/]",
            };
            AnsiConsole.MarkupLine(
                $"Paired over {p.Rows:N0} rows, by {measure}: {p.Candidate:F4} vs {p.Baseline:F4} — difference {p.Difference:+0.0000;-0.0000;0.0000} " +
                $"(95% interval {p.Low:+0.0000;-0.0000;0.0000} to {p.High:+0.0000;-0.0000;0.0000}) — {verdict}.");
        }
        else
        {
            AnsiConsole.MarkupLine($"[grey]Not paired: {Markup.Escape(comparison.NotPaired ?? "no row both models answered with a label.")}[/]");
        }
        AnsiConsole.WriteLine();
    }

    private static void ShowSlices(string sliceBy, IReadOnlyList<EvaluationSlices.Slice> slices, IReadOnlyList<string> metricNames)
    {
        AnsiConsole.Write(new Rule($"[green]By {Markup.Escape(sliceBy)}[/]").LeftJustified());

        var labelWidth = slices.Select(s => s.Value.Length).Append(sliceBy.Length).Max();
        var rowsWidth = slices.Select(s => s.Rows.ToString(CultureInfo.InvariantCulture).Length + 1).Append("Rows".Length).Max();
        string Cell(EvaluationSlices.Slice slice, string name) =>
            slice.Metrics?.TryGetValue(name, out var v) == true && !double.IsNaN(v)
                ? v.ToString("F4", CultureInfo.InvariantCulture)
                : "-";
        var metrics = metricNames
            .Select(name => (name, slices.Select(s => Cell(s, name).Length).Append(name.Length).Max()))
            .ToList();
        foreach (var group in SliceMetricGroups(labelWidth, rowsWidth, metrics, AnsiConsole.Profile.Width))
        {
            var table = new Table().Border(TableBorder.Rounded);
            table.AddColumn(Markup.Escape(sliceBy));
            table.AddColumn(new TableColumn("Rows").RightAligned());
            foreach (var name in group)
                table.AddColumn(new TableColumn(Markup.Escape(name)).RightAligned());

            foreach (var slice in slices)
            {
                var cells = new List<string>
                {
                    Markup.Escape(slice.Value),
                    slice.Small ? $"[yellow]{slice.Rows}*[/]" : slice.Rows.ToString(CultureInfo.InvariantCulture),
                };
                foreach (var name in group)
                    cells.Add(Cell(slice, name) is "-" ? "[grey]-[/]" : Cell(slice, name));
                table.AddRow(cells.ToArray());
            }
            AnsiConsole.Write(table);
        }

        if (slices.Any(s => s.Small))
            AnsiConsole.MarkupLine($"[grey]* fewer than {EvaluationSlices.SmallRows} rows — these metrics vary a lot between samples[/]");
        foreach (var failed in slices.Where(s => s.Error != null))
            AnsiConsole.MarkupLineInterpolated($"[grey]{failed.Value}: not scored — {failed.Error}[/]");
        AnsiConsole.WriteLine();
    }

    /// <summary>
    /// The slice table's metrics, split into groups that each fit the console beside the slice and row
    /// columns. One table carrying every metric made Spectre wrap the numbers themselves at 80 columns —
    /// <c>0.862</c> on one line and <c>1</c> under it — so a value could not be read at all.
    /// </summary>
    internal static IReadOnlyList<IReadOnlyList<string>> SliceMetricGroups(
        int labelWidth, int rowsWidth, IReadOnlyList<(string Name, int Width)> metrics, int consoleWidth)
    {
        // A rounded table draws one border on the left, then each column's text, a space either side
        // and the border after it.
        static int Column(int text) => text + 3;
        var fixedWidth = 1 + Column(labelWidth) + Column(rowsWidth);

        var groups = new List<IReadOnlyList<string>>();
        var current = new List<string>();
        var width = fixedWidth;
        foreach (var (name, textWidth) in metrics)
        {
            var column = Column(textWidth);
            if (current.Count > 0 && width + column > consoleWidth)
            {
                groups.Add(current);
                current = [];
                width = fixedWidth;
            }
            current.Add(name);
            width += column;
        }
        if (current.Count > 0 || groups.Count == 0)
            groups.Add(current);
        return groups;
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
