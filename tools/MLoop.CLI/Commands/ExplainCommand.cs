using System.CommandLine;
using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop explain - Which input columns a trained model relies on: the fall of its score when a column's values are shuffled among
/// the rows of a labeled file (permutation importance), column by column.
/// </summary>
/// <remarks>
/// Not <c>mloop analyze importance</c>, which reads the data: how strongly each column relates to the label, whatever model is trained
/// on it. This reads a model: a column it never used falls by nothing however strongly it relates to the label.
/// </remarks>
public static class ExplainCommand
{
    /// <summary>
    /// How many columns the table lists. On a table of hundreds of columns the model relies on a handful and the rest fall the score by
    /// about nothing (the largest of two hundred ninety-five noise columns fell it by 0.007 against 0.015 for the weakest of five real
    /// ones): listed in full they bury the answer. <c>--json</c> carries every column.
    /// </summary>
    internal const int DisplayedColumns = 20;

    /// <summary>
    /// Scorings from which a <c>--json</c> run narrates its progress. Each takes about 0.05–0.13 s on a few hundred to a thousand columns
    /// (measured), so below this a run ends within the time a caller waits without a sign of life.
    /// </summary>
    internal const int NarratedFrom = 300;

    /// <summary>
    /// The line to write when <paramref name="done"/> of <paramref name="total"/> scorings have finished and a new tenth of them has been
    /// reached since <paramref name="reportedTenth"/>; null when it is not yet time, or the run is too short to narrate.
    /// </summary>
    internal static string? ProgressNote(int done, int total, ref int reportedTenth)
    {
        if (total < NarratedFrom)
            return null;
        var tenth = done * 10 / total;
        if (tenth <= reportedTenth)
            return null;
        reportedTenth = tenth;
        return $"explain: {tenth * 10}% ({done} of {total} scorings)";
    }

    public static Command Create()
    {
        var experimentArg = new Argument<string?>("experiment-id")
        {
            Description = "Experiment to explain (defaults to the production model)",
            Arity = ArgumentArity.ZeroOrOne
        };
        var nameOption = new Option<string?>("--name", "-n") { Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')" };
        var dataOption = new Option<string?>("--data", "-d") { Description = "Labeled file the model is scored on, with the label column (default: datasets/test.csv)" };
        var weightOption = new Option<string?>("--weight") { Description = "Column that weights the rows (defaults as `mloop evaluate --weight` does)" };
        var repeatsOption = new Option<int>("--repeats") { Description = "Shuffles per column; the importance is their mean", DefaultValueFactory = _ => ModelExplanation.DefaultRepeats };
        var rowsOption = new Option<int>("--rows") { Description = "Most rows scored; a larger file is sampled (every column costs --repeats scorings)", DefaultValueFactory = _ => ModelExplanation.DefaultRows };
        var seedOption = new Option<int>("--seed") { Description = "Seed of the sample and the shuffles — the same seed gives the same numbers", DefaultValueFactory = _ => ModelExplanation.DefaultSeed };
        var jsonOption = new Option<bool>("--json") { Description = "Output the result as JSON" };
        var recordsOption = DataFileOptions.Records();

        var command = new Command("explain", "Show which input columns a trained model relies on (permutation importance on a labeled file)");
        command.Arguments.Add(experimentArg);
        command.Options.Add(nameOption);
        command.Options.Add(dataOption);
        command.Options.Add(weightOption);
        command.Options.Add(repeatsOption);
        command.Options.Add(rowsOption);
        command.Options.Add(seedOption);
        command.Options.Add(jsonOption);
        command.Options.Add(recordsOption);

        command.SetAction(parseResult => ExecuteAsync(
            parseResult.GetValue(experimentArg), parseResult.GetValue(nameOption), parseResult.GetValue(dataOption), parseResult.GetValue(weightOption),
            parseResult.GetValue(repeatsOption), parseResult.GetValue(rowsOption), parseResult.GetValue(seedOption),
            parseResult.GetValue(jsonOption), parseResult.GetValue(recordsOption)));
        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? experimentId, string? modelName, string? data, string? weight, int repeats, int rows, int seed, bool jsonOutput, string? records)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;
        try
        {
            if (repeats < 1)
                return Fail("--repeats is at least 1: one shuffle per column.");
            if (rows < 2)
                return Fail("--rows is at least 2: one row cannot be shuffled.");

            var fileSystem = new FileSystemManager();
            var discovery = new ProjectDiscovery(fileSystem);
            string projectRoot;
            try { projectRoot = discovery.FindRoot(); }
            catch (InvalidOperationException) { return Fail(ProjectDiscovery.NotInsideProjectCause); }

            var name = CommandContext.ResolveModelName(modelName);
            var store = new ExperimentStore(fileSystem, discovery);
            var registry = new ModelRegistry(fileSystem, discovery, store);
            var production = await registry.GetProductionAsync(name, CancellationToken.None);
            var id = string.IsNullOrWhiteSpace(experimentId) ? production?.ExperimentId : experimentId.Trim();
            if (id is null)
                return Fail($"No model is in production for '{name}' — name an experiment.");
            var (modelPath, refusal) = ExperimentModels.Find(store, name, id);
            if (modelPath is null)
                return Fail(refusal!);
            var experiment = await store.LoadAsync(name, id, CancellationToken.None);
            if (!ModelExplanation.Supports(experiment.Task))
                return Fail($"Permutation importance explains binary-classification, multiclass-classification and regression models; {id} is {experiment.Task}.");

            string? dataPath = null;
            if (!string.IsNullOrWhiteSpace(data))
            {
                dataPath = ProjectPaths.Resolve(data, projectRoot);
                if (!TabularDataFile.Exists(dataPath))
                    return Fail($"Data not found: {data}");
                dataPath = await TabularDataFile.AsCsvAsync(dataPath, records);
            }
            else if (new DatasetDiscovery(fileSystem).FindDatasets(projectRoot)?.TestPath is { } discovered)
            {
                dataPath = discovered;
            }
            if (dataPath is null)
                return Fail("Give the labeled file to score the model on with --data (or keep one as datasets/test.csv).");

            var (weightColumn, weightWarning) = ModelComparison.ResolveWeight(weight, experiment, dataPath);
            if (weightWarning is not null)
                WarningConsole.Warn(Markup.Escape(weightWarning));

            ModelExplanation.Explanation explanation = null!;
            if (jsonOutput)
            {
                // No progress bar in machine mode, and a thousand columns take minutes: say how far it is on stderr, a tenth at a time.
                var reportedTenth = 0;
                explanation = await ModelExplanation.ExplainAsync(new EvaluationEngine(), modelPath, dataPath, experiment, weightColumn, repeats, rows, seed,
                    (done, total) =>
                    {
                        if (ProgressNote(done, total, ref reportedTenth) is { } note)
                            AnsiConsole.WriteLine(note);
                    });
            }
            else
                await AnsiConsole.Progress().StartAsync(async ctx =>
                {
                    var task = ctx.AddTask("[yellow]Shuffling columns...[/]", maxValue: 1);
                    explanation = await ModelExplanation.ExplainAsync(new EvaluationEngine(), modelPath, dataPath, experiment, weightColumn, repeats, rows, seed,
                        (done, total) => { task.MaxValue = total; task.Value = done; });
                });

            return Emit(name, id, dataPath, weightColumn, explanation, jsonOutput);
        }
        catch (Exception ex)
        {
            if (jsonOutput)
                return Fail(ex.Message);
            ErrorSuggestions.DisplayError(ex, "explain");
            return 1;
        }
    }

    private static int Emit(string model, string experimentId, string dataPath, string? weightColumn, ModelExplanation.Explanation e, bool jsonOutput)
    {
        if (jsonOutput)
        {
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                model,
                experimentId,
                data = Path.GetFileName(dataPath),
                weightColumn,
                metric = e.Metric,
                higherIsBetter = e.HigherIsBetter,
                baseline = e.Baseline,
                rows = e.Rows,
                repeats = e.Repeats,
                features = e.Features.Select(f => new { column = f.Column, importance = f.Importance, standardDeviation = f.StandardDeviation }),
                notShuffled = e.NotShuffled,
                warnings = JsonOutputScope.Warnings,
            }, new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));
            return 0;
        }

        AnsiConsole.MarkupLineInterpolated($"[bold]{experimentId}[/] on {Path.GetFileName(dataPath)}: {e.Rows:N0} rows, {e.Metric} {e.Baseline.ToString("0.0000", CultureInfo.InvariantCulture)} ({(e.HigherIsBetter ? "higher" : "lower")} is better), {e.Repeats} shuffle(s) per column");
        var table = new Table().Border(TableBorder.Rounded).AddColumn("Column").AddColumn(new TableColumn($"Fall in {e.Metric}").RightAligned()).AddColumn(new TableColumn("± sd").RightAligned());
        foreach (var f in e.Features.Take(DisplayedColumns))
            table.AddRow(Markup.Escape(f.Column), f.Importance.ToString("+0.0000;-0.0000;0.0000", CultureInfo.InvariantCulture), f.StandardDeviation.ToString("0.0000", CultureInfo.InvariantCulture));
        AnsiConsole.Write(table);
        if (e.Features.Count > DisplayedColumns)
            AnsiConsole.MarkupLineInterpolated($"[grey]... and {e.Features.Count - DisplayedColumns} more columns (the {DisplayedColumns} the model relies on most are shown; the next falls {e.Features[DisplayedColumns].Importance.ToString("+0.0000;-0.0000;0.0000", CultureInfo.InvariantCulture)}; --json lists every column)[/]");
        AnsiConsole.MarkupLine("[grey]A column's fall is how much the score drops when its values are shuffled among the rows: the model relies on it that much. Near zero: it does not.[/]");
        if (e.NotShuffled.Count > 0)
            AnsiConsole.MarkupLineInterpolated($"[grey]Not shuffled (the model does not read them as features): {string.Join(", ", e.NotShuffled)}[/]");
        return 0;
    }

    private static int Fail(string message)
    {
        ErrorConsole.Error(Markup.Escape(message));
        return 1;
    }
}
