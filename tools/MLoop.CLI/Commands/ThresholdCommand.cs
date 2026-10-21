using System.CommandLine;
using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop threshold - Chooses a binary classifier's decision threshold for a target precision of one of
/// its classes on an evaluation set, shows the one it has, or removes it.
/// </summary>
/// <remarks>
/// The threshold lives beside the model (<see cref="DecisionThreshold"/>), so predict, serve, evaluate
/// and the gate answer at it once it is chosen, and promote carries it to production.
/// </remarks>
public static class ThresholdCommand
{
    public static Command Create()
    {
        var experimentArg = new Argument<string?>("experiment-id")
        {
            Description = "Experiment to set the threshold of (defaults to the production model)",
            Arity = ArgumentArity.ZeroOrOne
        };
        var nameOption = new Option<string?>("--name", "-n") { Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')" };
        var precisionOption = new Option<double?>("--precision")
        {
            Description = "Target precision, from 0 to 1, of the positive class (or of --precision-of): the threshold that reaches it on --data while keeping the most of that class's recall is chosen"
        };
        var precisionOfOption = new Option<string?>("--precision-of")
        {
            Description = "The class whose precision --precision targets, by its label value (default: the positive class) — e.g. the class a wrong answer is costly for"
        };
        var dataOption = new Option<string?>("--data") { Description = "Labeled evaluation file the threshold is chosen on (required with --precision)" };
        var weightOption = new Option<string?>("--weight") { Description = "Column that weights the rows (defaults as `mloop evaluate --weight` does)" };
        var clearOption = new Option<bool>("--clear") { Description = "Remove the threshold: the model answers at the trainer's own 0.5 again" };
        var jsonOption = new Option<bool>("--json") { Description = "Output the result as JSON" };

        var command = new Command("threshold", "Choose a binary classifier's decision threshold for a target precision, or show or remove it");
        command.Arguments.Add(experimentArg);
        command.Options.Add(nameOption);
        command.Options.Add(precisionOption);
        command.Options.Add(precisionOfOption);
        command.Options.Add(dataOption);
        command.Options.Add(weightOption);
        command.Options.Add(clearOption);
        command.Options.Add(jsonOption);
        var recordsOption = DataFileOptions.Records();
        command.Options.Add(recordsOption);

        command.SetAction(parseResult => ExecuteAsync(
            parseResult.GetValue(experimentArg), parseResult.GetValue(nameOption), parseResult.GetValue(precisionOption),
            parseResult.GetValue(dataOption), parseResult.GetValue(weightOption), parseResult.GetValue(clearOption),
            parseResult.GetValue(jsonOption), parseResult.GetValue(recordsOption), parseResult.GetValue(precisionOfOption)));
        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? experimentId, string? modelName, double? precision, string? data, string? weight, bool clear, bool jsonOutput, string? records, string? precisionOf = null)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;
        try
        {
            if (clear && precision is not null)
                return Fail("--clear removes the threshold and --precision chooses one: give one of them.", jsonOutput);
            if (precision is { } p && (p <= 0 || p > 1 || double.IsNaN(p)))
                return Fail($"--precision is a share above 0 and at most 1, not {p.ToString(CultureInfo.InvariantCulture)}.", jsonOutput);
            if (precisionOf is not null && precision is null)
                return Fail("--precision-of names the class whose precision --precision targets: give --precision too.", jsonOutput);
            if (precision is not null && string.IsNullOrWhiteSpace(data))
                return Fail("--precision needs --data: the labeled rows the threshold is chosen on.", jsonOutput);

            var fileSystem = new FileSystemManager();
            var discovery = new ProjectDiscovery(fileSystem);
            string projectRoot;
            try { projectRoot = discovery.FindRoot(); }
            catch (InvalidOperationException) { return Fail(ProjectDiscovery.NotInsideProjectCause, jsonOutput); }

            var name = CommandContext.ResolveModelName(modelName);
            var store = new ExperimentStore(fileSystem, discovery);
            var registry = new ModelRegistry(fileSystem, discovery, store);
            var production = await registry.GetProductionAsync(name, CancellationToken.None);
            var id = string.IsNullOrWhiteSpace(experimentId) ? production?.ExperimentId : experimentId.Trim();
            if (id is null)
                return Fail($"No model is in production for '{name}' — name an experiment.", jsonOutput);
            var (modelPath, refusal) = ExperimentModels.Find(store, name, id);
            if (modelPath is null)
                return Fail(refusal!, jsonOutput);
            var experiment = await store.LoadAsync(name, id, CancellationToken.None);
            if (TaskTypes.Canonical(experiment.Task) != "binary-classification")
                return Fail($"A decision threshold is a binary classifier's; {id} is {experiment.Task}.", jsonOutput);

            // The class is named the way the label was written; which side of the threshold it is on
            // is the training conversion's decision, read from the saved schema.
            var vocabulary = BinaryLabelVocabulary.Of(experiment.Config.InputSchema);
            // Production answers from its own copy: a threshold set on the experiment in production is set there too.
            var productionModel = production?.ExperimentId.Equals(id, StringComparison.OrdinalIgnoreCase) == true
                ? Path.Combine(registry.GetProductionPath(name), ExperimentLayout.ModelFileName)
                : null;

            if (clear)
            {
                foreach (var path in new[] { modelPath, productionModel }.OfType<string>())
                    if (File.Exists(DecisionThreshold.PathBeside(path)))
                        File.Delete(DecisionThreshold.PathBeside(path));
                return Emit(name, id, null, "cleared", productionModel is not null, jsonOutput, vocabulary);
            }

            if (precision is not { } target)
                return Emit(name, id, DecisionThreshold.ReadBeside(modelPath), "current", productionModel is not null, jsonOutput, vocabulary);

            var dataPath = Path.GetFullPath(Path.IsPathRooted(data!) ? data! : Path.Combine(projectRoot, data!));
            if (!TabularDataFile.Exists(dataPath))
                return Fail($"Data not found: {data}", jsonOutput);
            dataPath = await TabularDataFile.AsCsvAsync(dataPath, records);
            var (weightColumn, weightWarning) = ModelComparison.ResolveWeight(weight, experiment, dataPath);
            if (weightWarning is not null)
                WarningConsole.Warn(Markup.Escape(weightWarning));

            var (ofPositive, className, classError) = ResolveClass(precisionOf, vocabulary);
            if (classError is not null)
                return Fail(classError, jsonOutput);

            var rows = await new EvaluationEngine().ThresholdRowsAsync(modelPath, dataPath, experiment.Config.LabelColumn,
                experiment.Config.InputSchema, weightColumn);
            if (rows is null)
                return Fail($"{id} gives no probability (its trainer is not calibrated), and a decision threshold is a probability.", jsonOutput);
            if (DecisionThreshold.Choose(rows, target, ofPositive) is not { } chosen)
                return Fail(string.Create(CultureInfo.InvariantCulture,
                    $"No threshold reaches a precision of {target:0.###} for {className ?? "the positive class"} on {Path.GetFileName(data)} — not even answering it only for the rows the model is most confident about."), jsonOutput);

            var threshold = new DecisionThreshold(chosen.Threshold, target, chosen.Precision, chosen.Recall, rows.Count,
                Path.GetFileName(data!), weightColumn, DateTimeOffset.UtcNow, className);
            foreach (var path in new[] { modelPath, productionModel }.OfType<string>())
                threshold.WriteBeside(path);
            return Emit(name, id, threshold, "chosen", productionModel is not null, jsonOutput, vocabulary);
        }
        catch (Exception ex)
        {
            if (jsonOutput)
                return Fail(ex.Message, jsonOutput: true);
            ErrorSuggestions.DisplayError(ex, "threshold");
            return 1;
        }
    }

    /// <summary>
    /// Which class <paramref name="precisionOf"/> names: positive or negative, and the name to record
    /// (<c>null</c> = the positive class, unnamed — the form every older threshold has). A model whose
    /// label was already Boolean has no names, so it answers to <c>true</c>/<c>false</c>.
    /// </summary>
    internal static (bool OfPositive, string? ClassName, string? Error) ResolveClass(
        string? precisionOf, (string Negative, string Positive)? vocabulary)
    {
        if (string.IsNullOrWhiteSpace(precisionOf))
            return (true, null, null);
        var asked = precisionOf.Trim();
        var (negative, positive) = vocabulary ?? ("false", "true");
        if (asked.Equals(positive, StringComparison.OrdinalIgnoreCase))
            return (true, positive, null);
        if (asked.Equals(negative, StringComparison.OrdinalIgnoreCase))
            return (false, negative, null);
        return (true, null, $"--precision-of {asked} is not one of this model's classes: {negative}, {positive}.");
    }

    private static int Emit(string model, string experimentId, DecisionThreshold? threshold, string status, bool inProduction, bool jsonOutput,
        (string Negative, string Positive)? vocabulary = null)
    {
        if (jsonOutput)
        {
            Console.WriteLine(JsonSerializer.Serialize(new { model, experimentId, status, inProduction, threshold },
                new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));
            return 0;
        }
        var where = inProduction ? " (in production)" : "";
        if (threshold is null)
        {
            if (status == "cleared")
                AnsiConsole.MarkupLineInterpolated($"[green]>[/] {experimentId}{where} answers at the trainer's own threshold (0.5) again.");
            else
                AnsiConsole.MarkupLineInterpolated($"[grey]{experimentId}{where} has no decision threshold of its own: it answers {vocabulary?.Positive ?? "positive"} from a probability of 0.5.[/]");
            return 0;
        }
        AnsiConsole.MarkupLine(string.Create(CultureInfo.InvariantCulture,
            $"[green]>[/] {Markup.Escape(experimentId)}{Markup.Escape(where)} answers {Markup.Escape(vocabulary?.Positive ?? "positive")} from a probability of [cyan]{threshold.Threshold:0.####}[/]" +
            $"{(vocabulary is { } v ? $" ({Markup.Escape(v.Negative)} below it)" : "")} — " +
            $"{(threshold.PrecisionOf is { } c ? $"{Markup.Escape(c)} " : "")}precision {threshold.Precision:0.000} (target {threshold.TargetPrecision:0.###}), recall {threshold.Recall:0.000} " +
            $"over {threshold.Rows:N0} rows of {Markup.Escape(threshold.ChosenOn)}{(threshold.WeightColumn is { } w ? $", weighted by {Markup.Escape(w)}" : "")}."));
        if (status == "chosen" && !inProduction)
            AnsiConsole.MarkupLineInterpolated($"[grey]It goes to production with the model: mloop promote {experimentId}[/]");
        return 0;
    }

    private static int Fail(string message, bool jsonOutput)
    {
        if (jsonOutput)
        {
            JsonError.Emit(message);
            Console.Error.WriteLine(message);
        }
        else
        {
            ErrorConsole.Error(message);
        }
        return 1;
    }
}
