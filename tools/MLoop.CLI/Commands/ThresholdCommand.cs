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
        var guaranteeOption = new Option<double?>("--guarantee")
        {
            Description = "Risk δ, above 0 and below 1 (e.g. 0.05): choose the lowest threshold whose precision is certified to reach --precision, " +
                          "wrong in at most δ of the files it could be run on — or refuse when no threshold on --data can be certified"
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
        command.Options.Add(guaranteeOption);
        command.Options.Add(dataOption);
        command.Options.Add(weightOption);
        command.Options.Add(clearOption);
        command.Options.Add(jsonOption);
        var recordsOption = DataFileOptions.Records();
        command.Options.Add(recordsOption);

        command.SetAction(parseResult => ExecuteAsync(
            parseResult.GetValue(experimentArg), parseResult.GetValue(nameOption), parseResult.GetValue(precisionOption),
            parseResult.GetValue(dataOption), parseResult.GetValue(weightOption), parseResult.GetValue(clearOption),
            parseResult.GetValue(jsonOption), parseResult.GetValue(recordsOption), parseResult.GetValue(precisionOfOption),
            parseResult.GetValue(guaranteeOption)));
        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? experimentId, string? modelName, double? precision, string? data, string? weight, bool clear, bool jsonOutput, string? records,
        string? precisionOf = null, double? guarantee = null)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;
        try
        {
            if (guarantee is { } risk && (!(risk > 0 && risk < 1) || double.IsNaN(risk)))
                return Fail($"--guarantee is a risk above 0 and below 1 (0.05 = wrong in at most one file of twenty), not {risk.ToString(CultureInfo.InvariantCulture)}.");
            if (guarantee is not null && precision is null)
                return Fail("--guarantee certifies the precision --precision asks for: give --precision too.");
            if (guarantee is not null && precision is 1)
                return Fail("A precision of 1 cannot be certified from rows: no finite file proves it. Ask for less, e.g. 0.99.");
            if (clear && precision is not null)
                return Fail("--clear removes the threshold and --precision chooses one: give one of them.");
            if (precision is { } p && (p <= 0 || p > 1 || double.IsNaN(p)))
                return Fail($"--precision is a share above 0 and at most 1, not {p.ToString(CultureInfo.InvariantCulture)}.");
            if (precisionOf is not null && precision is null)
                return Fail("--precision-of names the class whose precision --precision targets: give --precision too.");
            if (precision is not null && string.IsNullOrWhiteSpace(data))
                return Fail("--precision needs --data: the labeled rows the threshold is chosen on.");

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
            if (TaskTypes.Canonical(experiment.Task) != "binary-classification")
                return Fail($"A decision threshold is a binary classifier's; {id} is {experiment.Task}.");

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

            var dataPath = ProjectPaths.Resolve(data!, projectRoot);
            if (!TabularDataFile.Exists(dataPath))
                return Fail($"Data not found: {data}");
            dataPath = await TabularDataFile.AsCsvAsync(dataPath, records);
            var (weightColumn, weightWarning) = ModelComparison.ResolveWeight(weight, experiment, dataPath);
            if (weightWarning is not null)
                WarningConsole.Warn(Markup.Escape(weightWarning));

            var (ofPositive, className, classError) = ResolveClass(precisionOf, vocabulary);
            if (classError is not null)
                return Fail(classError);

            var rows = await new EvaluationEngine().ThresholdRowsAsync(modelPath, dataPath, experiment.Config.LabelColumn,
                experiment.Config.InputSchema, weightColumn);
            if (rows is null)
                return Fail($"{id} gives no probability (its trainer is not calibrated), and a decision threshold is a probability.");
            (double Threshold, double Precision, double Recall) chosen;
            DecisionGuarantee? certificate = null;
            if (guarantee is { } delta)
            {
                if (PrecisionGuarantee.Choose(rows, target, delta, ofPositive) is not { } certified)
                {
                    var needed = PrecisionGuarantee.MinimumRows(target, delta);
                    return Fail(string.Create(CultureInfo.InvariantCulture,
                        $"No threshold on {Path.GetFileName(data)} is certified to reach a precision of {target:0.###} for {className ?? "the positive class"} at risk {delta:0.###}: " +
                        $"the strictest threshold that can be tested answers at least {needed:N0} rows, every one right, and the walk down from there stops at the first threshold whose precision it cannot reject. " +
                        $"The file has {rows.Count(r => r.Positive == ofPositive):N0} rows of that class. Nothing was written — a larger file, a lower target or a larger risk is what it takes."));
                }
                chosen = (certified.Threshold, certified.Precision, certified.Recall);
                certificate = certified.Guarantee;
            }
            else if (DecisionThreshold.Choose(rows, target, ofPositive) is { } usual)
                chosen = usual;
            else
                return Fail(string.Create(CultureInfo.InvariantCulture,
                    $"No threshold reaches a precision of {target:0.###} for {className ?? "the positive class"} on {Path.GetFileName(data)} — not even answering it only for the rows the model is most confident about."));

            var evidence = DecisionThreshold.EvidenceFor(rows, chosen.Threshold, chosen.Precision, ofPositive);
            var threshold = new DecisionThreshold(chosen.Threshold, target, chosen.Precision, chosen.Recall, rows.Count,
                Path.GetFileName(data!), weightColumn, DateTimeOffset.UtcNow, className,
                evidence.Answered, evidence.ClassRows, evidence.LowerBound, certificate);
            // A certified threshold has said what its file proves; the warning is for the one that has not.
            if (certificate is null && DecisionThreshold.IsUncertified(target, evidence.LowerBound))
                WarningConsole.Warn(Markup.Escape(
                    string.Create(CultureInfo.InvariantCulture,
                        $"The precision {chosen.Precision:0.000} was measured on {evidence.Answered:N0} rows the threshold answers as {className ?? "the positive class"} ") +
                    string.Create(CultureInfo.InvariantCulture,
                        $"({evidence.ClassRows:N0} of that class in the file): at 95 % confidence it is at least {evidence.LowerBound:0.000}, {target - evidence.LowerBound:0.000} below the target {target:0.###}. ") +
                    "The threshold is the lowest one that reached the target on this very file, so new data usually does worse than this reading — " +
                    "choose it on a larger file, or read the target as not yet certified."));
            foreach (var path in new[] { modelPath, productionModel }.OfType<string>())
                threshold.WriteBeside(path);
            return Emit(name, id, threshold, "chosen", productionModel is not null, jsonOutput, vocabulary);
        }
        catch (Exception ex)
        {
            if (jsonOutput)
                return Fail(ex.Message);
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
            Console.WriteLine(JsonSerializer.Serialize(new { model, experimentId, status, inProduction, threshold, warnings = JsonOutputScope.Warnings },
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
            $"over {threshold.Rows:N0} rows of {Markup.Escape(threshold.ChosenOn)}{(threshold.WeightColumn is { } w ? $", weighted by {Markup.Escape(w)}" : "")}" +
            $"{(threshold is { Answered: { } answered, PrecisionLowerBound: { } bound } ? $"; it answers {answered:N0} of them, and that precision is certified to {bound:0.000} at 95 %" : "")}." +
            $"{(threshold.Guarantee is { } g ? $" Guaranteed: the true precision is at least {g.LowerBound:0.000}, wrong in at most {g.Delta:0.###} of the files this could be run on." : "")}"));
        if (status == "chosen" && !inProduction)
            AnsiConsole.MarkupLineInterpolated($"[grey]It goes to production with the model: mloop promote {experimentId}[/]");
        return 0;
    }

    private static int Fail(string message)
    {
        ErrorConsole.Error(Markup.Escape(message));
        return 1;
    }
}
