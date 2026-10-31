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
/// mloop calibrate - Fits a classifier's probability calibration on labeled rows, without retraining,
/// shows the one it has, or removes it.
/// </summary>
/// <remarks>
/// The calibration lives beside the model (<see cref="ProbabilityCalibration"/>), so predict, serve,
/// evaluate and a threshold chosen afterwards read the calibrated probability, and promote carries it.
/// It is refused while the model has a decision threshold: that threshold was chosen on the
/// probabilities the calibration replaces, and would no longer mean the precision it was chosen for.
/// </remarks>
public static class CalibrateCommand
{
    public static Command Create()
    {
        var experimentArg = new Argument<string?>("experiment-id")
        {
            Description = "Experiment to calibrate (defaults to the production model)",
            Arity = ArgumentArity.ZeroOrOne
        };
        var nameOption = new Option<string?>("--name", "-n") { Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')" };
        var dataOption = new Option<string?>("--data") { Description = "Labeled rows the calibration is fit on — rows the model was not trained on" };
        var methodOption = new Option<string?>("--method")
        {
            Description = "Binary: platt (default — two parameters, steady on a few hundred rows) or isotonic (no shape assumed, needs more rows). Multiclass: temperature (default — one parameter, answers unchanged)"
        };
        var weightOption = new Option<string?>("--weight") { Description = "Column that weights the rows (defaults as `mloop evaluate --weight` does)" };
        var clearOption = new Option<bool>("--clear") { Description = "Remove the calibration: the model answers with its own probabilities again" };
        var jsonOption = new Option<bool>("--json") { Description = "Output the result as JSON" };

        var command = new Command("calibrate", "Fit a classifier's probability calibration on labeled rows without retraining, or show or remove it");
        command.Arguments.Add(experimentArg);
        command.Options.Add(nameOption);
        command.Options.Add(dataOption);
        command.Options.Add(methodOption);
        command.Options.Add(weightOption);
        command.Options.Add(clearOption);
        command.Options.Add(jsonOption);
        var recordsOption = DataFileOptions.Records();
        command.Options.Add(recordsOption);

        command.SetAction(parseResult => ExecuteAsync(
            parseResult.GetValue(experimentArg), parseResult.GetValue(nameOption), parseResult.GetValue(dataOption),
            parseResult.GetValue(methodOption), parseResult.GetValue(weightOption), parseResult.GetValue(clearOption),
            parseResult.GetValue(jsonOption), parseResult.GetValue(recordsOption)));
        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? experimentId, string? modelName, string? data, string? method, string? weight, bool clear, bool jsonOutput, string? records)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;
        try
        {
            if (clear && data is not null)
                return Fail("--clear removes the calibration and --data fits one: give one of them.");

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
            var task = TaskTypes.Canonical(experiment.Task);
            var methods = task switch
            {
                "binary-classification" => ProbabilityCalibration.BinaryMethods,
                "multiclass-classification" => ProbabilityCalibration.MulticlassMethods,
                _ => null,
            };
            if (methods is null)
                return Fail($"Probability calibration is a classifier's; {id} is {experiment.Task}.");
            method = method?.Trim().ToLowerInvariant() ?? methods[0];
            if (!methods.Contains(method))
                return Fail($"--method {method} does not calibrate a {task} model: {string.Join(", ", methods)}.");
            var binary = task == "binary-classification";

            // Production answers from its own copy: a calibration fit on the experiment in production is set there too.
            var productionModel = production?.ExperimentId.Equals(id, StringComparison.OrdinalIgnoreCase) == true
                ? Path.Combine(registry.GetProductionPath(name), ExperimentLayout.ModelFileName)
                : null;
            var targets = new[] { modelPath, productionModel }.OfType<string>().ToArray();

            if (clear)
            {
                // The file also holds the calibration training fitted (out of fold), which the experiment's recorded metrics describe:
                // removing it is not undoing something the user did, and only retraining brings it back.
                if (ProbabilityCalibration.ReadBeside(modelPath) is { FitOn: ProbabilityCalibration.TrainingFitSource })
                    WarningConsole.Warn(Markup.Escape(
                        $"The calibration beside {id} is the one training fitted on out-of-fold answers; the metrics recorded for {id} are the calibrated model's. " +
                        "Clearing it leaves the trainer's own probabilities, which can be far from calibrated, and only retraining brings the fitted one back."));
                foreach (var path in targets)
                    if (File.Exists(ProbabilityCalibration.PathBeside(path)))
                        File.Delete(ProbabilityCalibration.PathBeside(path));
                return Emit(name, id, null, "cleared", productionModel is not null, jsonOutput);
            }

            if (data is null)
                return Emit(name, id, ProbabilityCalibration.ReadBeside(modelPath), "current", productionModel is not null, jsonOutput);

            if (binary && DecisionThreshold.ReadBeside(modelPath) is not null)
                return Fail($"{id} has a decision threshold, chosen on the probabilities a calibration replaces — it would no longer reach the precision it was chosen for. " +
                            $"Remove it (mloop threshold {id} --clear), calibrate, then choose it again.");

            var dataPath = ProjectPaths.Resolve(data, projectRoot);
            if (!TabularDataFile.Exists(dataPath))
                return Fail($"Data not found: {data}");
            dataPath = await TabularDataFile.AsCsvAsync(dataPath, records);
            var (weightColumn, weightWarning) = ModelComparison.ResolveWeight(weight, experiment, dataPath);
            if (weightWarning is not null)
                WarningConsole.Warn(Markup.Escape(weightWarning));

            // The model's own probabilities: a calibration already beside it is what is being replaced.
            var engine = new EvaluationEngine();
            ProbabilityCalibration? calibration;
            if (binary)
            {
                var rows = await engine.ThresholdRowsAsync(modelPath, dataPath, experiment.Config.LabelColumn,
                    experiment.Config.InputSchema, weightColumn, CancellationToken.None, ScoringLayers.None);
                if (rows is null)
                    return Fail($"{id} gives no probability (its trainer is not calibrated), so there is no probability to calibrate.");
                calibration = ProbabilityCalibration.Fit(rows, method, Path.GetFileName(data), weightColumn, DateTimeOffset.UtcNow);
                if (calibration is null)
                    return Fail($"{Path.GetFileName(data)} holds one class only: a calibration needs both.");
            }
            else
            {
                var rows = await engine.MulticlassCalibrationRowsAsync(modelPath, dataPath, experiment.Config.LabelColumn,
                    experiment.Config.InputSchema, weightColumn, CancellationToken.None, ScoringLayers.None);
                if (rows is null)
                    return Fail($"{id} gives no class probabilities to calibrate.");
                calibration = ProbabilityCalibration.FitTemperature(rows, Path.GetFileName(data), weightColumn, DateTimeOffset.UtcNow);
                if (calibration is null)
                    return Fail($"{Path.GetFileName(data)} has no labeled rows to calibrate on.");
            }

            foreach (var path in targets)
                calibration.WriteBeside(path);
            // Platt is a likelihood fit, not an ECE fit: on few rows, or rows the model already fits well,
            // it can leave the probabilities less calibrated than they were. Said plainly, not hidden.
            if (calibration is { EceBefore: { } eceBefore, EceAfter: { } eceAfter } && eceAfter > eceBefore)
                WarningConsole.Warn(Markup.Escape(
                    $"The calibration is worse than the model's own probabilities on these rows (ECE {Format(eceBefore)} → {Format(eceAfter)}). " +
                    $"With {calibration.Rows.ToString("N0", CultureInfo.InvariantCulture)} rows the model's probabilities may already be as good as it gets — " +
                    $"remove it (mloop calibrate {id} --clear), or fit on more rows the model was not trained on."));
            return Emit(name, id, calibration, "fit", productionModel is not null, jsonOutput);
        }
        catch (Exception ex)
        {
            if (jsonOutput)
                return Fail(ex.Message);
            ErrorSuggestions.DisplayError(ex, "calibrate");
            return 1;
        }
    }

    private static int Emit(string model, string experimentId, ProbabilityCalibration? calibration, string status, bool inProduction, bool jsonOutput)
    {
        if (jsonOutput)
        {
            Console.WriteLine(JsonSerializer.Serialize(new { model, experimentId, status, inProduction, calibration, warnings = JsonOutputScope.Warnings },
                new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));
            return 0;
        }
        var where = inProduction ? " (in production)" : "";
        if (calibration is null)
        {
            if (status == "cleared")
                AnsiConsole.MarkupLineInterpolated($"[green]>[/] {experimentId}{where} answers with its own probabilities again.");
            else
                AnsiConsole.MarkupLineInterpolated($"[grey]{experimentId}{where} has no calibration of its own: its probabilities are the trainer's.[/]");
            return 0;
        }
        AnsiConsole.MarkupLine(string.Create(CultureInfo.InvariantCulture,
            $"[green]>[/] {Markup.Escape(experimentId)}{Markup.Escape(where)} is calibrated by [cyan]{calibration.Method}[/]" +
            $"{(calibration.Temperature is { } t ? $" (T = {t:0.###})" : "")} on {calibration.Rows:N0} rows of " +
            $"{Markup.Escape(calibration.FitOn)}{(calibration.WeightColumn is { } w ? $", weighted by {Markup.Escape(w)}" : "")} — " +
            $"ECE {Format(calibration.EceBefore)} → [cyan]{Format(calibration.EceAfter)}[/] on those rows."));
        if (status == "fit")
        {
            AnsiConsole.MarkupLine("[grey]Measured on the rows it was fit on, so the after is optimistic — check it with mloop evaluate on other rows.[/]");
            if (!inProduction)
                AnsiConsole.MarkupLineInterpolated($"[grey]It goes to production with the model: mloop promote {experimentId}[/]");
        }
        return 0;
    }

    private static string Format(double? ece) => ece is { } v ? v.ToString("0.000", CultureInfo.InvariantCulture) : "—";

    private static int Fail(string message)
    {
        ErrorConsole.Error(Markup.Escape(message));
        return 1;
    }
}
