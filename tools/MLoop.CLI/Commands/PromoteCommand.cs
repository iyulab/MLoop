using System.CommandLine;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Evaluation;
using MLoop.Ops.Interfaces;
using MLoop.Ops.Services;
using Spectre.Console;
using MLoop.CLI.Infrastructure.Display;

namespace MLoop.CLI.Commands;

/// <summary>
/// mloop promote - Manually promotes an experiment to production
/// </summary>
public static class PromoteCommand
{
    public static Command Create()
    {
        var experimentIdArg = new Argument<string?>("experiment-id")
        {
            Description = "Experiment ID to promote (e.g., exp-003); omit when using --latest/--best",
            Arity = ArgumentArity.ZeroOrOne
        };

        var nameOption = new Option<string?>("--name", "-n")
        {
            Description = $"Model name (default: '{ConfigDefaults.DefaultModelName}')"
        };

        var latestOption = new Option<bool>("--latest")
        {
            Description = "Promote the most recent completed experiment",
            DefaultValueFactory = _ => false
        };

        var bestOption = new Option<bool>("--best")
        {
            Description = "Promote the completed experiment with the best metric (direction-aware)",
            DefaultValueFactory = _ => false
        };

        var forceOption = new Option<bool>("--force", "-f")
        {
            Description = "Skip confirmation when replacing the production model, and promote a model the quality gate refuses",
            DefaultValueFactory = _ => false
        };

        var noBackupOption = new Option<bool>("--no-backup")
        {
            Description = "Skip backup of current production model before promotion",
            DefaultValueFactory = _ => false
        };

        var jsonOption = new Option<bool>("--json")
        {
            Description = "Output the result as JSON (machine-readable, non-interactive)"
        };

        var decideOnlyOption = new Option<bool>("--decide-only")
        {
            Description = "Report which experiment would be promoted (direction-aware best/latest selection) without moving the production pointer; pairs with --json for external-SoR consumers",
            DefaultValueFactory = _ => false
        };

        var historyOption = new Option<bool>("--history")
        {
            Description = "List every change of the production slot — what took it, what it replaced, how, and the quality gate's outcome — instead of promoting",
            DefaultValueFactory = _ => false
        };

        var command = new Command("promote", "Promote an experiment to production");
        command.Arguments.Add(experimentIdArg);
        command.Options.Add(nameOption);
        command.Options.Add(latestOption);
        command.Options.Add(bestOption);
        command.Options.Add(forceOption);
        command.Options.Add(noBackupOption);
        command.Options.Add(jsonOption);
        command.Options.Add(decideOnlyOption);
        command.Options.Add(historyOption);

        command.SetAction((parseResult) =>
        {
            var experimentId = parseResult.GetValue(experimentIdArg);
            var name = parseResult.GetValue(nameOption);
            var latest = parseResult.GetValue(latestOption);
            var best = parseResult.GetValue(bestOption);
            var force = parseResult.GetValue(forceOption);
            var noBackup = parseResult.GetValue(noBackupOption);
            var json = parseResult.GetValue(jsonOption);
            var decideOnly = parseResult.GetValue(decideOnlyOption);
            if (parseResult.GetValue(historyOption))
                return experimentId != null || latest || best || force || decideOnly
                    ? Task.FromResult(HistoryConflict(json))
                    : ShowHistoryAsync(name, json);
            return ExecuteAsync(experimentId, name, latest, best, force, noBackup, json, decideOnly);
        });

        return command;
    }

    private static async Task<int> ExecuteAsync(
        string? experimentId,
        string? modelName,
        bool latest,
        bool best,
        bool force,
        bool noBackup,
        bool jsonOutput,
        bool decideOnly)
    {
        // In --json mode stdout must be pure JSON, so narration routes to stderr for the
        // duration — and the scope guarantees stdout still carries a document on an exit
        // that skips this command's own emitter.
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;

        try
        {
            // Exactly one selector: an explicit ID, --latest, or --best.
            var selectorCount = (experimentId != null ? 1 : 0) + (latest ? 1 : 0) + (best ? 1 : 0);
            if (selectorCount != 1)
            {
                WriteError(selectorCount == 0
                    ? "Specify an experiment: an explicit ID, --latest, or --best."
                    : "Use only one of: an explicit experiment ID, --latest, --best.", jsonOutput);
                return 1;
            }

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
                WriteError("Not inside a MLoop project. Run 'mloop init' to create a new project.", jsonOutput);
                return 1;
            }

            // Resolve model name (defaults to "default")
            var resolvedModelName = string.IsNullOrWhiteSpace(modelName)
                ? ConfigDefaults.DefaultModelName
                : modelName.Trim().ToLowerInvariant();

            var experimentStore = new ExperimentStore(fileSystem, projectDiscovery);
            var modelRegistry = new ModelRegistry(fileSystem, projectDiscovery, experimentStore);

            // Resolve --latest/--best to a concrete experiment ID.
            string? autoSelectionReason = null;
            if (experimentId == null)
            {
                var candidates = (await experimentStore.ListAsync(resolvedModelName, CancellationToken.None)).ToList();

                // --best chooses among the models the gate accepts (--force widens it to all of them);
                // the gate reads each experiment's own record.
                Dictionary<string, string>? refusals = null;
                if (best && !force)
                {
                    refusals = [];
                    foreach (var candidate in candidates.Where(c => c.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase)))
                    {
                        var data = await experimentStore.LoadAsync(resolvedModelName, candidate.ExperimentId, CancellationToken.None);
                        if (ModelRegistry.QualityGateRefusal(data, data.Config?.Metric) is { } why)
                            refusals[candidate.ExperimentId] = why;
                    }
                }

                var (selected, reason, error) = SelectExperiment(candidates, resolvedModelName, best, refusals);
                if (selected == null)
                {
                    WriteError(error!, jsonOutput);
                    return 1;
                }
                experimentId = selected;
                autoSelectionReason = reason;
            }

            // Verify experiment exists for this model
            if (!experimentStore.ExperimentExists(resolvedModelName, experimentId))
            {
                WriteError($"Experiment '{experimentId}' not found for model '{resolvedModelName}'. " +
                           $"Run 'mloop list --name {resolvedModelName}' to see all experiments.", jsonOutput);
                return 1;
            }

            // Load experiment to show details
            var experiment = await experimentStore.LoadAsync(resolvedModelName, experimentId, CancellationToken.None);

            // Check if experiment is completed
            if (!experiment.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
            {
                WriteError($"Cannot promote experiment with status '{experiment.Status}'. " +
                           "Only completed experiments can be promoted.", jsonOutput);
                return 1;
            }

            // The gate training applies before it promotes: a model no better than a trivial
            // baseline does not reach production by an explicit promote either, unless forced.
            var refusal = ModelRegistry.QualityGateRefusal(experiment, experiment.Config?.Metric);
            if (refusal != null)
            {
                if (!force)
                {
                    WriteError($"'{experimentId}' does not pass the quality gate: {refusal}. "
                               + "Use --force to promote it anyway.", jsonOutput);
                    return 1;
                }

                // Under --json, narration is routed to stderr, so the warning never breaks the document.
                AnsiConsole.MarkupLineInterpolated(
                        $"[yellow]Warning:[/] promoting despite the quality gate (--force): {refusal}.");
            }

            // Passes the gate, but its predicted labels are not what it is good for.
            var gateWarning = ModelRegistry.QualityGateWarning(experiment);
            if (gateWarning != null)
            {
                AnsiConsole.MarkupLineInterpolated($"[yellow]Warning:[/] {gateWarning}.");
            }

            if (!jsonOutput)
                ShowPromotionDetails(resolvedModelName, experiment, autoSelectionReason);

            // Check if replacing existing production model
            var currentProduction = await modelRegistry.GetProductionAsync(resolvedModelName, CancellationToken.None);

            // --decide-only: report the selection verdict without moving the production pointer.
            // Lets a consumer with its own system-of-record delegate mloop's direction-aware
            // best/latest selection (SelectExperiment) while applying the promotion in its own store.
            if (decideOnly)
            {
                if (jsonOutput)
                    OutputDecisionAsJson(resolvedModelName, experimentId, autoSelectionReason, currentProduction?.ExperimentId);
                else
                    ShowDecision(resolvedModelName, experimentId, autoSelectionReason, currentProduction?.ExperimentId);
                return 0;
            }

            if (currentProduction != null && !jsonOutput)
            {
                AnsiConsole.MarkupLine($"[yellow]Warning:[/] This will replace the current production model: [cyan]{currentProduction.ExperimentId}[/]");
                AnsiConsole.WriteLine();

                // --json is a non-interactive automation surface: never prompt there.
                if (!force && !Console.IsInputRedirected)
                {
                    if (!AnsiConsole.Confirm("Continue with promotion?"))
                    {
                        AnsiConsole.MarkupLine("[grey]Promotion cancelled.[/]");
                        return 0;
                    }
                }
            }

            // Create promotion manager for backup + history
            var promotionManager = new FilePromotionManager(projectRoot);

            // Backup current production before promotion
            string? backupPath = null;
            if (!noBackup && currentProduction != null)
            {
                backupPath = await promotionManager.BackupProductionAsync(resolvedModelName);
            }

            if (jsonOutput)
            {
                await modelRegistry.PromoteAsync(resolvedModelName, experimentId, CancellationToken.None);
            }
            else
            {
                // Promote with progress indicator
                await AnsiConsole.Status()
                    .Spinner(Spinner.Known.Dots)
                    .StartAsync("[yellow]Promoting to production...[/]", async ctx =>
                    {
                        await modelRegistry.PromoteAsync(resolvedModelName, experimentId, CancellationToken.None);
                        ctx.Status("[green]Promotion complete![/]");
                    });
            }

            if (jsonOutput)
            {
                OutputAsJson(resolvedModelName, experimentId, autoSelectionReason,
                    currentProduction?.ExperimentId, backupPath, gateWarning);
            }
            else
            {
                ShowPromotionResult(resolvedModelName, experimentId, backupPath);
            }

            return 0;
        }
        catch (Exception ex)
        {
            if (jsonOutput)
            {
                WriteError(ex.Message, jsonOutput: true);
                return 1;
            }
            ErrorSuggestions.DisplayError(ex, "promote");
            return 1;
        }
    }

    /// <summary>
    /// Picks the newest (--latest) or best-metric (--best) completed experiment. Metric comparison
    /// is direction-aware via <see cref="MetricDirection"/> (rmse-style = lower wins).
    /// </summary>
    /// <param name="gateRefusals">
    /// For --best: experiments the quality gate refuses, with the reason. They are not candidates —
    /// the best experiment is the best one that may go to production, not the highest number. An
    /// accuracy of 0.993 from a model answering one class outranked every model that had learned.
    /// </param>
    internal static (string? ExperimentId, string? Reason, string? Error) SelectExperiment(
        IReadOnlyList<ExperimentSummary> candidates, string modelName, bool best,
        IReadOnlyDictionary<string, string>? gateRefusals = null)
    {
        var experiments = candidates
            .Where(e => e.Status.Equals("Completed", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (experiments.Count == 0)
            return (null, null,
                $"No completed experiments found for model '{modelName}'. Run 'mloop train --name {modelName}' first.");

        if (!best)
        {
            var newest = experiments.OrderByDescending(e => e.Timestamp).First();
            return (newest.ExperimentId, "latest", null);
        }

        var scored = experiments.Where(e => e.BestMetric.HasValue && !string.IsNullOrEmpty(e.MetricName)).ToList();
        if (scored.Count == 0)
            return (null, null,
                $"No completed experiment for model '{modelName}' has a recorded metric — cannot select --best. " +
                "Use --latest or an explicit experiment ID.");

        if (gateRefusals is { Count: > 0 })
        {
            var refused = scored.Where(e => gateRefusals.ContainsKey(e.ExperimentId)).ToList();
            scored = [.. scored.Except(refused)];
            if (scored.Count == 0)
                return (null, null,
                    $"No completed experiment for model '{modelName}' passes the quality gate — "
                    + string.Join("; ", refused.Select(e => $"{e.ExperimentId}: {gateRefusals[e.ExperimentId]}"))
                    + ". Promote one explicitly with --force to override.");
        }

        // Comparing across different metric names is meaningless; require a single metric.
        var metricNames = scored.Select(e => e.MetricName!.ToLowerInvariant()).Distinct().ToList();
        if (metricNames.Count > 1)
            return (null, null,
                $"Experiments for model '{modelName}' were optimized for different metrics ({string.Join(", ", metricNames)}) — " +
                "--best cannot compare across them. Use --latest or an explicit experiment ID.");

        // The same metric name over differently weighted rows is not one scale either: a metric
        // weighted toward rare, re-checked cases and an unweighted one describe different
        // distributions, so "best" between them is not a judgement --best can make.
        // Promotion is an act, so a mix is refused rather than narrowed the way status narrows it.
        {
            static string Describe(string? w) => w is null ? "unweighted" : $"weighted by '{w}'";
            var groups = scored
                .GroupBy(e => ExperimentRanking.Measurement.Of(e).WeightColumn)
                .ToList();
            if (groups.Count > 1)
                return (null, null,
                    $"Experiments for model '{modelName}' were measured with different row weights ("
                    + string.Join("; ", groups.Select(g =>
                        $"{string.Join(", ", g.Select(e => e.ExperimentId))}: {Describe(g.Key)}"))
                    + ") — --best cannot compare across them. Score the candidates on one evaluation set "
                    + "(mloop evaluate <experiment> <file> --weight <column>) and promote one by its ID.");
        }

        var lowerIsBetter = MetricDirection.IsLowerBetter(metricNames[0]);
        var winner = (lowerIsBetter
                ? scored.OrderBy(e => e.BestMetric!.Value)
                : scored.OrderByDescending(e => e.BestMetric!.Value))
            .ThenByDescending(e => e.Timestamp) // deterministic tie-break: newer wins
            .First();

        return (winner.ExperimentId, $"best {metricNames[0]}={winner.BestMetric!.Value:F4}", null);
    }

    private static void ShowPromotionDetails(string modelName, ExperimentData experiment, string? autoSelectionReason)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule($"[bold]Promoting to Production[/]").LeftJustified());
        AnsiConsole.WriteLine();

        if (autoSelectionReason != null)
            AnsiConsole.MarkupLine($"[grey]Auto-selected ({autoSelectionReason})[/]");

        var table = new Table()
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey);

        table.AddColumn("[bold]Property[/]");
        table.AddColumn("[bold]Value[/]");

        table.AddRow("Model", $"[cyan]{modelName}[/]");
        table.AddRow("Experiment ID", $"[cyan]{experiment.ExperimentId}[/]");
        table.AddRow("Task", $"[yellow]{experiment.Task}[/]");
        table.AddRow("Timestamp", TimestampDisplay.Local(experiment.Timestamp));

        if (experiment.Metrics != null && experiment.Metrics.Any())
        {
            var metricsStr = string.Join(", ", experiment.Metrics.Select(m => $"{m.Key}={m.Value:F4}"));
            table.AddRow("Metrics", $"[yellow]{metricsStr}[/]");
        }

        if (experiment.Result != null)
        {
            table.AddRow("Best Trainer", $"[green]{Markup.Escape(experiment.Result.BestTrainer)}[/]");
            table.AddRow("Training Time", $"{experiment.Result.TrainingTimeSeconds:F2}s");
        }

        AnsiConsole.Write(table);
        AnsiConsole.WriteLine();
    }

    private static void ShowPromotionResult(string modelName, string experimentId, string? backupPath)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[green]Promotion Complete![/]").LeftJustified());
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{modelName}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Experiment: [cyan]{experimentId}[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Status: [green bold]production[/]");
        if (backupPath != null)
        {
            ValueLine.Write("[green]>[/] Backup: ", backupPath);
        }
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine("[yellow]Next steps:[/]");
        AnsiConsole.MarkupLine($"  mloop predict --name {modelName}");
        AnsiConsole.MarkupLine($"  mloop info --name {modelName}");
        AnsiConsole.WriteLine();
    }

    private static void OutputAsJson(
        string modelName, string experimentId, string? autoSelectionReason,
        string? previousProduction, string? backupPath, string? qualityWarning)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        Console.WriteLine(JsonSerializer.Serialize(new
        {
            Model = modelName,
            ExperimentId = experimentId,
            Status = "promoted",
            SelectedBy = autoSelectionReason ?? "explicit",
            PreviousProduction = previousProduction,
            BackupPath = backupPath,
            QualityWarning = qualityWarning,
        }, options));
    }

    /// <summary>
    /// --decide-only JSON: the promotion verdict only (which experiment, how it was selected,
    /// whether it would replace the current production), with the pointer left untouched
    /// (<c>applied: false</c>) so an external-SoR consumer applies it in its own store.
    /// </summary>
    private static void OutputDecisionAsJson(
        string modelName, string experimentId, string? autoSelectionReason, string? currentProduction)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        Console.WriteLine(JsonSerializer.Serialize(new
        {
            Model = modelName,
            ExperimentId = experimentId,
            Status = "decided",
            SelectedBy = autoSelectionReason ?? "explicit",
            CurrentProduction = currentProduction,
            WouldReplace = currentProduction != null && currentProduction != experimentId,
            Applied = false,
        }, options));
    }

    private static void ShowDecision(
        string modelName, string experimentId, string? autoSelectionReason, string? currentProduction)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[grey]Decision only — production pointer not moved.[/]");
        AnsiConsole.MarkupLine($"[green]>[/] Would promote: [cyan]{experimentId}[/]" +
            (autoSelectionReason != null ? $" [grey]({autoSelectionReason})[/]" : ""));
        if (currentProduction != null && currentProduction != experimentId)
            AnsiConsole.MarkupLine($"[green]>[/] Would replace current production: [cyan]{currentProduction}[/]");
        else if (currentProduction == experimentId)
            AnsiConsole.MarkupLine($"[green]>[/] Already in production.");
        AnsiConsole.MarkupLine($"[grey]  Run without [blue]--decide-only[/] to apply.[/]");
        AnsiConsole.WriteLine();
    }

    private static int HistoryConflict(bool jsonOutput)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;
        WriteError("--history lists the promotions and changes nothing: it takes no experiment, --latest, --best, --force or --decide-only.", jsonOutput);
        return 1;
    }

    /// <summary>
    /// <c>promote --history</c>: the production slot's changes, oldest first — the history
    /// <c>rollback</c> reads, with the quality gate's outcome at each change.
    /// </summary>
    private static async Task<int> ShowHistoryAsync(string? modelName, bool jsonOutput)
    {
        using var machineOutput = jsonOutput ? new JsonOutputScope() : null;
        string projectRoot;
        try
        {
            projectRoot = new ProjectDiscovery(new FileSystemManager()).FindRoot();
        }
        catch (InvalidOperationException)
        {
            WriteError("Not inside a MLoop project. Run 'mloop init' to create a new project.", jsonOutput);
            return 1;
        }

        var name = CommandContext.ResolveModelName(modelName);
        var history = await new FilePromotionManager(projectRoot).GetHistoryAsync(name);
        if (jsonOutput)
        {
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                modelName = name,
                promotions = history.Select(r => new
                {
                    experimentId = r.ExperimentId,
                    previousExperimentId = r.PreviousExperimentId,
                    action = r.Action,
                    timestamp = r.Timestamp.ToUniversalTime(),
                    gate = r.Gate,
                    gateDetail = r.GateDetail,
                    metric = r.Metric,
                    metricValue = r.MetricValue,
                }),
            }, new JsonSerializerOptions { WriteIndented = true }));
            return 0;
        }

        if (history.Count == 0)
        {
            AnsiConsole.MarkupLineInterpolated($"[grey]No promotion recorded for '{name}'.[/]");
            return 0;
        }
        // What each change replaced is the row above it, so the table leaves it to --json
        // (previousExperimentId) — with it the rows no longer fit 80 columns.
        // One metric for the whole history is the usual case: it goes in the heading, and the cells keep
        // only values — with the name in every cell the table outgrew 80 columns and rows split in two.
        var metrics = history.Select(r => r.Metric).Where(m => m is not null).Distinct().ToList();
        var oneMetric = metrics.Count == 1 ? metrics[0] : null;
        var table = new Table().Border(TableBorder.Rounded)
            .Title($"[bold]Promotions — {Markup.Escape(name)}[/]")
            .AddColumn(Markup.Escape(TimestampDisplay.ZoneHeading("When")))
            .AddColumn("Experiment")
            .AddColumn("How")
            .AddColumn("Gate")
            .AddColumn(new TableColumn(Markup.Escape(oneMetric ?? "Metric")).RightAligned());
        foreach (var r in history)
            table.AddRow(
                Markup.Escape(TimestampDisplay.LocalWithoutZone(r.Timestamp)),
                Markup.Escape(r.ExperimentId),
                Markup.Escape(r.Action),
                r.Gate switch
                {
                    PromotionGate.Forced => "[red]forced[/]",
                    PromotionGate.Passed when r.GateDetail is not null => "[yellow]passed*[/]",
                    PromotionGate.Passed => "[green]passed[/]",
                    PromotionGate.NotApplied => "[grey]not applied[/]",
                    _ => "[grey]-[/]",
                },
                Markup.Escape((oneMetric is null && r.Metric is not null ? r.Metric + " " : "")
                    + (r.MetricValue?.ToString("0.0000", System.Globalization.CultureInfo.InvariantCulture) ?? "-")));
        AnsiConsole.Write(table);
        foreach (var r in history.Where(r => r.GateDetail is not null))
            AnsiConsole.MarkupLineInterpolated($"[grey]{r.ExperimentId}: {r.GateDetail}[/]");
        return 0;
    }

    private static void WriteError(string message, bool jsonOutput)
    {
        if (jsonOutput)
        {
            // The JSON envelope stays on stdout — that is what a --json consumer parses — but the
            // cause is mirrored to stderr so "exit != 0 ⇒ stderr has a cause" holds in every mode,
            // including for consumers that only check the exit code and read stderr.
            JsonError.Emit(message);
            Console.Error.WriteLine(message);
        }
        else
        {
            ErrorConsole.Out.MarkupLineInterpolated($"[red]Error:[/] {message}");
        }
    }
}
