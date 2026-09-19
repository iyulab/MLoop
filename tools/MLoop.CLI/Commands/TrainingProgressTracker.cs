using System.Globalization;
using Spectre.Console;
using MLoop.Core.Models;

namespace MLoop.CLI.Commands;

/// <summary>
/// Holds the state the training progress bar needs between events: which time budget the
/// percentage is currently measured against, and how a trial should be named on one line.
/// </summary>
/// <remarks>
/// Extracted from <c>TrainCommand</c>'s progress callback so it can be tested. It was the
/// untested half of a defect that survived for a long time — the three mainstream tabular tasks
/// never reported a trial at all, so the bar sat at 0% and nothing went red.
/// </remarks>
public sealed class TrainingProgressTracker
{
    private double _budgetSeconds;

    /// <param name="configuredTimeLimitSeconds">
    /// The <c>--time</c> budget. Under auto-time this is only the starting value: the probe and
    /// main phases announce their own budgets and <see cref="EnterPhase"/> switches to them.
    /// </param>
    public TrainingProgressTracker(int configuredTimeLimitSeconds)
    {
        _budgetSeconds = configuredTimeLimitSeconds;
    }

    /// <summary>The budget the percentage is currently measured against.</summary>
    public double BudgetSeconds => _budgetSeconds;

    /// <summary>
    /// Absorbs an auto-time phase marker. Trials report elapsed-since-their-own-experiment-start,
    /// so each phase has to be measured against its own budget — using the configured limit for
    /// both would stall the bar through the probe and then send it backwards into main training.
    /// </summary>
    public void EnterPhase(TrainingProgress phaseEvent)
    {
        ArgumentNullException.ThrowIfNull(phaseEvent);

        switch (phaseEvent.Phase)
        {
            case TrainingPhase.ProbeStart:
                _budgetSeconds = phaseEvent.ProbeTimeSeconds;
                break;
            case TrainingPhase.ProbeComplete:
            case TrainingPhase.MainStart:
                _budgetSeconds = phaseEvent.FinalTimeSeconds;
                break;
            // ProbeConverged, ProbeFellBack and Complete end training; the budget they would
            // switch to is never used.
        }
    }

    /// <summary>
    /// Percentage to show for a trial update, or <c>null</c> when no budget is known and any
    /// number would be invented. Capped just below 100 so the bar completes only when training does.
    /// </summary>
    public double? PercentFor(TrainingProgress trial)
    {
        ArgumentNullException.ThrowIfNull(trial);

        if (_budgetSeconds <= 0)
            return null;

        return Math.Clamp(trial.ElapsedSeconds / _budgetSeconds * 100, 0, 99);
    }

    /// <summary>
    /// AutoML names a trial by its whole pipeline
    /// ("ReplaceMissingValues=&gt;Concatenate=&gt;FastTreeBinary"); only the trainer at the end
    /// differs between trials, and the prefix would crowd the metric off the line.
    /// </summary>
    public static string ShortTrainerName(string? trainerName)
    {
        if (string.IsNullOrWhiteSpace(trainerName))
            return "(unknown)";

        var parts = trainerName.Split("=>", StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return parts.Length > 0 ? parts[^1] : trainerName;
    }

    // The progress bar's descriptions are Spectre markup, rendered on the progress refresh thread.
    // Every value that comes from data or a trainer is escaped here: a fallback trainer is named
    // "SdcaLogisticRegression [manual fallback: …]", and read as markup that bracket is an unknown
    // style — an exception on a background thread, which ends the process. Whether a refresh caught
    // that description before training finished was a race, so it crashed only on slow machines.

    /// <summary>The progress description when training starts.</summary>
    public static string StartDescription(string modelName) =>
        $"[green]Training {Markup.Escape(modelName)}...[/]";

    /// <summary>The progress description for a phase boundary, or <c>null</c> to keep the current one.</summary>
    public static string? PhaseDescription(TrainingProgress p, string modelName) => p.Phase switch
    {
        TrainingPhase.ProbeStart => $"[cyan]Phase 1:[/] Probe ({p.ProbeTimeSeconds}s)...",
        TrainingPhase.ProbeComplete => $"[cyan]Phase 2:[/] Main training ({p.FinalTimeSeconds}s)...",
        TrainingPhase.ProbeConverged => "[green]Converged[/] in probe phase",
        TrainingPhase.ProbeFellBack => "[yellow]AutoML unavailable[/] for this data",
        TrainingPhase.Complete => $"[green]Finalizing {Markup.Escape(modelName)}...[/]",
        _ => null
    };

    /// <summary>The progress description for a completed trial.</summary>
    public static string TrialDescription(TrainingProgress p) =>
        $"[green]Trial {p.TrialNumber}:[/] {Markup.Escape(ShortTrainerName(p.TrainerName))} - "
        + $"{Markup.Escape(p.MetricName ?? "")}={p.Metric.ToString("F4", CultureInfo.InvariantCulture)}";
}
