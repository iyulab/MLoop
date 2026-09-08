using System.Text.Json;
using MLoop.Core.Models;

namespace MLoop.CLI.Commands;

/// <summary>
/// The machine-readable counterpart of <see cref="TrainPresenter"/>: <c>mloop train --json</c> writes
/// one JSON object per line to stdout instead of rendering a progress bar and tables.
/// </summary>
/// <remarks>
/// <para>
/// Consumers run <c>mloop train</c> as a subprocess and capture its output. Before this, the only
/// thing a capture contained for the whole training run was a single <c>Training model…: 0%</c> line
/// — the rich rendering is not parseable, and the trial stream that does exist reaches a redirected
/// stdout only as often as Spectre decides to flush.
/// </para>
/// <para>
/// Two contracts hold this together, and both are the caller's job to preserve (see
/// <c>TrainCommand</c>): <b>stdout carries nothing but these lines</b>, and the 0.28.0 channel rule
/// — a non-zero exit always leaves a cause on stderr — still applies. Every line is flushed as it is
/// written so a consumer can read the stream as it happens rather than after the process exits.
/// </para>
/// </remarks>
public sealed class TrainJsonEmitter(TextWriter output)
{
    // Nulls are omitted, not written: a phase event says nothing about values the run does not
    // have (a fixed-budget run has no probe time; nothing has a metric before training starts).
    // Writing 0 in their place would be indistinguishable from a measured zero.
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
    };

    private readonly TextWriter _output = output ?? throw new ArgumentNullException(nameof(output));
    private readonly Lock _gate = new();

    /// <summary>
    /// Writes one event. Serialized under a lock because training progress arrives on AutoML's worker
    /// threads: two half-written objects sharing a line would make the stream unparseable.
    /// </summary>
    private void Write(object payload)
    {
        var line = JsonSerializer.Serialize(payload, Options);
        lock (_gate)
        {
            _output.WriteLine(line);
            _output.Flush();
        }
    }

    private static string Now() => DateTime.UtcNow.ToString("o");

    /// <summary>
    /// A training-window boundary. The vocabulary is shared across budgeting modes so a consumer
    /// never branches on how the run was scheduled: auto-time emits <c>probe</c> → <c>main</c>
    /// (or <c>converged</c>), a fixed budget emits <c>main</c> directly, and every successful run
    /// ends the stream of phases with <c>complete</c>. Fields the phase has no fact for are
    /// omitted rather than zero-filled.
    /// </summary>
    /// <remarks>
    /// <c>trials</c> counts <b>the trials this run completed and reported</b> — the same number as
    /// the <c>trial</c> events on this stream and the rows of <c>trials.ndjson</c>, with no
    /// exceptions per task or per path. It was defined as "persisted trials" once, which read as
    /// self-consistent while the manual fallback reported a trial and persisted none: a consumer
    /// drawing trial nodes under a phase drew one node beneath a "0 trials" summary. The rule that
    /// keeps the three in step is <c>TrialLedger.IsReportable</c> — a reported trial always has a
    /// record, so no boundary can be drawn where the counts could differ.
    /// </remarks>
    public void Phase(TrainingProgress progress)
    {
        if (progress.Phase is not { } phase)
            return;

        // Which facts each boundary actually carries: a probe announces only its own budget; the
        // main window announces its budget (plus probe results under auto-time); the terminal
        // markers report what the search did. Metric values belong to trial/result events — a
        // boundary only repeats the probe's best where that is the fact being announced.
        var hasProbe = phase is TrainingPhase.ProbeStart or TrainingPhase.ProbeComplete or TrainingPhase.ProbeConverged or TrainingPhase.ProbeFellBack;
        var hasBudget = phase is TrainingPhase.ProbeComplete or TrainingPhase.MainStart;
        var hasMetric = phase is TrainingPhase.ProbeComplete or TrainingPhase.ProbeConverged
                                or TrainingPhase.ProbeFellBack;
        var hasTrials = phase is not (TrainingPhase.ProbeStart or TrainingPhase.MainStart);
        var hasElapsed = phase is TrainingPhase.Complete;

        Write(new
        {
            @event = "phase",
            phase = phase switch
            {
                TrainingPhase.ProbeStart => "probe",
                TrainingPhase.ProbeComplete => "main",
                TrainingPhase.MainStart => "main",
                TrainingPhase.ProbeConverged => "converged",
                TrainingPhase.ProbeFellBack => "fellback",
                TrainingPhase.Complete => "complete",
                _ => phase.ToString().ToLowerInvariant()
            },
            probeTimeSec = hasProbe ? progress.ProbeTimeSeconds : (int?)null,
            timeLimitSec = hasBudget ? progress.FinalTimeSeconds : (int?)null,
            trials = hasTrials ? progress.TrialNumber : (int?)null,
            metric = hasMetric ? progress.Metric : (double?)null,
            elapsedMs = hasElapsed ? (long)(progress.ElapsedSeconds * 1000) : (long?)null,
            ts = Now()
        });
    }

    /// <summary>One completed trial, as the progress channel reported it.</summary>
    public void Trial(TrainingProgress progress) => Write(new
    {
        @event = "trial",
        n = progress.TrialNumber,
        trainer = progress.TrainerName,
        metric = progress.MetricName,
        value = progress.Metric,
        elapsedMs = (long)(progress.ElapsedSeconds * 1000),
        ts = Now()
    });

    /// <summary>
    /// A condition the run recovered from. Carries no error code: MLoop has no such vocabulary yet,
    /// and inventing one per call site would be a code the consumer cannot rely on.
    /// </summary>
    /// <remarks>
    /// Fed by <c>MachineOutputScope.WarningSink</c>, which <c>WarningConsole</c> reports into —
    /// warnings had no common seam the way failures have the stderr sink, so one was built and the
    /// known sources (data-quality findings, label-drop notice, quality-gate block, post-train hook
    /// abort, Core's logger warnings incl. the metric sanitizer) route through it.
    /// </remarks>
    public void Warning(string message) => Write(new
    {
        @event = "warning",
        message,
        ts = Now()
    });

    /// <summary>The finished experiment. Emitted once, last, on a successful run.</summary>
    /// <remarks>
    /// <c>bestTrainer</c> is the display string and stays exactly as it was — consumers reading it
    /// today keep working. <c>trainer</c> is the same fact in parts, for consumers that need an
    /// identifier: <c>bestTrainer</c> has never been one (it carries hyperparameters like
    /// <c>KMeans (k=3)</c> and fallback notes), so reading it as one meant parsing an unspecified,
    /// per-task format. Absent parts are omitted, not blanked.
    /// </remarks>
    public void Result(TrainingResult result, string modelName) => Write(new
    {
        @event = "result",
        model = modelName,
        experimentId = result.ExperimentId,
        bestTrainer = result.BestTrainer,
        trainer = result.Trainer,
        metrics = result.Metrics,
        trainingTimeSec = result.TrainingTimeSeconds,
        modelPath = result.ModelPath,
        ts = Now()
    });

    /// <summary>
    /// The run's terminal failure. Written to stdout so a consumer reading only the event stream
    /// learns why it ended; the caller still writes the same cause to stderr, because "exit != 0 ⇒
    /// stderr is not empty" is a separate contract that predates this one.
    /// </summary>
    public void Error(string message) => Write(new
    {
        @event = "error",
        message,
        ts = Now()
    });
}
