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
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
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

    /// <summary>An auto-time budget transition: the probe starting, main training starting, or the probe converging.</summary>
    public void Phase(TrainingProgress progress)
    {
        if (progress.Phase is not { } phase)
            return;

        Write(new
        {
            @event = "phase",
            phase = phase switch
            {
                AutoTimePhase.ProbeStart => "probe",
                AutoTimePhase.ProbeComplete => "main",
                AutoTimePhase.ProbeConverged => "converged",
                _ => phase.ToString().ToLowerInvariant()
            },
            probeTimeSec = progress.ProbeTimeSeconds,
            timeLimitSec = progress.FinalTimeSeconds,
            trials = progress.TrialNumber,
            metric = progress.Metric,
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
    /// <b>Reserved — nothing calls this yet.</b> Errors were straightforward because every failure
    /// path already funnels through one stderr sink the scope can hook; warnings have no such seam
    /// (they are raised from Spectre render calls in the command and from <c>ILogger.Warning</c> in
    /// two assemblies), and in machine mode all of them are discarded with the rest of the
    /// narration. Wiring those sources is follow-up work, not something this shape does on its own.
    /// </remarks>
    public void Warning(string message) => Write(new
    {
        @event = "warning",
        message,
        ts = Now()
    });

    /// <summary>The finished experiment. Emitted once, last, on a successful run.</summary>
    public void Result(TrainingResult result, string modelName) => Write(new
    {
        @event = "result",
        model = modelName,
        experimentId = result.ExperimentId,
        bestTrainer = result.BestTrainer,
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
