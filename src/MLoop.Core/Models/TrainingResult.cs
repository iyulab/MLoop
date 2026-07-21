namespace MLoop.Core.Models;

/// <summary>
/// Result of a training operation (used by CLI layer).
/// For lower-level AutoML results, see <see cref="MLoop.Core.AutoML.AutoMLResult"/>.
/// </summary>
public class TrainingResult
{
    public required string ExperimentId { get; init; }
    public required string BestTrainer { get; init; }
    public required Dictionary<string, double> Metrics { get; init; }
    public required double TrainingTimeSeconds { get; init; }
    public required string ModelPath { get; init; }
    public long RowCount { get; init; }
    public InputSchemaInfo? Schema { get; init; }
}

/// <summary>
/// Training phase markers reported via IProgress&lt;TrainingProgress&gt;. The probe members only
/// occur under auto-time; <see cref="MainStart"/> and <see cref="Complete"/> bound the training
/// window on every path, so a consumer never has to know which budgeting mode produced the run.
/// </summary>
public enum TrainingPhase
{
    /// <summary>Probe phase is starting (auto-time only). Data: ProbeTimeSeconds.</summary>
    ProbeStart,
    /// <summary>Probe phase completed, main training will start (auto-time only). Data: ProbeTimeSeconds, Metric (best), TrialNumber (trials), FinalTimeSeconds.</summary>
    ProbeComplete,
    /// <summary>Probe converged early; main training skipped (auto-time only). Data: ProbeTimeSeconds, Metric (best), TrialNumber (trials).</summary>
    ProbeConverged,
    /// <summary>The training window is starting under a fixed budget (no probe). Data: FinalTimeSeconds.</summary>
    MainStart,
    /// <summary>The training window ended; post-training steps (save, evaluate, promote) follow. Every successful run ends its phase stream with this. Data: TrialNumber (trials retained in the experiment — matches trials.ndjson; under auto-time, discarded probe trials are not in it), ElapsedSeconds.</summary>
    Complete
}

/// <summary>
/// Progress information during training
/// </summary>
public class TrainingProgress
{
    public required int TrialNumber { get; init; }
    public required string TrainerName { get; init; }
    public required double Metric { get; init; }
    public required string MetricName { get; init; }
    public required double ElapsedSeconds { get; init; }

    // Auto-time phase reporting (null = normal trial update)
    public TrainingPhase? Phase { get; init; }
    public int ProbeTimeSeconds { get; init; }
    public int FinalTimeSeconds { get; init; }
}
