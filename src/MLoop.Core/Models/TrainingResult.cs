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
/// Auto-time training phase markers reported via IProgress&lt;TrainingProgress&gt;.
/// </summary>
public enum AutoTimePhase
{
    /// <summary>Probe phase is starting. Data: ProbeTimeSeconds.</summary>
    ProbeStart,
    /// <summary>Probe phase completed, main training will start. Data: ProbeTimeSeconds, Metric (best), TrialNumber (trials), FinalTimeSeconds.</summary>
    ProbeComplete,
    /// <summary>Probe converged early; main training skipped. Data: ProbeTimeSeconds, Metric (best).</summary>
    ProbeConverged
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
    public AutoTimePhase? Phase { get; init; }
    public int ProbeTimeSeconds { get; init; }
    public int FinalTimeSeconds { get; init; }
}
