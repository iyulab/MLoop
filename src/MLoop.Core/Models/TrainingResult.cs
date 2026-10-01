namespace MLoop.Core.Models;

/// <summary>
/// Result of a training operation (used by CLI layer).
/// For lower-level AutoML results, see <see cref="MLoop.Core.AutoML.AutoMLResult"/>.
/// </summary>
public class TrainingResult
{
    public required string ExperimentId { get; init; }

    /// <summary>What was trained, in parts — name, hyperparameters, fallback reason.</summary>
    public required TrainerDescriptor Trainer { get; init; }

    /// <summary>The human-facing rendering of <see cref="Trainer"/>. Derived, not stored.</summary>
    public string BestTrainer => Trainer.Display;
    public required Dictionary<string, double> Metrics { get; init; }
    public required double TrainingTimeSeconds { get; init; }
    public required string ModelPath { get; init; }

    /// <summary>
    /// The experiment's Markdown report, when one was written beside the model. Null when the
    /// report could not be rendered — the run is still a success, and the caller says nothing
    /// about a file that is not there.
    /// </summary>
    public string? ReportPath { get; init; }

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
    /// <summary>
    /// AutoML could not run on this data during the probe and a direct pipeline stood in for it, so
    /// main training is skipped (auto-time only): the cause is a property of the data, and a second
    /// pass under a larger budget would repeat the same failure and the same warnings.
    /// Data: ProbeTimeSeconds, Metric (best), TrialNumber (trials).
    /// </summary>
    ProbeFellBack,
    /// <summary>The training window is starting under a fixed budget (no probe). Data: FinalTimeSeconds.</summary>
    MainStart,
    /// <summary>
    /// A deep-learning trainer computed the features of one more input item before its first epoch
    /// (image classification runs every image through a pretrained network once; on a CPU that pass
    /// is most of the fit). Data: Step (items done), Steps (items in all), ElapsedSeconds.
    /// </summary>
    Featurize,
    /// <summary>
    /// A deep-learning trainer finished an epoch. Its fit runs at most a fixed number of epochs, so
    /// this — not the time budget — is how far along it is. Data: Epoch (finished), MaxEpochs,
    /// StopsEarly (the fit may end before MaxEpochs), ElapsedSeconds.
    /// </summary>
    Epoch,
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

    // Deep-learning epoch reporting (Phase = Epoch)
    public int Epoch { get; init; }
    public int MaxEpochs { get; init; }
    public bool StopsEarly { get; init; }

    // Deep-learning featurization reporting (Phase = Featurize)
    public int Step { get; init; }
    public int Steps { get; init; }
}
