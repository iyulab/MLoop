namespace MLoop.Core.Storage;

/// <summary>
/// Canonical on-disk layout constants for MLoop experiments — the single authority both the
/// writer (<c>MLoop.CLI.ExperimentStore</c>) and the readers (<c>MLoop.Ops</c> services, the
/// REST API) converge on, so the experiment-staging layout cannot drift across the assembly
/// boundary. Layout: <c>models/{model}/staging/{expId}/{model.zip, metadata.json, metrics.json,
/// config.json, trials.ndjson, leaderboard.json, report.md}</c> with the per-model experiment
/// counter at <c>staging</c>'s sibling index file.
/// <para>
/// Lives in <c>MLoop.Core</c> because that is the common ancestor of <c>MLoop.CLI</c> (which
/// references Ops) and <c>MLoop.Ops</c> (which references Core) — the only home that lets both
/// share one declaration without an inverted CLI←Ops dependency. Before this, CLI's ExperimentStore
/// and Ops's <c>OpsStorage</c> each hardcoded the layout independently, and three Ops services had
/// drifted to a non-existent <c>experiments</c> directory — so the time-based trigger and model
/// comparison read paths that never exist on a real project.
/// </para>
/// </summary>
public static class ExperimentLayout
{
    /// <summary>Top-level per-project directory holding every model namespace.</summary>
    public const string ModelsDirectory = "models";

    /// <summary>Experiments live here under each model — the MLOps convention, not "experiments".</summary>
    public const string StagingDirectory = "staging";

    /// <summary>The promoted model's slot under each model, sibling of <c>staging/</c>.</summary>
    public const string ProductionDirectory = "production";

    /// <summary>The serialized ML.NET model artifact — under each <c>staging/{expId}/</c> and the
    /// promoted <c>production/</c> slot. The single most-duplicated layout literal before this.</summary>
    public const string ModelFileName = "model.zip";

    /// <summary>
    /// The training inputs' distribution, column by column (<c>MLoop.Core.Monitoring.DriftProfile</c>) —
    /// the reference drift is measured against. Written at training beside <see cref="ModelFileName"/>
    /// and copied on promote when present; absent for directory-based tasks and models trained
    /// before it existed.
    /// </summary>
    public const string DriftProfileFileName = "drift-profile.json";

    /// <summary>
    /// A binary classifier's decision threshold, when one was chosen for it
    /// (<c>MLoop.Core.Models.DecisionThreshold</c>) — the probability from which it answers the positive
    /// class. Beside <see cref="ModelFileName"/>, read by every path that loads the model, and copied on
    /// promote (a promoted model without one drops the previous model's). Absent: 0.5, the trainer's own.
    /// </summary>
    public const string DecisionThresholdFileName = "decision-threshold.json";

    /// <summary>
    /// A binary classifier's probability calibration, when one was fit for it on labeled rows
    /// (<c>MLoop.Core.Models.ProbabilityCalibration</c>) — the map from the model's own probability to
    /// a calibrated one. Beside <see cref="ModelFileName"/>, applied by every path that loads the model
    /// (before <see cref="DecisionThresholdFileName"/>), and copied on promote like the threshold.
    /// </summary>
    public const string CalibrationFileName = "calibration.json";

    /// <summary>
    /// The text columns a model reads through a text encoder, each with the encoder's identity
    /// (<c>MLoop.Core.Models.TextEncodings</c>) — the model was trained on their sentence vectors. Beside
    /// <see cref="ModelFileName"/>, put in front of the model by every path that loads it, and copied on promote
    /// like the calibration. Absent: the model reads no encoded column.
    /// </summary>
    public const string TextEncodersFileName = "text-encoders.json";

    /// <summary>
    /// An image-anomaly model's record (<c>MLoop.Core.ImageAnomaly.ImageAnomalyModel</c>): the encoder whose vectors the bank
    /// holds, the patch grid, how the bank was chosen and how far held-out normal images scored. Beside
    /// <see cref="ModelFileName"/> (which is only that experiment's header), copied on promote with
    /// <see cref="PatchBankFileName"/>. Absent: the model is not an image-anomaly one.
    /// </summary>
    public const string ImageAnomalyFileName = "image-anomaly.json";

    /// <summary>
    /// The normal images' patch vectors an image-anomaly model remembers (<c>MLoop.Core.ImageAnomaly.PatchBank</c>) — the model
    /// itself. Beside <see cref="ModelFileName"/>, copied on promote with <see cref="ImageAnomalyFileName"/>.
    /// </summary>
    public const string PatchBankFileName = "patch-bank.bin";

    /// <summary>② regression wave (heteroscedastic): the optional auxiliary σ(x) model that gives the
    /// conformal band its per-row width. Sits beside <see cref="ModelFileName"/> in <c>staging/{expId}/</c>
    /// and the promoted <c>production/</c> slot, and is copied on promote when present. Absent for
    /// non-regression tasks and homoscedastic fallbacks — predict then uses the constant-width band.</summary>
    public const string ResidualModelFileName = "residual-model.zip";

    /// <summary>Per-experiment metadata (task, trainer, timestamps) inside <c>staging/{expId}/</c>.</summary>
    public const string MetadataFileName = "metadata.json";

    /// <summary>Per-experiment evaluation metrics inside <c>staging/{expId}/</c>.</summary>
    public const string MetricsFileName = "metrics.json";

    /// <summary>Per-experiment training config inside <c>staging/{expId}/</c>.</summary>
    public const string ConfigFileName = "config.json";

    /// <summary>Per-model atomic experiment-id counter, sibling of <c>staging/</c>.</summary>
    public const string IndexFileName = "experiment-index.json";

    /// <summary>
    /// Every completed trial of the search, one JSON object per line, in completion order — inside
    /// <c>staging/{expId}/</c>. NDJSON rather than a JSON array so a long search can be streamed and
    /// tailed; absent for the task paths that fit a single pipeline instead of searching.
    /// </summary>
    public const string TrialsFileName = "trials.ndjson";

    /// <summary>
    /// The same trials ranked by the metric the search optimized, best first, inside
    /// <c>staging/{expId}/</c>. Written beside <see cref="TrialsFileName"/> because ranking needs the
    /// metric's direction (<c>MLoop.Core.Evaluation.MetricDirection</c>) — knowledge a reader of the
    /// raw trials file should not have to re-derive.
    /// </summary>
    public const string LeaderboardFileName = "leaderboard.json";

    /// <summary>
    /// The experiment as a person reads it — one Markdown page inside <c>staging/{expId}/</c>
    /// rendering what <see cref="MetadataFileName"/>, <see cref="MetricsFileName"/>,
    /// <see cref="LeaderboardFileName"/> and the recorded input schema already say. Written for
    /// every experiment, failed ones included; never read back by any command, so a stale or
    /// missing report breaks nothing.
    /// </summary>
    public const string ReportFileName = "report.md";

    /// <summary>
    /// The model as an ONNX graph, written by <c>mloop export onnx</c> inside <c>staging/{expId}/</c>
    /// (or a directory the user names). Only the export writes it; no command reads it back.
    /// </summary>
    public const string OnnxModelFileName = "model.onnx";

    /// <summary>
    /// The manifest beside <see cref="OnnxModelFileName"/>: the graph's inputs and the data columns
    /// behind them, its outputs, the classes, the source model's hash and the recorded metrics
    /// (<c>MLoop.Core.Export.OnnxManifest</c>).
    /// </summary>
    public const string OnnxManifestFileName = "model.onnx.json";

    // The per-model production pointer + snapshot lives in production/metadata.json (the
    // ProductionMetadata authority), co-located with model.zip. The parallel registry.json that used to
    // duplicate it was removed (ProductionMetadata remarks) — there is no separate production registry file.
}
