namespace MLoop.Core.Storage;

/// <summary>
/// Canonical on-disk layout constants for MLoop experiments — the single authority both the
/// writer (<c>MLoop.CLI.ExperimentStore</c>) and the readers (<c>MLoop.Ops</c> services, the
/// REST API) converge on, so the experiment-staging layout cannot drift across the assembly
/// boundary. Layout: <c>models/{model}/staging/{expId}/{metadata.json, metrics.json, config.json}</c>
/// with the per-model experiment counter at <c>staging</c>'s sibling index file.
/// <para>
/// Lives in <c>MLoop.Core</c> because that is the common ancestor of <c>MLoop.CLI</c> (which
/// references Ops) and <c>MLoop.Ops</c> (which references Core) — the only home that lets both
/// share one declaration without an inverted CLI←Ops dependency. Before this, CLI's ExperimentStore
/// and Ops's <c>OpsStorage</c> each hardcoded the layout independently, and three Ops services had
/// drifted to a non-existent <c>experiments</c> directory — so the time-based trigger and model
/// comparison read paths that never exist on a real project (F-32 / F-33).
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

    // The per-model production pointer + snapshot lives in production/metadata.json (the
    // ProductionMetadata authority), co-located with model.zip. The parallel registry.json that used to
    // duplicate it was removed (ProductionMetadata remarks) — there is no separate production registry file.
}
