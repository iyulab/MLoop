namespace MLoop.Ops;

/// <summary>
/// Canonical on-disk layout constants for MLoop experiments, shared across MLoop.Ops services so
/// they cannot drift from the layout MLoop.CLI's <c>ExperimentStore</c> actually writes:
/// <c>models/{model}/staging/{expId}/{metadata.json, metrics.json, ...}</c>. Three Ops services had
/// each independently hardcoded a non-existent <c>experiments</c> directory, so the time-based
/// trigger and model comparison read paths that never exist on a real project (F-32 / F-33).
/// </summary>
internal static class OpsStorage
{
    public const string ModelsDirectory = "models";

    /// <summary>Experiments live here — the MLOps convention, not "experiments".</summary>
    public const string StagingDirectory = "staging";

    public const string MetadataFileName = "metadata.json";
    public const string MetricsFileName = "metrics.json";
}
