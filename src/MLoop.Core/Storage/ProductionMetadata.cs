using System.Text.Json;

namespace MLoop.Core.Storage;

/// <summary>
/// Typed view of <c>models/{name}/production/metadata.json</c> — the single authority for "what
/// experiment is currently in production" plus its promotion-time snapshot (metrics/task/trainer/label).
/// Written by promote and read by every consumer of production state: the CLI <c>ModelRegistry</c> and
/// <c>ModelNameResolver</c>, and the Ops <c>FilePromotionManager</c>. Co-located with <c>model.zip</c> in
/// the production directory, so the production pointer and the deployed model can never disagree.
/// </summary>
/// <remarks>
/// This supersedes the parallel <c>registry.json</c> <c>production</c> entry, which duplicated exactly
/// this record (same experimentId/promotedAt/task/bestTrainer/labelColumn/metrics) and was written by the
/// same promote call. Two artifacts holding one fact is the cross-artifact drift class the Single-Source
/// Authorities doctrine exists to prevent (BUG-48 was rooted in reading the wrong one); the <c>registry.json</c>
/// write path was removed the same way the obsolete <c>model-registry.json</c> read path was (cycle-93).
/// Pre-existing on-disk <c>registry.json</c> files are harmless stale sidecars — no reader consults them.
/// </remarks>
public sealed record ProductionMetadata
{
    public string? ModelName { get; init; }
    public string? ExperimentId { get; init; }
    public DateTime? PromotedAt { get; init; }
    public Dictionary<string, double>? Metrics { get; init; }
    public string? Task { get; init; }
    public string? BestTrainer { get; init; }
    public string? LabelColumn { get; init; }

    // Mirrors the writer's serialization (FileSystemManager: camelCase names, case-insensitive) so the
    // typed read cannot drift from the on-disk casing.
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true
    };

    /// <summary>
    /// Reads the production metadata given a model's production directory
    /// (<c>models/{name}/production</c>). Returns <c>null</c> when no model has been promoted (the file is
    /// absent) or the file cannot be parsed — the same "no production model" signal the readers already
    /// handle. Uses plain file IO (no CLI filesystem abstraction) so this stays a Core-owned authority.
    /// </summary>
    public static async Task<ProductionMetadata?> ReadAsync(
        string productionDirectory,
        CancellationToken cancellationToken = default)
    {
        var path = Path.Combine(productionDirectory, ExperimentLayout.MetadataFileName);
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            await using var stream = File.OpenRead(path);
            return await JsonSerializer
                .DeserializeAsync<ProductionMetadata>(stream, Options, cancellationToken)
                .ConfigureAwait(false);
        }
        catch
        {
            return null;
        }
    }
}
