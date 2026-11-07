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
/// Authorities doctrine exists to prevent (a promotion defect was rooted in reading the wrong one); the <c>registry.json</c>
/// write path was removed the same way the obsolete <c>model-registry.json</c> read path was.
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

    /// <summary>
    /// The SHA-256 of <c>model.zip</c> as the promotion left it (<see cref="Models.ModelFingerprint"/>), or <c>null</c> for a slot promoted
    /// before this was recorded. <see cref="Models.ModelIntegrity"/> compares the file with it.
    /// </summary>
    public string? ModelSha256 { get; init; }

    // Mirrors the writer's serialization (FileSystemManager: camelCase names, case-insensitive) so the
    // typed read cannot drift from the on-disk casing.
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true
    };

    /// <summary>
    /// Reads the production metadata given a model's production directory
    /// (<c>models/{name}/production</c>). Returns <c>null</c> only when no model has been promoted - the file is
    /// absent. A file that is there and cannot be read is not "no production model": it throws
    /// <see cref="UnreadableStateFileException"/> with the path and the way back, so a caller cannot report damage as
    /// absence (the old catch-all returned <c>null</c> for both, and "Train and promote a model first" followed).
    /// Uses plain file IO (no CLI filesystem abstraction) so this stays a Core-owned authority.
    /// </summary>
    /// <exception cref="UnreadableStateFileException">The file exists but is not valid metadata.</exception>
    public static async Task<ProductionMetadata?> ReadAsync(
        string productionDirectory,
        CancellationToken cancellationToken = default)
    {
        var (value, damage) = await TryReadAsync(productionDirectory, cancellationToken).ConfigureAwait(false);
        return damage is null ? value : throw damage;
    }

    /// <summary>
    /// <see cref="ReadAsync"/> for a caller that can go on without the pointer - promoting writes it anew, a listing shows
    /// the other models: the damage comes back as a value to report instead of an exception to unwind through.
    /// </summary>
    public static async Task<(ProductionMetadata? Value, UnreadableStateFileException? Damage)> TryReadAsync(
        string productionDirectory,
        CancellationToken cancellationToken = default)
    {
        var path = Path.Combine(productionDirectory, ExperimentLayout.MetadataFileName);
        if (!File.Exists(path))
        {
            return (null, null);
        }

        try
        {
            await using var stream = File.OpenRead(path);
            var value = await JsonSerializer
                .DeserializeAsync<ProductionMetadata>(stream, Options, cancellationToken)
                .ConfigureAwait(false);
            return value is null
                ? (null, Damaged(path, "the file holds no metadata", null))
                : (value, null);
        }
        catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
        {
            return (null, Damaged(path, ex is JsonException ? "not valid JSON" : ex.Message.Split('\n')[0].TrimEnd(), ex));
        }
    }

    private static UnreadableStateFileException Damaged(string path, string reason, Exception? inner) =>
        new(path, reason,
            "The model files are probably intact: restore this file from git, or run 'mloop promote <experiment-id>' to write it again.",
            inner);
}
