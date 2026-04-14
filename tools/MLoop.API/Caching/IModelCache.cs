using Microsoft.ML;

namespace MLoop.API.Caching;

/// <summary>
/// In-memory cache for loaded ML.NET <see cref="ITransformer"/> instances. Enables warm-path prediction
/// in <c>mloop serve</c> by avoiding repeated model.zip deserialization for frequently used models.
/// </summary>
public interface IModelCache
{
    /// <summary>
    /// Returns the cached <see cref="ITransformer"/> for <paramref name="modelPath"/>, loading it
    /// on first access. If the file's last-write time has advanced since the cached entry was loaded,
    /// the entry is refreshed (transparent <c>/promote</c> handling).
    /// </summary>
    ITransformer GetOrLoad(string modelPath);

    /// <summary>
    /// Removes the entry for <paramref name="modelPath"/> if present.
    /// </summary>
    bool Evict(string modelPath);

    /// <summary>
    /// Drops every cached entry.
    /// </summary>
    void Clear();

    /// <summary>
    /// Point-in-time snapshot of cache statistics for the <c>/cache/stats</c> endpoint.
    /// </summary>
    ModelCacheStats GetStats();
}

public sealed class ModelCacheStats
{
    public int CachedCount { get; init; }
    public int MaxCachedModels { get; init; }
    public long TotalHits { get; init; }
    public long TotalMisses { get; init; }
    public long TotalReloads { get; init; }
    public long TotalEvictions { get; init; }
    public IReadOnlyList<CachedModelInfo> Models { get; init; } = Array.Empty<CachedModelInfo>();
}

public sealed class CachedModelInfo
{
    public required string ModelPath { get; init; }
    public DateTime LoadedAtUtc { get; init; }
    public DateTime LastAccessAtUtc { get; init; }
    public DateTime FileLastWriteTimeUtc { get; init; }
    public long HitCount { get; init; }
}
