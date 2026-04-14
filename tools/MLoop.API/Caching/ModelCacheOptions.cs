namespace MLoop.API.Caching;

/// <summary>
/// Binds to the <c>Serve:Cache</c> section in appsettings.json.
/// </summary>
public sealed class ModelCacheOptions
{
    public const string SectionName = "Serve:Cache";

    /// <summary>
    /// Maximum number of distinct model files held in memory. When exceeded, the least-recently
    /// accessed entry is evicted. Set to 0 to disable caching entirely.
    /// </summary>
    public int MaxCachedModels { get; set; } = 8;

    /// <summary>
    /// Model names (relative to the project's <c>models/{name}/production/model.zip</c> layout)
    /// to preload on server startup. Paths that do not yet exist are skipped with a warning.
    /// </summary>
    public List<string> PreloadModels { get; set; } = new();
}
