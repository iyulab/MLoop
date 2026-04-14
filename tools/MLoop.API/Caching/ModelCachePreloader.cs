using Microsoft.Extensions.Options;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.API.Caching;

/// <summary>
/// Preloads production models on host startup so the first <c>POST /predict</c> for each configured
/// model lands on a warm cache entry instead of paying the cold deserialization cost.
/// </summary>
internal sealed class ModelCachePreloader : IHostedService
{
    private readonly IModelCache _cache;
    private readonly IModelRegistry _registry;
    private readonly ModelCacheOptions _options;
    private readonly ILogger<ModelCachePreloader> _logger;

    public ModelCachePreloader(
        IModelCache cache,
        IModelRegistry registry,
        IOptions<ModelCacheOptions> options,
        ILogger<ModelCachePreloader> logger)
    {
        _cache = cache;
        _registry = registry;
        _options = options.Value;
        _logger = logger;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        if (_options.MaxCachedModels <= 0 || _options.PreloadModels.Count == 0)
        {
            return Task.CompletedTask;
        }

        foreach (var modelName in _options.PreloadModels)
        {
            if (cancellationToken.IsCancellationRequested) break;

            var trimmed = modelName?.Trim().ToLowerInvariant();
            if (string.IsNullOrEmpty(trimmed)) continue;

            var modelPath = Path.Combine(_registry.GetProductionPath(trimmed), "model.zip");
            if (!File.Exists(modelPath))
            {
                _logger.LogWarning("Preload: skipping '{ModelName}' — no production model at {Path}",
                    trimmed, modelPath);
                continue;
            }

            try
            {
                _cache.GetOrLoad(modelPath);
                _logger.LogInformation("Preload: loaded '{ModelName}' into cache", trimmed);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Preload: failed to load '{ModelName}' from {Path}",
                    trimmed, modelPath);
            }
        }

        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
