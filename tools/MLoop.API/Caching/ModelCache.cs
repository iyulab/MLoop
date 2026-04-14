using System.Collections.Concurrent;
using Microsoft.Extensions.Options;
using Microsoft.ML;

namespace MLoop.API.Caching;

/// <summary>
/// Default <see cref="IModelCache"/>. ITransformers are cached by absolute model-path key with
/// mtime-based invalidation and LRU eviction.
/// </summary>
/// <remarks>
/// <para>
/// ML.NET's <see cref="ITransformer.Transform"/> is thread-safe, so a single cached instance may be
/// used concurrently by multiple request threads. Each caller still needs its own <see cref="MLContext"/>
/// for ancillary operations (TextLoader etc.).
/// </para>
/// <para>
/// The cache protects against the thundering-herd problem (N concurrent requests for the same uncached
/// model) with <see cref="Lazy{T}"/> entries.
/// </para>
/// </remarks>
public sealed class ModelCache : IModelCache
{
    private readonly ILogger<ModelCache> _logger;
    private readonly ModelCacheOptions _options;
    private readonly MLContext _loaderContext = new(seed: 42);
    private readonly ConcurrentDictionary<string, Lazy<CacheEntry>> _entries = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _evictionLock = new();

    private long _hits;
    private long _misses;
    private long _reloads;
    private long _evictions;

    public ModelCache(IOptions<ModelCacheOptions> options, ILogger<ModelCache> logger)
    {
        _options = options.Value;
        _logger = logger;
    }

    public ITransformer GetOrLoad(string modelPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelPath);

        if (_options.MaxCachedModels <= 0)
        {
            return LoadFromDisk(modelPath);
        }

        var key = Path.GetFullPath(modelPath);
        var currentMtime = File.GetLastWriteTimeUtc(key);

        while (true)
        {
            var observedMtime = currentMtime;
            var lazy = _entries.GetOrAdd(key, _ => NewLazyEntry(key, observedMtime, countAsMiss: true));

            CacheEntry entry;
            try
            {
                entry = lazy.Value;
            }
            catch
            {
                _entries.TryRemove(new KeyValuePair<string, Lazy<CacheEntry>>(key, lazy));
                throw;
            }

            if (entry.FileLastWriteTimeUtc < observedMtime)
            {
                var replacement = NewLazyEntry(key, observedMtime, countAsMiss: false);
                if (_entries.TryUpdate(key, replacement, lazy))
                {
                    Interlocked.Increment(ref _reloads);
                    _logger.LogInformation("Model cache: reloaded '{Path}' (mtime changed)", key);
                }
                continue;
            }

            entry.RecordHit();
            Interlocked.Increment(ref _hits);
            EnsureCapacity();
            return entry.Transformer;
        }
    }

    public bool Evict(string modelPath)
    {
        var key = Path.GetFullPath(modelPath);
        if (_entries.TryRemove(key, out _))
        {
            Interlocked.Increment(ref _evictions);
            _logger.LogInformation("Model cache: evicted '{Path}' (explicit)", key);
            return true;
        }
        return false;
    }

    public void Clear()
    {
        var removed = _entries.Count;
        _entries.Clear();
        if (removed > 0)
        {
            Interlocked.Add(ref _evictions, removed);
            _logger.LogInformation("Model cache: cleared {Count} entries", removed);
        }
    }

    public ModelCacheStats GetStats()
    {
        var snapshot = _entries.Values
            .Where(l => l.IsValueCreated)
            .Select(l => l.Value.Snapshot())
            .OrderByDescending(m => m.LastAccessAtUtc)
            .ToList();

        return new ModelCacheStats
        {
            CachedCount = snapshot.Count,
            MaxCachedModels = _options.MaxCachedModels,
            TotalHits = Interlocked.Read(ref _hits),
            TotalMisses = Interlocked.Read(ref _misses),
            TotalReloads = Interlocked.Read(ref _reloads),
            TotalEvictions = Interlocked.Read(ref _evictions),
            Models = snapshot
        };
    }

    private Lazy<CacheEntry> NewLazyEntry(string key, DateTime mtime, bool countAsMiss) =>
        new(() =>
        {
            if (countAsMiss) Interlocked.Increment(ref _misses);
            var transformer = LoadFromDisk(key);
            return new CacheEntry(key, transformer, mtime);
        }, LazyThreadSafetyMode.ExecutionAndPublication);

    private ITransformer LoadFromDisk(string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}", modelPath);

        return _loaderContext.Model.Load(modelPath, out _);
    }

    private void EnsureCapacity()
    {
        if (_entries.Count <= _options.MaxCachedModels) return;

        lock (_evictionLock)
        {
            while (_entries.Count > _options.MaxCachedModels)
            {
                var victim = _entries
                    .Where(kv => kv.Value.IsValueCreated)
                    .OrderBy(kv => kv.Value.Value.LastAccessTicks)
                    .FirstOrDefault();

                if (victim.Key is null) break;

                if (_entries.TryRemove(new KeyValuePair<string, Lazy<CacheEntry>>(victim.Key, victim.Value)))
                {
                    Interlocked.Increment(ref _evictions);
                    _logger.LogInformation("Model cache: evicted '{Path}' (LRU, cache size {Count}>{Max})",
                        victim.Key, _entries.Count, _options.MaxCachedModels);
                }
            }
        }
    }

    private sealed class CacheEntry
    {
        public string ModelPath { get; }
        public ITransformer Transformer { get; }
        public DateTime LoadedAtUtc { get; }
        public DateTime FileLastWriteTimeUtc { get; }

        private long _lastAccessTicks;
        private long _hitCount;

        public long LastAccessTicks => Interlocked.Read(ref _lastAccessTicks);

        public CacheEntry(string modelPath, ITransformer transformer, DateTime fileLastWriteTimeUtc)
        {
            ModelPath = modelPath;
            Transformer = transformer;
            LoadedAtUtc = DateTime.UtcNow;
            FileLastWriteTimeUtc = fileLastWriteTimeUtc;
            _lastAccessTicks = LoadedAtUtc.Ticks;
        }

        public void RecordHit()
        {
            Interlocked.Exchange(ref _lastAccessTicks, DateTime.UtcNow.Ticks);
            Interlocked.Increment(ref _hitCount);
        }

        public CachedModelInfo Snapshot() => new()
        {
            ModelPath = ModelPath,
            LoadedAtUtc = LoadedAtUtc,
            LastAccessAtUtc = new DateTime(Interlocked.Read(ref _lastAccessTicks), DateTimeKind.Utc),
            FileLastWriteTimeUtc = FileLastWriteTimeUtc,
            HitCount = Interlocked.Read(ref _hitCount)
        };
    }
}
