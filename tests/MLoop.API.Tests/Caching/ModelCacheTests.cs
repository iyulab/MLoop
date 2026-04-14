using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.API.Caching;

namespace MLoop.API.Tests.Caching;

public sealed class ModelCacheTests : IDisposable
{
    private readonly MLContext _ml = new(seed: 42);
    private readonly List<string> _tempFiles = new();

    public void Dispose()
    {
        foreach (var f in _tempFiles)
        {
            try { File.Delete(f); } catch { }
        }
    }

    private string SaveTinyModel()
    {
        var data = _ml.Data.LoadFromEnumerable(new[]
        {
            new Sample { X = 1f, Y = 2f }, new Sample { X = 2f, Y = 4f },
            new Sample { X = 3f, Y = 6f }, new Sample { X = 4f, Y = 8f },
            new Sample { X = 5f, Y = 10f },
        });
        var pipeline = _ml.Transforms.Concatenate("Features", "X")
            .Append(_ml.Regression.Trainers.Sdca(labelColumnName: "Y"));
        var model = pipeline.Fit(data);

        var path = Path.Combine(Path.GetTempPath(), $"mloop-cache-test-{Guid.NewGuid():N}.zip");
        _tempFiles.Add(path);
        _ml.Model.Save(model, data.Schema, path);
        return path;
    }

    private static ModelCache NewCache(int maxModels = 8)
        => new(Options.Create(new ModelCacheOptions { MaxCachedModels = maxModels }),
               NullLogger<ModelCache>.Instance);

    [Fact]
    public void GetOrLoad_FirstCall_IsMiss_SubsequentCall_IsHit()
    {
        var path = SaveTinyModel();
        var cache = NewCache();

        var m1 = cache.GetOrLoad(path);
        var m2 = cache.GetOrLoad(path);

        Assert.Same(m1, m2);

        var stats = cache.GetStats();
        Assert.Equal(1, stats.CachedCount);
        Assert.Equal(1, stats.TotalMisses);
        Assert.Equal(2, stats.TotalHits); // both GetOrLoad calls record a hit on successful return
    }

    [Fact]
    public void GetOrLoad_MtimeChange_ReloadsTransformer()
    {
        var path = SaveTinyModel();
        var cache = NewCache();

        var first = cache.GetOrLoad(path);

        // Resave (same content is fine — what matters is mtime advancing)
        Thread.Sleep(1100); // ensure mtime tick difference (filesystem resolution varies)
        var data = _ml.Data.LoadFromEnumerable(new[]
        {
            new Sample { X = 10f, Y = 20f }, new Sample { X = 20f, Y = 40f },
            new Sample { X = 30f, Y = 60f }, new Sample { X = 40f, Y = 80f },
            new Sample { X = 50f, Y = 100f },
        });
        var pipeline = _ml.Transforms.Concatenate("Features", "X")
            .Append(_ml.Regression.Trainers.Sdca(labelColumnName: "Y"));
        var newModel = pipeline.Fit(data);
        _ml.Model.Save(newModel, data.Schema, path);

        var second = cache.GetOrLoad(path);

        Assert.NotSame(first, second);
        Assert.True(cache.GetStats().TotalReloads >= 1);
    }

    [Fact]
    public void Evict_RemovesEntry()
    {
        var path = SaveTinyModel();
        var cache = NewCache();

        cache.GetOrLoad(path);
        Assert.Equal(1, cache.GetStats().CachedCount);

        Assert.True(cache.Evict(path));
        Assert.Equal(0, cache.GetStats().CachedCount);
        Assert.False(cache.Evict(path));
    }

    [Fact]
    public void Clear_RemovesAllEntries()
    {
        var p1 = SaveTinyModel();
        var p2 = SaveTinyModel();
        var cache = NewCache();

        cache.GetOrLoad(p1);
        cache.GetOrLoad(p2);
        Assert.Equal(2, cache.GetStats().CachedCount);

        cache.Clear();
        Assert.Equal(0, cache.GetStats().CachedCount);
    }

    [Fact]
    public void LruEviction_WhenOverMax_RemovesLeastRecentlyUsed()
    {
        var p1 = SaveTinyModel();
        var p2 = SaveTinyModel();
        var p3 = SaveTinyModel();
        var cache = NewCache(maxModels: 2);

        cache.GetOrLoad(p1);
        Thread.Sleep(10);
        cache.GetOrLoad(p2);
        Thread.Sleep(10);
        cache.GetOrLoad(p1); // p1 becomes MRU

        Thread.Sleep(10);
        cache.GetOrLoad(p3); // triggers eviction of p2 (LRU)

        var stats = cache.GetStats();
        Assert.Equal(2, stats.CachedCount);
        Assert.Contains(stats.Models, m =>
            string.Equals(m.ModelPath, Path.GetFullPath(p1), StringComparison.OrdinalIgnoreCase));
        Assert.Contains(stats.Models, m =>
            string.Equals(m.ModelPath, Path.GetFullPath(p3), StringComparison.OrdinalIgnoreCase));
        Assert.DoesNotContain(stats.Models, m =>
            string.Equals(m.ModelPath, Path.GetFullPath(p2), StringComparison.OrdinalIgnoreCase));
        Assert.True(stats.TotalEvictions >= 1);
    }

    [Fact]
    public void Disabled_WhenMaxZero_AlwaysLoadsFromDisk()
    {
        var path = SaveTinyModel();
        var cache = NewCache(maxModels: 0);

        var m1 = cache.GetOrLoad(path);
        var m2 = cache.GetOrLoad(path);

        Assert.NotSame(m1, m2);
        Assert.Equal(0, cache.GetStats().CachedCount);
    }

    [Fact]
    public void GetOrLoad_MissingFile_ThrowsAndCleansEntry()
    {
        var cache = NewCache();
        var missing = Path.Combine(Path.GetTempPath(), $"nope-{Guid.NewGuid():N}.zip");

        Assert.Throws<FileNotFoundException>(() => cache.GetOrLoad(missing));

        // Stats should show no cached entry; faulted Lazy was cleaned.
        Assert.Equal(0, cache.GetStats().CachedCount);

        // After the file is created, a second call should succeed without being stuck on the faulted Lazy.
        var real = SaveTinyModel();
        File.Copy(real, missing);
        _tempFiles.Add(missing);

        var model = cache.GetOrLoad(missing);
        Assert.NotNull(model);
    }

    [Fact]
    public void ConcurrentGetOrLoad_SameKey_LoadsOnce()
    {
        var path = SaveTinyModel();
        var cache = NewCache();
        var results = new ITransformer[32];

        Parallel.For(0, 32, i =>
        {
            results[i] = cache.GetOrLoad(path);
        });

        var first = results[0];
        Assert.All(results, r => Assert.Same(first, r));

        var stats = cache.GetStats();
        Assert.Equal(1, stats.TotalMisses);
    }

    private sealed class Sample
    {
        public float X { get; set; }
        public float Y { get; set; }
    }
}
