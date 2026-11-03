using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Search.Tests;

/// <summary>
/// Rows kept in memory are filled before they are read: a reader that met a copy still being filled could see a row
/// that was not written yet and fail on a vector column.
/// </summary>
public class RowCacheTests
{
    private sealed class Row
    {
        [VectorType(3)] public float[] Features { get; set; } = new float[3];
        public float Label { get; set; }
    }

    private static IEnumerable<Row> Rows(int n, Action onEnumerate)
    {
        onEnumerate();
        for (var i = 0; i < n; i++)
            yield return new Row { Features = [i, i + 0.5f, -i], Label = i % 2 };
    }

    [Fact]
    public void TheRowsAreAllThere_AndTheSourceWasReadOnce()
    {
        var reads = 0;
        var ml = new MLContext(1);
        var source = ml.Data.LoadFromEnumerable(Rows(50_000, () => reads++));

        var cached = RowCache.Of(ml, source);

        var sums = new List<double>();
        for (var pass = 0; pass < 2; pass++)
        {
            var column = cached.Schema["Features"];
            using var cursor = cached.GetRowCursor([column]);
            var get = cursor.GetGetter<VBuffer<float>>(column);
            VBuffer<float> value = default;
            double sum = 0;
            var rows = 0;
            while (cursor.MoveNext())
            {
                get(ref value);
                sum += value.GetValues()[1];
                rows++;
            }
            Assert.Equal(50_000, rows);
            sums.Add(sum);
        }

        Assert.Equal(sums[0], sums[1]);
        Assert.Equal(Enumerable.Range(0, 50_000).Sum(i => i + 0.5), sums[0]);
        Assert.Equal(1, reads);
    }

    [Fact]
    public void ANullArgument_IsRefused()
    {
        var ml = new MLContext(1);
        var data = ml.Data.LoadFromEnumerable(Rows(1, () => { }));

        Assert.Throws<ArgumentNullException>(() => RowCache.Of(null!, data));
        Assert.Throws<ArgumentNullException>(() => RowCache.Of(ml, null!));
    }
}
