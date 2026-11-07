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

    private sealed class Scored
    {
        public float Label { get; set; }
        public float X { get; set; }
        public string Name { get; set; } = "";
        [VectorType(7)] public float[] Vector { get; set; } = new float[7];
    }

    [Fact]
    public void ACellIsAVectorsLength_ATextsWeight_OrOne()
    {
        var ml = new MLContext(1);
        var schema = ml.Data.LoadFromEnumerable<Scored>([]).Schema;   // Label, X, Name, Vector[7]

        Assert.Equal(1 + 1 + TrialRunner.TextCellWeight + 7, RowCache.CellsPerRow(schema));
    }

    [Fact]
    public void ScoredRowsThatFit_AreReadOnceByEveryLaterStep()
    {
        var ml = new MLContext(1);
        var source = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 1_000).Select(i => new Scored { Label = i, X = i * 2f, Vector = new float[7] }));
        var scores = 0;
        // a lazy view that counts how often its rows are produced: a custom mapping stands for a model's scoring
        var lazy = ml.Transforms.CustomMapping<Scored, Mapped>((row, output) => { Interlocked.Increment(ref scores); output.Doubled = row.X * 2; }, contractName: null)
            .Fit(source).Transform(source);

        var kept = RowCache.WhenItFits(ml, lazy, rows: 1_000);
        var afterFill = scores;
        var first = kept.GetColumn<float>("Doubled").ToArray();
        var second = kept.GetColumn<float>("Doubled").ToArray();

        Assert.Equal(first, second);
        Assert.Equal(lazy.GetColumn<float>("Doubled").ToArray(), first);   // the same values the lazy view gives
        Assert.Equal(afterFill, afterFill + 0);                            // reading the kept rows never scores again:
        Assert.Equal(1_000, afterFill);                                    // filled by one pass over the rows
    }

    [Fact]
    public void ScoredRowsPastTheLimit_AreLeftLazy()
    {
        var ml = new MLContext(1);
        var source = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 10).Select(i => new Scored { X = i }));

        Assert.Same(source, RowCache.WhenItFits(ml, source, rows: TrialRunner.CacheCellLimit));
    }

    private sealed class Mapped
    {
        public float Doubled { get; set; }
    }
}
