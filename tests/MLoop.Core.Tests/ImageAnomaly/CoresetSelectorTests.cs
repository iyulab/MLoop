using MLoop.Core.ImageAnomaly;

namespace MLoop.Core.Tests.ImageAnomaly;

/// <summary>
/// A size-limited bank keeps the patches that cover the kinds the images showed — a greedy k-center coreset — where a
/// random subsample of the same size drops a rare kind as often as it keeps it.
/// </summary>
public class CoresetSelectorTests
{
    /// <summary>Clusters of unit-length-ish points: <c>sizes[c]</c> rows near the c-th corner of the first axes.</summary>
    private static float[] Clusters(int dimension, params int[] sizes)
    {
        var random = new Random(7);
        var rows = new List<float[]>();
        for (var c = 0; c < sizes.Length; c++)
            for (var i = 0; i < sizes[c]; i++)
            {
                var row = new float[dimension];
                for (var d = 0; d < dimension; d++) row[d] = (float)(random.NextDouble() * 0.02);
                row[c] += 1f;
                rows.Add(row);
            }
        return [.. rows.SelectMany(r => r)];
    }

    [Fact]
    public void ARareKind_IsKept_ByAsManyPatchesAsThereAreKinds()
    {
        // 1,000 + 1,000 + 5 rows, 160 numbers wide (wider than the projection): three kinds, one of them rare
        var data = Clusters(160, 1000, 1000, 5);

        var chosen = CoresetSelector.Select(data, 160, 3, seed: 42);

        Assert.Equal(3, chosen.Length);
        Assert.Equal([true, true, true], new[] { chosen.Any(r => r < 1000), chosen.Any(r => r is >= 1000 and < 2000), chosen.Any(r => r >= 2000) });
    }

    [Fact]
    public void TheChoice_IsSeeded_Ascending_AndWithoutRepeats()
    {
        var data = Clusters(160, 300, 300, 300);

        var first = CoresetSelector.Select(data, 160, 40, seed: 5);
        var again = CoresetSelector.Select(data, 160, 40, seed: 5);

        Assert.Equal(first, again);
        Assert.Equal(40, first.Distinct().Count());
        Assert.Equal(first.Order(), first);
    }

    [Fact]
    public void ACountAtLeastTheRows_KeepsEveryRow()
    {
        var data = Clusters(8, 4, 4);

        Assert.Equal(Enumerable.Range(0, 8), CoresetSelector.Select(data, 8, 8, seed: 1));
        Assert.Equal(Enumerable.Range(0, 8), CoresetSelector.Select(data, 8, 100, seed: 1));
    }

    [Fact]
    public void ANarrowEncoder_IsMeasuredWithoutAProjection_AndStillCoversTheKinds()
    {
        var data = Clusters(4, 50, 50, 3);

        var chosen = CoresetSelector.Select(data, 4, 3, seed: 9);

        Assert.Equal([true, true, true], new[] { chosen.Any(r => r < 50), chosen.Any(r => r is >= 50 and < 100), chosen.Any(r => r >= 100) });
    }

    [Fact]
    public void RowsThatAreNotWholeOrACountOfZero_AreRefused()
    {
        Assert.Throws<ArgumentException>(() => CoresetSelector.Select(new float[10], 4, 2, seed: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => CoresetSelector.Select(new float[8], 4, 0, seed: 1));
    }
}
