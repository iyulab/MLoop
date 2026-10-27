using System.Numerics;
using System.Runtime.InteropServices;

namespace MLoop.Core.ImageAnomaly;

/// <summary>
/// What an image's patches measure against a <see cref="PatchBank"/>: for each patch the cosine distance (1 − dot, the
/// vectors being unit length) to the nearest vector in the bank, and for the image the largest of them. A patch of a
/// kind the normal images never showed is far from all of them, wherever in the image it is — and the distances, laid
/// over the grid, are where.
/// </summary>
public static class PatchBankScorer
{
    // Rows of the bank taken together: a block of 256 rows × 384 numbers is 384 KB, read once and kept in cache
    // while every query patch is measured against it.
    private const int BlockRows = 256;

    /// <summary>
    /// For each of <paramref name="queries"/> the distance to the nearest vector of <paramref name="bank"/>.
    /// The bank is read once, in blocks spread over the cores; each query is measured against each block while the
    /// block is in cache.
    /// </summary>
    /// <exception cref="ArgumentException">A query's length is not the bank's dimension, or the bank is empty.</exception>
    public static float[] NearestDistances(PatchBank bank, IReadOnlyList<float[]> queries)
    {
        ArgumentNullException.ThrowIfNull(bank);
        ArgumentNullException.ThrowIfNull(queries);
        if (bank.Count == 0)
            throw new ArgumentException("The bank holds no vectors: nothing to measure against.", nameof(bank));
        foreach (var query in queries)
            if (query.Length != bank.Dimension)
                throw new ArgumentException($"A query of {query.Length} numbers against a bank of {bank.Dimension}.", nameof(queries));

        var best = new float[queries.Count];
        Array.Fill(best, float.NegativeInfinity);
        var blocks = (bank.Count + BlockRows - 1) / BlockRows;
        var gate = new object();

        Parallel.For(0, blocks,
            () => NewLocal(queries.Count),
            (block, _, local) =>
            {
                var first = block * BlockRows;
                var last = Math.Min(first + BlockRows, bank.Count);
                for (var q = 0; q < queries.Count; q++)
                {
                    var query = queries[q];
                    var top = local[q];
                    for (var row = first; row < last; row++)
                    {
                        var dot = Dot(query, bank[row]);
                        if (dot > top) top = dot;
                    }
                    local[q] = top;
                }
                return local;
            },
            local =>
            {
                lock (gate)
                    for (var q = 0; q < best.Length; q++)
                        if (local[q] > best[q]) best[q] = local[q];
            });

        for (var q = 0; q < best.Length; q++)
            best[q] = Math.Max(0f, 1f - best[q]);
        return best;

        static float[] NewLocal(int count)
        {
            var local = new float[count];
            Array.Fill(local, float.NegativeInfinity);
            return local;
        }
    }

    /// <summary>The image's score — the largest of its patches' nearest distances — and the distances themselves.</summary>
    public static (float Score, float[] Distances) Score(PatchBank bank, IReadOnlyList<float[]> patches)
    {
        var distances = NearestDistances(bank, patches);
        return (distances.Length == 0 ? 0f : distances.Max(), distances);
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var width = Vector<float>.Count;
        var sum = Vector<float>.Zero;
        var i = 0;
        if (a.Length >= width)
        {
            var va = MemoryMarshal.Cast<float, Vector<float>>(a[..(a.Length / width * width)]);
            var vb = MemoryMarshal.Cast<float, Vector<float>>(b[..(b.Length / width * width)]);
            for (var v = 0; v < va.Length; v++)
                sum += va[v] * vb[v];
            i = va.Length * width;
        }
        var total = Vector.Sum(sum);
        for (; i < a.Length; i++)
            total += a[i] * b[i];
        return total;
    }
}
