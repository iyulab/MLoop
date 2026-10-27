using System.Numerics;

namespace MLoop.Core.ImageAnomaly;

/// <summary>
/// Chooses which patches a size-limited bank keeps: a greedy k-center coreset (the choice PatchCore makes) — start from
/// one patch and keep adding the patch farthest from every one kept, so that however few are kept, no kind of patch the
/// images showed is left far from all of them. A random subsample of the same size drops a rare kind (a texture seen on
/// one image in fifty) as often as it keeps it; the coreset does not.
/// </summary>
/// <remarks>
/// Distances are measured after a seeded random projection to <see cref="ProjectedDimension"/> numbers (the
/// Johnson–Lindenstrauss step PatchCore uses), which keeps the work at patches × kept × 128 however wide the encoder is.
/// What is chosen is a row of the original data, never the projected one.
/// </remarks>
public static class CoresetSelector
{
    /// <summary>The width the patches are projected to before their distances are measured.</summary>
    public const int ProjectedDimension = 128;

    private const int ChunkRows = 2048;

    /// <summary>
    /// The <paramref name="count"/> rows of <paramref name="data"/> (rows of <paramref name="dimension"/> numbers) the
    /// coreset keeps, ascending. All rows when <paramref name="count"/> is at least the number of rows.
    /// </summary>
    /// <param name="data">The patches, row after row.</param>
    /// <param name="dimension">The numbers in one patch.</param>
    /// <param name="count">How many patches to keep.</param>
    /// <param name="seed">Seeds the projection and the first patch: the same data gives the same coreset.</param>
    /// <param name="cancellationToken">Stops the choice between two patches.</param>
    public static int[] Select(float[] data, int dimension, int count, int seed, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (dimension <= 0 || data.Length % dimension != 0)
            throw new ArgumentException($"{data.Length} numbers are not rows of {dimension}.", nameof(data));
        if (count <= 0)
            throw new ArgumentOutOfRangeException(nameof(count), "A coreset keeps at least one patch.");
        var rows = data.Length / dimension;
        if (rows == 0)
            return [];
        if (count >= rows)
            return [.. Enumerable.Range(0, rows)];

        var random = new Random(seed);
        var width = Math.Min(ProjectedDimension, dimension);
        var projected = Project(data, dimension, rows, width, random);

        // distance² of every row to the nearest kept one; the next patch kept is where it is largest.
        var nearest = new float[rows];
        Array.Fill(nearest, float.PositiveInfinity);
        var chosen = new int[count];
        var next = random.Next(rows);
        var partials = new (float Distance, int Row)[(rows + ChunkRows - 1) / ChunkRows];

        for (var kept = 0; kept < count; kept++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            chosen[kept] = next;
            var from = projected.AsMemory(next * width, width);
            Parallel.For(0, partials.Length, chunk =>
            {
                var source = from.Span;
                var first = chunk * ChunkRows;
                var last = Math.Min(first + ChunkRows, rows);
                var farthest = (Distance: -1f, Row: first);
                for (var row = first; row < last; row++)
                {
                    var distance = SquaredDistance(source, projected.AsSpan(row * width, width));
                    if (distance < nearest[row]) nearest[row] = distance;
                    if (nearest[row] > farthest.Distance) farthest = (nearest[row], row);
                }
                partials[chunk] = farthest;
            });
            var best = partials[0];
            for (var i = 1; i < partials.Length; i++)
                if (partials[i].Distance > best.Distance) best = partials[i];
            next = best.Row;
        }

        Array.Sort(chosen);
        return chosen;
    }

    private static float[] Project(float[] data, int dimension, int rows, int width, Random random)
    {
        if (width == dimension)
            return data;

        // A Gaussian matrix, scaled so a distance keeps its length on average.
        var matrix = new float[width * dimension];
        var scale = 1f / MathF.Sqrt(width);
        for (var i = 0; i < matrix.Length; i++)
        {
            var u1 = 1.0 - random.NextDouble();
            var u2 = random.NextDouble();
            matrix[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2)) * scale;
        }

        var projected = new float[checked(rows * width)];
        Parallel.For(0, (rows + ChunkRows - 1) / ChunkRows, chunk =>
        {
            var first = chunk * ChunkRows;
            var last = Math.Min(first + ChunkRows, rows);
            for (var row = first; row < last; row++)
            {
                var source = data.AsSpan(row * dimension, dimension);
                for (var k = 0; k < width; k++)
                    projected[row * width + k] = Dot(source, matrix.AsSpan(k * dimension, dimension));
            }
        });
        return projected;
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sum = Vector<float>.Zero;
        var i = 0;
        for (; i <= a.Length - Vector<float>.Count; i += Vector<float>.Count)
            sum += new Vector<float>(a[i..]) * new Vector<float>(b[i..]);
        var total = Vector.Sum(sum);
        for (; i < a.Length; i++)
            total += a[i] * b[i];
        return total;
    }

    private static float SquaredDistance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sum = Vector<float>.Zero;
        var i = 0;
        for (; i <= a.Length - Vector<float>.Count; i += Vector<float>.Count)
        {
            var d = new Vector<float>(a[i..]) - new Vector<float>(b[i..]);
            sum += d * d;
        }
        var total = Vector.Sum(sum);
        for (; i < a.Length; i++)
        {
            var d = a[i] - b[i];
            total += d * d;
        }
        return total;
    }
}
