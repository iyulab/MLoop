using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace MLoop.Core.Training;

/// <summary>
/// How many principal components a table's anomaly model keeps. The anomaly score is how far a row is from what the kept
/// components rebuild, so the number kept decides what a score means: with as many components as columns the rebuild is exact
/// and every score is rounding noise (a five-column table scored an area under the curve of 0.44 — worse than chance), and
/// with too few the model reads ordinary variation as strange.
/// </summary>
/// <remarks>
/// The rank is the fewest components that carry <see cref="ExplainedShare"/> of the variance of the standardized columns — a
/// property of the data, which no fixed rule (every column, half, a quarter) was: on eight public tables none of those won
/// on all of them, and the variance rule did. The variances come from one fit at the highest rank allowed, read off its own
/// eigenvectors, because the trainer reports its eigenvectors and not their eigenvalues.
/// </remarks>
internal static class AnomalyPcaRank
{
    /// <summary>The share of the variance the kept components carry.</summary>
    public const double ExplainedShare = 0.9;

    /// <summary>The most components kept, however many columns carry the variance.</summary>
    public const int MaxRank = 20;

    /// <summary>The most components a table of <paramref name="featureCount"/> columns can keep and still leave a rebuild error to score.</summary>
    public static int Ceiling(int featureCount) => Math.Max(1, Math.Min(featureCount - 1, MaxRank));

    /// <summary>
    /// The fewest components, in order of the variance they carry, whose sum reaches <see cref="ExplainedShare"/> of
    /// <paramref name="totalVariance"/>; <paramref name="ceiling"/> when they do not reach it (or there is no variance to share).
    /// </summary>
    public static int Choose(IReadOnlyList<double> componentVariance, double totalVariance, int ceiling)
    {
        ArgumentNullException.ThrowIfNull(componentVariance);
        if (!double.IsFinite(totalVariance) || totalVariance <= 0)
            return ceiling;

        double cumulative = 0;
        var ordered = componentVariance.OrderByDescending(v => v).ToArray();
        for (var i = 0; i < ordered.Length; i++)
        {
            cumulative += ordered[i];
            if (cumulative >= ExplainedShare * totalVariance)
                return Math.Min(i + 1, ceiling);
        }
        return ceiling;
    }

    /// <summary>
    /// The variance each of the fitted model's components carries over <paramref name="featurized"/> (rows with a
    /// <c>Features</c> vector), and the variance in all directions together.
    /// </summary>
    public static (double[] Component, double Total) Measure(IDataView featurized, PcaModelParameters pca)
    {
        ArgumentNullException.ThrowIfNull(featurized);
        ArgumentNullException.ThrowIfNull(pca);

        VBuffer<float>[] vectors = default!;
        pca.GetEigenVectors(ref vectors, out _);
        VBuffer<float> meanBuffer = default;
        pca.GetMean(ref meanBuffer);
        var mean = meanBuffer.DenseValues().ToArray();
        var axes = vectors.Select(v => v.DenseValues().ToArray()).ToArray();

        var component = new double[axes.Length];
        double total = 0;
        var features = featurized.Schema["Features"];
        using var cursor = featurized.GetRowCursor(new[] { features });
        var getter = cursor.GetGetter<VBuffer<float>>(features);
        VBuffer<float> row = default;
        while (cursor.MoveNext())
        {
            getter(ref row);
            var values = row.DenseValues().ToArray();
            for (var j = 0; j < values.Length && j < mean.Length; j++)
                values[j] -= mean[j];

            total += values.Sum(v => (double)v * v);
            for (var k = 0; k < axes.Length; k++)
            {
                double projection = 0;
                var axis = axes[k];
                for (var j = 0; j < values.Length && j < axis.Length; j++)
                    projection += axis[j] * values[j];
                component[k] += projection * projection;
            }
        }
        return (component, total);
    }
}
