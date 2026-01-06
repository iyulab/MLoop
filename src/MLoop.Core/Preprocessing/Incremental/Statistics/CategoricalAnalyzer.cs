using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;

namespace MLoop.Core.Preprocessing.Incremental.Statistics;

/// <summary>
/// Analyzer for categorical columns providing frequency and entropy metrics.
/// </summary>
public static class CategoricalAnalyzer
{
    /// <summary>
    /// Analyzes a categorical column and calculates comprehensive statistics.
    /// </summary>
    public static CategoricalStats Analyze(DataFrameColumn column, AnalysisConfiguration config)
    {
        ArgumentNullException.ThrowIfNull(column);
        ArgumentNullException.ThrowIfNull(config);

        // Count value frequencies
        var valueCounts = CountValues(column);

        if (valueCounts.Count == 0)
        {
            return CreateEmptyStats();
        }

        // Sort by frequency descending
        var sortedCounts = valueCounts.OrderByDescending(kvp => kvp.Value).ToList();

        // Take top N values
        var topValues = sortedCounts
            .Take(config.MaxCategoricalValues)
            .Select(kvp => (kvp.Key, kvp.Value))
            .ToList();

        // Calculate entropy
        long totalCount = valueCounts.Values.Sum();
        double entropy = CalculateEntropy(valueCounts.Values, totalCount);

        return new CategoricalStats
        {
            Count = totalCount,
            UniqueCount = valueCounts.Count,
            TopValues = topValues,
            Entropy = entropy
        };
    }

    /// <summary>
    /// Counts occurrences of each unique value in the column.
    /// </summary>
    private static Dictionary<string, long> CountValues(DataFrameColumn column)
    {
        var counts = new Dictionary<string, long>();

        for (int i = 0; i < column.Length; i++)
        {
            var value = column[i]?.ToString() ?? "NULL";

            if (counts.TryGetValue(value, out var count))
            {
                counts[value] = count + 1;
            }
            else
            {
                counts[value] = 1;
            }
        }

        return counts;
    }

    /// <summary>
    /// Calculates Shannon entropy: H = -Σ(p(x) * log2(p(x))).
    /// </summary>
    /// <remarks>
    /// Entropy measures information content:
    /// - 0: All values are the same (no information)
    /// - log2(n): Uniform distribution (maximum information)
    /// </remarks>
    private static double CalculateEntropy(IEnumerable<long> counts, long total)
    {
        if (total == 0) return 0.0;

        double entropy = 0.0;

        foreach (var count in counts)
        {
            if (count > 0)
            {
                double probability = (double)count / total;
                entropy -= probability * Math.Log2(probability);
            }
        }

        return entropy;
    }

    /// <summary>
    /// Creates empty stats for columns with no valid values.
    /// </summary>
    private static CategoricalStats CreateEmptyStats()
    {
        return new CategoricalStats
        {
            Count = 0,
            UniqueCount = 0,
            TopValues = new List<(string Value, long Frequency)>(),
            Entropy = 0.0
        };
    }
}
