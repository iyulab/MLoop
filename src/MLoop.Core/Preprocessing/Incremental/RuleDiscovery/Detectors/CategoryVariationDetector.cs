using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;

/// <summary>
/// Detects category variations due to typos, case differences, or similar spellings.
/// Helps identify categories that should be merged or normalized.
/// </summary>
public sealed class CategoryVariationDetector : IPatternDetector
{
    private const int MaxCategories = 100; // Threshold for categorical data
    private const double SimilarityThreshold = 0.85; // Levenshtein similarity

    public PatternType PatternType => PatternType.CategoryVariation;

    public Task<IReadOnlyList<DetectedPattern>> DetectAsync(
        DataFrameColumn column,
        string columnName,
        CancellationToken cancellationToken = default)
    {
        var patterns = new List<DetectedPattern>();

        if (!ColumnTypeHelper.IsStringColumn(column))
        {
            return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
        }

        // Count unique categories
        var categoryCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var caseVariations = new Dictionary<string, HashSet<string>>();

        for (long i = 0; i < column.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var value = DetectorHelpers.GetStringValue(column, i);
            if (string.IsNullOrWhiteSpace(value) || DetectorHelpers.IsMissingValue(value))
                continue;

            var trimmed = value.Trim();
            var normalized = trimmed.ToUpperInvariant();

            categoryCounts.TryGetValue(normalized, out var count);
            categoryCounts[normalized] = count + 1;

            if (!caseVariations.ContainsKey(normalized))
            {
                caseVariations[normalized] = new HashSet<string>();
            }
            caseVariations[normalized].Add(trimmed);
        }

        // Skip if too many unique values (likely not categorical)
        if (categoryCounts.Count > MaxCategories)
        {
            return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
        }

        // Detect case variations
        var caseVariationPattern = DetectCaseVariations(
            column,
            columnName,
            caseVariations);

        if (caseVariationPattern != null)
            patterns.Add(caseVariationPattern);

        // Detect similar categories (potential typos)
        var typoPattern = DetectSimilarCategories(
            column,
            columnName,
            categoryCounts.Keys.ToList(),
            cancellationToken);

        if (typoPattern != null)
            patterns.Add(typoPattern);

        return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
    }

    public bool IsApplicable(DataFrameColumn column)
    {
        return ColumnTypeHelper.IsStringColumn(column);
    }

    private DetectedPattern? DetectCaseVariations(
        DataFrameColumn column,
        string columnName,
        Dictionary<string, HashSet<string>> caseVariations)
    {
        var categoriesWithVariations = caseVariations
            .Where(kvp => kvp.Value.Count > 1)
            .ToList();

        if (categoriesWithVariations.Count == 0)
            return null;

        var examples = new List<string>();
        foreach (var variation in categoriesWithVariations.Take(3))
        {
            examples.Add($"{string.Join(", ", variation.Value)}");
        }

        var totalVariations = categoriesWithVariations.Sum(v => v.Value.Count - 1);
        var affectedPercentage = DetectorHelpers.CalculatePercentage(
            totalVariations,
            column.Length);

        return new DetectedPattern
        {
            Type = PatternType.CategoryVariation,
            ColumnName = columnName,
            Description = $"Case variations in {categoriesWithVariations.Count} categories",
            Severity = Severity.Low,
            Occurrences = totalVariations,
            TotalRows = (int)column.Length,
            Confidence = 0.95,
            Examples = examples,
            SuggestedFix = "Normalize to lowercase or proper case"
        };
    }

    private DetectedPattern? DetectSimilarCategories(
        DataFrameColumn column,
        string columnName,
        List<string> categories,
        CancellationToken cancellationToken)
    {
        var similarPairs = new List<(string, string)>();

        for (int i = 0; i < categories.Count - 1; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            for (int j = i + 1; j < categories.Count; j++)
            {
                var similarity = CalculateLevenshteinSimilarity(
                    categories[i],
                    categories[j]);

                if (similarity >= SimilarityThreshold)
                {
                    similarPairs.Add((categories[i], categories[j]));
                }
            }
        }

        if (similarPairs.Count == 0)
            return null;

        var examples = similarPairs
            .Take(5)
            .Select(p => $"{p.Item1} ≈ {p.Item2}")
            .ToList();

        var affectedCount = similarPairs.Count * 2; // Each pair affects 2 categories
        var affectedPercentage = DetectorHelpers.CalculatePercentage(
            affectedCount,
            column.Length);

        return new DetectedPattern
        {
            Type = PatternType.CategoryVariation,
            ColumnName = columnName,
            Description = $"{similarPairs.Count} similar category pairs (potential typos)",
            Severity = Severity.Medium,
            Occurrences = affectedCount,
            TotalRows = (int)column.Length,
            Confidence = 0.80, // Fuzzy matching is less certain
            Examples = examples,
            SuggestedFix = "Review similar categories and merge if typos, keep separate if distinct"
        };
    }

    /// <summary>
    /// Calculate Levenshtein similarity (0-1, where 1 is identical).
    /// </summary>
    private static double CalculateLevenshteinSimilarity(string s1, string s2)
    {
        var distance = CalculateLevenshteinDistance(s1, s2);
        var maxLength = Math.Max(s1.Length, s2.Length);

        if (maxLength == 0)
            return 1.0;

        return 1.0 - ((double)distance / maxLength);
    }

    /// <summary>
    /// Calculate Levenshtein distance between two strings.
    /// </summary>
    private static int CalculateLevenshteinDistance(string s1, string s2)
    {
        if (string.IsNullOrEmpty(s1))
            return s2?.Length ?? 0;

        if (string.IsNullOrEmpty(s2))
            return s1.Length;

        var matrix = new int[s1.Length + 1, s2.Length + 1];

        for (int i = 0; i <= s1.Length; i++)
            matrix[i, 0] = i;

        for (int j = 0; j <= s2.Length; j++)
            matrix[0, j] = j;

        for (int i = 1; i <= s1.Length; i++)
        {
            for (int j = 1; j <= s2.Length; j++)
            {
                var cost = s1[i - 1] == s2[j - 1] ? 0 : 1;

                matrix[i, j] = Math.Min(
                    Math.Min(
                        matrix[i - 1, j] + 1,      // Deletion
                        matrix[i, j - 1] + 1),     // Insertion
                    matrix[i - 1, j - 1] + cost);  // Substitution
            }
        }

        return matrix[s1.Length, s2.Length];
    }
}
