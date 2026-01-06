using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;

/// <summary>
/// Detects type inconsistencies in string columns.
/// Identifies mixed numeric/string, coercion issues, and type mismatches.
/// </summary>
public sealed class TypeInconsistencyDetector : IPatternDetector
{
    public PatternType PatternType => PatternType.TypeInconsistency;

    public Task<IReadOnlyList<DetectedPattern>> DetectAsync(
        DataFrameColumn column,
        string columnName,
        CancellationToken cancellationToken = default)
    {
        var patterns = new List<DetectedPattern>();

        // Only applicable to string columns (typed columns are consistent by definition)
        if (!ColumnTypeHelper.IsStringColumn(column))
        {
            return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
        }

        var inferredType = ColumnTypeHelper.InferColumnType(column);

        // Only report if mixed or ambiguous types detected
        if (inferredType != ColumnTypeHelper.InferredType.Mixed)
        {
            return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
        }

        // Analyze type distribution
        var typeDistribution = AnalyzeTypeDistribution(column, cancellationToken);

        if (typeDistribution.InconsistentCount > 0)
        {
            var affectedPercentage = DetectorHelpers.CalculatePercentage(
                typeDistribution.InconsistentCount,
                column.Length);

            var severity = DetectorHelpers.DetermineSeverity(affectedPercentage);

            var pattern = new DetectedPattern
            {
                Type = PatternType.TypeInconsistency,
                ColumnName = columnName,
                Description = BuildDescription(typeDistribution),
                Severity = severity,
                Occurrences = typeDistribution.InconsistentCount,
                TotalRows = (int)column.Length,
                Confidence = 0.95, // High confidence in type detection
                Examples = typeDistribution.Examples,
                SuggestedFix = DetermineSuggestedFix(typeDistribution)
            };

            patterns.Add(pattern);
        }

        return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
    }

    public bool IsApplicable(DataFrameColumn column)
    {
        // Only applicable to string columns
        return ColumnTypeHelper.IsStringColumn(column);
    }

    private static TypeDistribution AnalyzeTypeDistribution(
        DataFrameColumn column,
        CancellationToken cancellationToken)
    {
        var distribution = new TypeDistribution();

        for (long i = 0; i < column.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var value = DetectorHelpers.GetStringValue(column, i);
            if (string.IsNullOrWhiteSpace(value) || DetectorHelpers.IsMissingValue(value))
            {
                distribution.NullCount++;
                continue;
            }

            // Check each type
            if (ColumnTypeHelper.TryParseNumeric(value, out _))
            {
                distribution.NumericCount++;
                if (distribution.NumericExamples.Count < 3)
                    distribution.NumericExamples.Add(value);
            }
            else if (ColumnTypeHelper.TryParseDateTime(value, out _))
            {
                distribution.DateTimeCount++;
                if (distribution.DateTimeExamples.Count < 3)
                    distribution.DateTimeExamples.Add(value);
            }
            else if (ColumnTypeHelper.TryParseBoolean(value, out _))
            {
                distribution.BooleanCount++;
                if (distribution.BooleanExamples.Count < 3)
                    distribution.BooleanExamples.Add(value);
            }
            else
            {
                distribution.TextCount++;
                if (distribution.TextExamples.Count < 3)
                    distribution.TextExamples.Add(value);
            }
        }

        // Calculate inconsistency
        var typeCounts = new[]
        {
            distribution.NumericCount,
            distribution.DateTimeCount,
            distribution.BooleanCount,
            distribution.TextCount
        }.Where(c => c > 0).ToList();

        // Inconsistent if multiple types present and minority type is >5%
        if (typeCounts.Count > 1)
        {
            var totalValues = distribution.NumericCount + distribution.DateTimeCount +
                            distribution.BooleanCount + distribution.TextCount;
            var minorityCount = typeCounts.Min();
            var minorityRatio = (double)minorityCount / totalValues;

            if (minorityRatio > 0.05) // >5% minority type
            {
                distribution.InconsistentCount = minorityCount;
            }
        }

        return distribution;
    }

    private static string BuildDescription(TypeDistribution dist)
    {
        var parts = new List<string>();

        if (dist.NumericCount > 0)
            parts.Add($"{dist.NumericCount} numeric");
        if (dist.DateTimeCount > 0)
            parts.Add($"{dist.DateTimeCount} datetime");
        if (dist.BooleanCount > 0)
            parts.Add($"{dist.BooleanCount} boolean");
        if (dist.TextCount > 0)
            parts.Add($"{dist.TextCount} text");

        return $"Mixed types: {string.Join(", ", parts)}";
    }

    private static string DetermineSuggestedFix(TypeDistribution dist)
    {
        // Recommend conversion to majority type
        var maxCount = Math.Max(
            Math.Max(dist.NumericCount, dist.DateTimeCount),
            Math.Max(dist.BooleanCount, dist.TextCount));

        if (maxCount == dist.NumericCount)
            return "Convert to numeric, handle non-numeric as NULL or default";
        if (maxCount == dist.DateTimeCount)
            return "Convert to datetime, standardize format";
        if (maxCount == dist.BooleanCount)
            return "Convert to boolean, map text values";

        return "Keep as text, validate and normalize values";
    }

    private sealed class TypeDistribution
    {
        public int NumericCount { get; set; }
        public int DateTimeCount { get; set; }
        public int BooleanCount { get; set; }
        public int TextCount { get; set; }
        public int NullCount { get; set; }
        public int InconsistentCount { get; set; }

        public List<string> NumericExamples { get; } = new();
        public List<string> DateTimeExamples { get; } = new();
        public List<string> BooleanExamples { get; } = new();
        public List<string> TextExamples { get; } = new();

        public List<string> Examples
        {
            get
            {
                var examples = new List<string>();
                examples.AddRange(NumericExamples);
                examples.AddRange(DateTimeExamples);
                examples.AddRange(BooleanExamples);
                examples.AddRange(TextExamples);
                return examples.Take(5).ToList();
            }
        }
    }
}
