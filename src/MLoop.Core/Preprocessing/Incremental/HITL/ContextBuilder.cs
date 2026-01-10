using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;
using System.Text;

namespace MLoop.Core.Preprocessing.Incremental.HITL;

/// <summary>
/// Builds rich context information for HITL questions.
/// </summary>
internal sealed class ContextBuilder
{
    /// <summary>
    /// Builds context string for a preprocessing rule.
    /// </summary>
    public string BuildContext(
        PreprocessingRule rule,
        DataFrame sample,
        SampleAnalysis analysis)
    {
        var context = new StringBuilder();

        // Basic statistics
        var affectedPercentage = (double)rule.AffectedRows / sample.Rows.Count;
        context.Append($"Found {rule.AffectedRows:N0} affected records in '{rule.ColumnNames[0]}' column ");
        context.Append($"({affectedPercentage:P1} of {sample.Rows.Count:N0} records)");

        // Add pattern-specific context
        context.AppendLine();
        context.Append(BuildPatternSpecificContext(rule, sample));

        return context.ToString();
    }

    /// <summary>
    /// Builds context specific to the pattern type.
    /// </summary>
    private string BuildPatternSpecificContext(PreprocessingRule rule, DataFrame sample)
    {
        return rule.PatternType switch
        {
            PatternType.MissingValue => BuildMissingValueContext(rule, sample),
            PatternType.OutlierAnomaly => BuildOutlierContext(rule, sample),
            PatternType.CategoryVariation => BuildCategoryContext(rule, sample),
            PatternType.TypeInconsistency => BuildTypeContext(rule, sample),
            _ => string.Empty
        };
    }

    private string BuildMissingValueContext(PreprocessingRule rule, DataFrame sample)
    {
        if (rule.Examples == null || rule.Examples.Count == 0)
            return string.Empty;

        var context = new StringBuilder();
        context.Append("Null indicators found: ");
        context.Append(string.Join(", ", rule.Examples.Take(5).Select(e => $"'{e}'")));

        return context.ToString();
    }

    private string BuildOutlierContext(PreprocessingRule rule, DataFrame sample)
    {
        if (rule.Examples == null || rule.Examples.Count == 0)
            return string.Empty;

        var context = new StringBuilder();
        context.Append($"Outlier values: {string.Join(", ", rule.Examples.Take(3))}");

        // Calculate stats if numeric column
        var column = sample.Columns[rule.ColumnNames[0]];
        if (column is PrimitiveDataFrameColumn<double> doubleCol)
        {
            var mean = CalculateMean(doubleCol);
            var median = CalculateMedian(doubleCol);
            context.Append($" (vs mean: {mean:F1}, median: {median:F1})");
        }
        else if (column is PrimitiveDataFrameColumn<int> intCol)
        {
            var mean = CalculateMean(intCol);
            var median = CalculateMedian(intCol);
            context.Append($" (vs mean: {mean:F0}, median: {median:F0})");
        }

        return context.ToString();
    }

    private string BuildCategoryContext(PreprocessingRule rule, DataFrame sample)
    {
        if (rule.Examples == null || rule.Examples.Count == 0)
            return string.Empty;

        var variations = rule.Examples.Take(5).ToList();
        return $"Category variations found: {string.Join(", ", variations.Select(v => $"'{v}'"))}";
    }

    private string BuildTypeContext(PreprocessingRule rule, DataFrame sample)
    {
        if (rule.Examples == null || rule.Examples.Count < 2)
            return string.Empty;

        return $"Mixed types detected: {rule.Examples[0]} (type A) vs {rule.Examples[1]} (type B)";
    }

    /// <summary>
    /// Calculates mean for a double column.
    /// </summary>
    private static double CalculateMean(PrimitiveDataFrameColumn<double> column)
    {
        double sum = 0;
        int count = 0;

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value.HasValue && !double.IsNaN(value.Value) && !double.IsInfinity(value.Value))
            {
                sum += value.Value;
                count++;
            }
        }

        return count > 0 ? sum / count : 0;
    }

    /// <summary>
    /// Calculates mean for an int column.
    /// </summary>
    private static double CalculateMean(PrimitiveDataFrameColumn<int> column)
    {
        long sum = 0;
        int count = 0;

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value.HasValue)
            {
                sum += value.Value;
                count++;
            }
        }

        return count > 0 ? (double)sum / count : 0;
    }

    /// <summary>
    /// Calculates median for a double column.
    /// </summary>
    private static double CalculateMedian(PrimitiveDataFrameColumn<double> column)
    {
        var values = new List<double>();

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value.HasValue && !double.IsNaN(value.Value) && !double.IsInfinity(value.Value))
            {
                values.Add(value.Value);
            }
        }

        if (values.Count == 0)
            return 0;

        values.Sort();
        var mid = values.Count / 2;

        return values.Count % 2 == 0
            ? (values[mid - 1] + values[mid]) / 2
            : values[mid];
    }

    /// <summary>
    /// Calculates median for an int column.
    /// </summary>
    private static double CalculateMedian(PrimitiveDataFrameColumn<int> column)
    {
        var values = new List<int>();

        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value.HasValue)
            {
                values.Add(value.Value);
            }
        }

        if (values.Count == 0)
            return 0;

        values.Sort();
        var mid = values.Count / 2;

        return values.Count % 2 == 0
            ? (values[mid - 1] + values[mid]) / 2.0
            : values[mid];
    }
}
