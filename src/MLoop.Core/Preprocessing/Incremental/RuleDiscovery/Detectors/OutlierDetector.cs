using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;

/// <summary>
/// Detects statistical outliers in numeric columns.
/// Uses Z-score and IQR methods for outlier detection.
/// </summary>
public sealed class OutlierDetector : IPatternDetector
{
    private const double DefaultZScoreThreshold = 3.0;

    public PatternType PatternType => PatternType.OutlierAnomaly;

    public Task<IReadOnlyList<DetectedPattern>> DetectAsync(
        DataFrameColumn column,
        string columnName,
        CancellationToken cancellationToken = default)
    {
        var patterns = new List<DetectedPattern>();

        if (!ColumnTypeHelper.IsNumericColumn(column))
        {
            return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
        }

        // Use both methods and take the union
        var zScoreOutliers = StatisticalHelper.DetectOutliersZScore(column, DefaultZScoreThreshold);
        var iqrOutliers = StatisticalHelper.DetectOutliersIQR(column);

        // Combine outliers (union of both methods)
        var allOutlierIndices = zScoreOutliers
            .Select(o => o.RowIndex)
            .Union(iqrOutliers.Select(o => o.RowIndex))
            .ToHashSet();

        if (allOutlierIndices.Count == 0)
        {
            return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
        }

        var affectedPercentage = DetectorHelpers.CalculatePercentage(
            allOutlierIndices.Count,
            column.Length);

        // Only report if outliers are significant but not overwhelming
        if (affectedPercentage < 0.01 || affectedPercentage > 0.30)
        {
            // Too few (<1%) or too many (>30%) suggests data issue, not outliers
            return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
        }

        var severity = DetermineSeverity(affectedPercentage);

        // Get example values
        var examples = GetOutlierExamples(column, allOutlierIndices);

        var pattern = new DetectedPattern
        {
            Type = PatternType.OutlierAnomaly,
            ColumnName = columnName,
            Description = BuildDescription(zScoreOutliers, iqrOutliers, affectedPercentage),
            Severity = severity,
            Occurrences = allOutlierIndices.Count,
            TotalRows = (int)column.Length,
            Confidence = 0.85, // Statistical methods are reliable but context-dependent
            Examples = examples,
            SuggestedFix = DetermineSuggestedFix(severity)
        };

        patterns.Add(pattern);

        return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
    }

    public bool IsApplicable(DataFrameColumn column)
    {
        return ColumnTypeHelper.IsNumericColumn(column);
    }

    private static Severity DetermineSeverity(double affectedPercentage)
    {
        return affectedPercentage switch
        {
            >= 0.20 => Severity.High,      // >20% outliers
            >= 0.10 => Severity.Medium,    // 10-20% outliers
            _ => Severity.Low              // 1-10% outliers
        };
    }

    private static List<string> GetOutlierExamples(
        DataFrameColumn column,
        HashSet<long> outlierIndices)
    {
        var examples = new List<string>();
        var sampleIndices = outlierIndices.Take(5);

        foreach (var index in sampleIndices)
        {
            var value = column[index];
            if (value != null)
            {
                examples.Add($"Row {index}: {value}");
            }
        }

        return examples;
    }

    private static string BuildDescription(
        List<StatisticalHelper.OutlierInfo> zScoreOutliers,
        List<StatisticalHelper.OutlierInfo> iqrOutliers,
        double affectedPercentage)
    {
        var methods = new List<string>();

        if (zScoreOutliers.Count > 0)
            methods.Add($"{zScoreOutliers.Count} by Z-score");

        if (iqrOutliers.Count > 0)
            methods.Add($"{iqrOutliers.Count} by IQR");

        return $"{affectedPercentage:P1} outliers detected ({string.Join(", ", methods)})";
    }

    private static string DetermineSuggestedFix(Severity severity)
    {
        return severity switch
        {
            Severity.High => "Review data collection process, consider capping or removing extreme values",
            Severity.Medium => "Investigate outliers, consider using robust statistical methods",
            Severity.Low => "Document outliers, consider keeping if legitimate data points",
            _ => "Review and decide based on domain knowledge"
        };
    }
}
