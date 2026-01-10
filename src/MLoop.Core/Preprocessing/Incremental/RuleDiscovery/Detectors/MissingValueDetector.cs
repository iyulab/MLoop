using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Contracts;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;

/// <summary>
/// Detects missing values in columns.
/// Identifies NULL, "N/A", empty strings, and common missing value representations.
/// </summary>
public sealed class MissingValueDetector : IPatternDetector
{
    public PatternType PatternType => PatternType.MissingValue;

    public Task<IReadOnlyList<DetectedPattern>> DetectAsync(
        DataFrameColumn column,
        string columnName,
        CancellationToken cancellationToken = default)
    {
        var patterns = new List<DetectedPattern>();

        int missingCount = 0;
        var examples = new List<string>();

        for (long i = 0; i < column.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var value = DetectorHelpers.GetStringValue(column, i);

            if (DetectorHelpers.IsMissingValue(value))
            {
                missingCount++;

                // Collect examples (up to 5 unique representations)
                var representation = value ?? "[NULL]";
                if (examples.Count < 5 && !examples.Contains(representation))
                {
                    examples.Add(representation);
                }
            }
        }

        // Only report if there are missing values
        if (missingCount > 0)
        {
            var affectedPercentage = DetectorHelpers.CalculatePercentage(missingCount, column.Length);
            var severity = DetectorHelpers.DetermineSeverity(affectedPercentage);

            var pattern = new DetectedPattern
            {
                Type = PatternType.MissingValue,
                ColumnName = columnName,
                Description = $"{affectedPercentage:P1} missing values ({string.Join(", ", examples)})",
                Severity = severity,
                Occurrences = missingCount,
                TotalRows = (int)column.Length,
                Confidence = 1.0, // Deterministic detection
                Examples = examples,
                SuggestedFix = DetermineSuggestedFix(severity, affectedPercentage)
            };

            patterns.Add(pattern);
        }

        return Task.FromResult<IReadOnlyList<DetectedPattern>>(patterns);
    }

    public bool IsApplicable(DataFrameColumn column)
    {
        // Missing value detection applies to all column types
        return true;
    }

    private static string DetermineSuggestedFix(Severity severity, double affectedPercentage)
    {
        return severity switch
        {
            Severity.Critical => "Consider dropping column or collecting better data",
            Severity.High => "Impute with median/mode or use predictive model",
            Severity.Medium => "Impute with median/mode or forward/backward fill",
            Severity.Low => "Impute with median/mode or drop rows",
            _ => "Review and decide based on domain knowledge"
        };
    }
}
