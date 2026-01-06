using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging;
using MLoop.Core.Preprocessing.Incremental.Contracts;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.Statistics;

namespace MLoop.Core.Preprocessing.Incremental;

/// <summary>
/// Main analyzer for DataFrame samples providing comprehensive statistical analysis.
/// </summary>
public sealed class SampleAnalyzer : ISampleAnalyzer
{
    private readonly ILogger<SampleAnalyzer>? _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="SampleAnalyzer"/> class.
    /// </summary>
    public SampleAnalyzer(ILogger<SampleAnalyzer>? logger = null)
    {
        _logger = logger;
    }

    /// <inheritdoc/>
    public async Task<SampleAnalysis> AnalyzeAsync(
        DataFrame sample,
        int stageNumber,
        AnalysisConfiguration? config = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(sample);

        config ??= new AnalysisConfiguration();

        _logger?.LogInformation(
            "Analyzing sample stage {Stage}: {Rows} rows, {Columns} columns",
            stageNumber, sample.Rows.Count, sample.Columns.Count);

        // Analyze each column
        var columnAnalyses = new List<ColumnAnalysis>();

        await Task.Run(() =>
        {
            for (int i = 0; i < sample.Columns.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var column = sample.Columns[i];
                var analysis = AnalyzeColumn(column, column.Name, i);
                columnAnalyses.Add(analysis);
            }
        }, cancellationToken);

        // Calculate quality score
        double qualityScore = CalculateQualityScore(columnAnalyses);

        // Estimate memory footprint
        long estimatedMemory = EstimateMemoryFootprint(sample);

        var sampleAnalysis = new SampleAnalysis
        {
            StageNumber = stageNumber,
            SampleRatio = (double)sample.Rows.Count / sample.Rows.Count, // Will be set by caller
            Timestamp = DateTime.UtcNow,
            RowCount = sample.Rows.Count,
            ColumnCount = sample.Columns.Count,
            Columns = columnAnalyses,
            QualityScore = qualityScore,
            EstimatedMemoryBytes = estimatedMemory
        };

        _logger?.LogInformation(
            "Analysis complete: Quality {Quality:P0}, {Issues} issues",
            qualityScore, sampleAnalysis.AllQualityIssues.Count);

        return sampleAnalysis;
    }

    /// <inheritdoc/>
    public ColumnAnalysis AnalyzeColumn(DataFrameColumn column, string columnName, int columnIndex)
    {
        ArgumentNullException.ThrowIfNull(column);

        var config = new AnalysisConfiguration();

        // Count nulls
        long nullCount = CountNulls(column);
        long nonNullCount = column.Length - nullCount;

        // Detect data type
        var dataType = DetectDataType(column);

        // Analyze based on type
        NumericStats? numericStats = null;
        CategoricalStats? categoricalStats = null;

        if (dataType is DataType.Integer or DataType.Float or DataType.Double or DataType.Decimal && nonNullCount > 0)
        {
            numericStats = NumericAnalyzer.Analyze(column, config);
        }
        else if (dataType is DataType.String or DataType.Boolean && nonNullCount > 0)
        {
            categoricalStats = CategoricalAnalyzer.Analyze(column, config);
        }

        // Detect quality issues
        var qualityIssues = DetectQualityIssues(column, nullCount, numericStats, categoricalStats);

        // Generate recommendations
        var recommendations = GenerateRecommendations(dataType, nullCount, (long)column.Length, numericStats, categoricalStats);

        return new ColumnAnalysis
        {
            ColumnName = columnName,
            ColumnIndex = columnIndex,
            DataType = dataType,
            TotalRows = column.Length,
            NonNullCount = nonNullCount,
            NullCount = nullCount,
            NumericStats = numericStats,
            CategoricalStats = categoricalStats,
            QualityIssues = qualityIssues,
            RecommendedActions = recommendations
        };
    }

    /// <inheritdoc/>
    public bool HasConverged(SampleAnalysis previous, SampleAnalysis current, double threshold = 0.01)
    {
        ArgumentNullException.ThrowIfNull(previous);
        ArgumentNullException.ThrowIfNull(current);

        // Compare key statistics across columns
        var variances = new List<double>();

        for (int i = 0; i < Math.Min(previous.Columns.Count, current.Columns.Count); i++)
        {
            var prevCol = previous.Columns[i];
            var currCol = current.Columns[i];

            if (prevCol.NumericStats != null && currCol.NumericStats != null)
            {
                // Compare means
                var meanVariance = Math.Abs(prevCol.NumericStats.Mean - currCol.NumericStats.Mean) /
                                 Math.Max(Math.Abs(prevCol.NumericStats.Mean), 1.0);
                variances.Add(meanVariance);

                // Compare standard deviations
                var stdVariance = Math.Abs(prevCol.NumericStats.StandardDeviation - currCol.NumericStats.StandardDeviation) /
                                Math.Max(prevCol.NumericStats.StandardDeviation, 1.0);
                variances.Add(stdVariance);
            }

            if (prevCol.CategoricalStats != null && currCol.CategoricalStats != null)
            {
                // Compare entropy
                var entropyVariance = Math.Abs(prevCol.CategoricalStats.Entropy - currCol.CategoricalStats.Entropy) /
                                    Math.Max(prevCol.CategoricalStats.Entropy, 1.0);
                variances.Add(entropyVariance);
            }
        }

        if (variances.Count == 0) return false;

        // Converged if average variance is below threshold
        double avgVariance = variances.Average();
        bool converged = avgVariance < threshold;

        _logger?.LogDebug(
            "Convergence check: avg variance {Variance:F4}, threshold {Threshold:F4}, converged: {Converged}",
            avgVariance, threshold, converged);

        return converged;
    }

    /// <summary>
    /// Counts null values in a column.
    /// </summary>
    private static long CountNulls(DataFrameColumn column)
    {
        long count = 0;
        for (int i = 0; i < column.Length; i++)
        {
            if (column[i] == null)
                count++;
        }
        return count;
    }

    /// <summary>
    /// Detects the data type of a column.
    /// </summary>
    private static DataType DetectDataType(DataFrameColumn column)
    {
        return column switch
        {
            PrimitiveDataFrameColumn<int> => DataType.Integer,
            PrimitiveDataFrameColumn<long> => DataType.Integer,
            PrimitiveDataFrameColumn<float> => DataType.Float,
            PrimitiveDataFrameColumn<double> => DataType.Double,
            PrimitiveDataFrameColumn<decimal> => DataType.Decimal,
            PrimitiveDataFrameColumn<bool> => DataType.Boolean,
            StringDataFrameColumn => DataType.String,
            _ => DataType.Unknown
        };
    }

    /// <summary>
    /// Detects data quality issues in a column.
    /// </summary>
    private static List<DataQualityIssue> DetectQualityIssues(
        DataFrameColumn column,
        long nullCount,
        NumericStats? numericStats,
        CategoricalStats? categoricalStats)
    {
        var issues = new List<DataQualityIssue>();

        double missingPercentage = (double)nullCount / column.Length * 100.0;

        // High missing values
        if (missingPercentage > 50)
        {
            issues.Add(new DataQualityIssue
            {
                Severity = IssueSeverity.High,
                IssueType = "HighMissingValues",
                Description = $"{missingPercentage:F1}% missing values",
                SuggestedFix = "Consider dropping this column or imputing values"
            });
        }
        else if (missingPercentage > 20)
        {
            issues.Add(new DataQualityIssue
            {
                Severity = IssueSeverity.Medium,
                IssueType = "ModerateMissingValues",
                Description = $"{missingPercentage:F1}% missing values",
                SuggestedFix = "Consider imputation strategy (mean, median, mode)"
            });
        }

        // High outlier count
        if (numericStats != null && numericStats.OutlierPercentage > 5)
        {
            issues.Add(new DataQualityIssue
            {
                Severity = IssueSeverity.Medium,
                IssueType = "HighOutliers",
                Description = $"{numericStats.OutlierPercentage:F1}% outliers detected",
                SuggestedFix = "Review outliers - may indicate data quality issues or genuine extreme values"
            });
        }

        // High cardinality categorical
        if (categoricalStats != null && categoricalStats.IsHighCardinality)
        {
            issues.Add(new DataQualityIssue
            {
                Severity = IssueSeverity.Low,
                IssueType = "HighCardinality",
                Description = $"High cardinality: {categoricalStats.UniqueCount} unique values",
                SuggestedFix = "Consider target encoding or embedding instead of one-hot encoding"
            });
        }

        return issues;
    }

    /// <summary>
    /// Generates preprocessing recommendations.
    /// </summary>
    private static List<string> GenerateRecommendations(
        DataType dataType,
        long nullCount,
        long totalCount,
        NumericStats? numericStats,
        CategoricalStats? categoricalStats)
    {
        var recommendations = new List<string>();

        if (nullCount > 0)
        {
            double missingPercentage = (double)nullCount / totalCount * 100.0;

            if (missingPercentage > 50)
                recommendations.Add("Drop column due to excessive missing values");
            else if (dataType is DataType.Integer or DataType.Float or DataType.Double)
                recommendations.Add("Fill missing numeric values with median");
            else
                recommendations.Add("Fill missing categorical values with mode or 'Unknown'");
        }

        if (categoricalStats != null)
        {
            if (categoricalStats.IsLowCardinality)
                recommendations.Add("Apply one-hot encoding");
            else if (categoricalStats.IsHighCardinality)
                recommendations.Add("Apply target encoding or embedding");
            else if (categoricalStats.IsLikelyIdentifier)
                recommendations.Add("Drop column - likely an identifier with no predictive value");
        }

        if (numericStats != null && numericStats.OutlierCount > 0)
        {
            recommendations.Add("Review outliers and consider clipping or transformation");
        }

        return recommendations;
    }

    /// <summary>
    /// Calculates overall quality score (0.0 to 1.0).
    /// </summary>
    private static double CalculateQualityScore(List<ColumnAnalysis> columns)
    {
        if (columns.Count == 0) return 0.0;

        double totalScore = 0.0;

        foreach (var col in columns)
        {
            double colScore = 1.0;

            // Penalize missing values
            double missingPenalty = col.MissingPercentage / 100.0 * 0.5;
            colScore -= missingPenalty;

            // Penalize quality issues
            int criticalIssues = col.QualityIssues.Count(i => i.Severity >= IssueSeverity.High);
            double issuePenalty = Math.Min(criticalIssues * 0.2, 0.4);
            colScore -= issuePenalty;

            totalScore += Math.Max(colScore, 0.0);
        }

        return totalScore / columns.Count;
    }

    /// <summary>
    /// Estimates memory footprint in bytes.
    /// </summary>
    private static long EstimateMemoryFootprint(DataFrame sample)
    {
        long bytes = 0;

        foreach (var column in sample.Columns)
        {
            long columnBytes = column switch
            {
                PrimitiveDataFrameColumn<int> => column.Length * sizeof(int),
                PrimitiveDataFrameColumn<long> => column.Length * sizeof(long),
                PrimitiveDataFrameColumn<float> => column.Length * sizeof(float),
                PrimitiveDataFrameColumn<double> => column.Length * sizeof(double),
                PrimitiveDataFrameColumn<bool> => column.Length * sizeof(bool),
                StringDataFrameColumn => column.Length * 50, // Rough estimate
                _ => column.Length * 8
            };

            bytes += columnBytes;
        }

        return bytes;
    }
}
