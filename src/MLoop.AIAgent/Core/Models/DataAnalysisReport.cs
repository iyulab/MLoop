namespace MLoop.AIAgent.Core.Models;

/// <summary>
/// Comprehensive data analysis report
/// </summary>
public class DataAnalysisReport
{
    /// <summary>
    /// File path of analyzed data
    /// </summary>
    public required string FilePath { get; init; }

    /// <summary>
    /// Number of rows in the dataset
    /// </summary>
    public required int RowCount { get; init; }

    /// <summary>
    /// Number of columns in the dataset
    /// </summary>
    public required int ColumnCount { get; init; }

    /// <summary>
    /// File size in bytes
    /// </summary>
    public long FileSizeBytes { get; init; }

    /// <summary>
    /// Column analysis results
    /// </summary>
    public required List<ColumnAnalysis> Columns { get; init; }

    /// <summary>
    /// Recommended target variable for ML
    /// </summary>
    public TargetRecommendation? RecommendedTarget { get; init; }

    /// <summary>
    /// Data quality issues detected
    /// </summary>
    public required DataQualityIssues QualityIssues { get; init; }

    /// <summary>
    /// Overall ML readiness assessment
    /// </summary>
    public required MLReadinessAssessment MLReadiness { get; init; }

    /// <summary>
    /// Analysis timestamp
    /// </summary>
    public DateTime AnalyzedAt { get; init; } = DateTime.UtcNow;
}

/// <summary>
/// Analysis of a single column
/// </summary>
public class ColumnAnalysis
{
    /// <summary>
    /// Column name
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Inferred data type
    /// </summary>
    public required DataType InferredType { get; init; }

    /// <summary>
    /// Number of non-null values
    /// </summary>
    public required int NonNullCount { get; init; }

    /// <summary>
    /// Number of null/missing values
    /// </summary>
    public required int NullCount { get; init; }

    /// <summary>
    /// Percentage of missing values (0-100)
    /// </summary>
    public double MissingPercentage =>
        NonNullCount + NullCount > 0
            ? (NullCount / (double)(NonNullCount + NullCount)) * 100.0
            : 0.0;

    /// <summary>
    /// Number of unique values
    /// </summary>
    public required int UniqueCount { get; init; }

    /// <summary>
    /// Numeric statistics (if applicable)
    /// </summary>
    public NumericStatistics? NumericStats { get; init; }

    /// <summary>
    /// Categorical statistics (if applicable)
    /// </summary>
    public CategoricalStatistics? CategoricalStats { get; init; }

    /// <summary>
    /// Sample values from the column
    /// </summary>
    public List<string> SampleValues { get; init; } = new();
}

/// <summary>
/// Data type classification
/// </summary>
public enum DataType
{
    Numeric,
    Categorical,
    Text,
    DateTime,
    Boolean,
    Unknown
}

/// <summary>
/// Statistical summary for numeric columns
/// </summary>
public class NumericStatistics
{
    public required double Mean { get; init; }
    public required double Median { get; init; }
    public required double StandardDeviation { get; init; }
    public required double Variance { get; init; }
    public required double Min { get; init; }
    public required double Max { get; init; }
    public required double Q1 { get; init; } // 25th percentile
    public required double Q3 { get; init; } // 75th percentile
    public double IQR => Q3 - Q1;
    public int OutlierCount { get; init; }
    public List<double> OutlierValues { get; init; } = new();
}

/// <summary>
/// Statistical summary for categorical columns
/// </summary>
public class CategoricalStatistics
{
    public required string MostFrequentValue { get; init; }
    public required int MostFrequentCount { get; init; }
    public required Dictionary<string, int> ValueCounts { get; init; }
    public double Cardinality => ValueCounts.Count;
    public bool IsHighCardinality => ValueCounts.Count > 50; // Arbitrary threshold
}

/// <summary>
/// Target variable recommendation
/// </summary>
public class TargetRecommendation
{
    public required string ColumnName { get; init; }
    public required MLProblemType ProblemType { get; init; }
    public required double Confidence { get; init; } // 0-1
    public required string Reason { get; init; }
    public List<string> AlternativeTargets { get; init; } = new();
}

/// <summary>
/// ML problem type classification
/// </summary>
public enum MLProblemType
{
    BinaryClassification,
    MulticlassClassification,
    Regression,
    Unknown
}

/// <summary>
/// Data quality issues summary
/// </summary>
public class DataQualityIssues
{
    public List<string> ColumnsWithMissingValues { get; init; } = new();
    public List<string> ColumnsWithOutliers { get; init; } = new();
    public List<string> HighCardinalityColumns { get; init; } = new();
    public List<string> ConstantColumns { get; init; } = new(); // All values the same
    public List<string> DuplicateColumns { get; init; } = new();
    public int DuplicateRowCount { get; init; }

    public bool HasIssues =>
        ColumnsWithMissingValues.Count > 0 ||
        ColumnsWithOutliers.Count > 0 ||
        HighCardinalityColumns.Count > 0 ||
        ConstantColumns.Count > 0 ||
        DuplicateRowCount > 0;
}

/// <summary>
/// ML readiness assessment
/// </summary>
public class MLReadinessAssessment
{
    public required bool IsReady { get; init; }
    public required double ReadinessScore { get; init; } // 0-1
    public required List<string> BlockingIssues { get; init; }
    public required List<string> Warnings { get; init; }
    public required List<string> Recommendations { get; init; }
}
