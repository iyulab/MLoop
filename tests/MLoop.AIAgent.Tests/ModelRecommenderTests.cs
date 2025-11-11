using MLoop.AIAgent.Core;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Tests;

public class ModelRecommenderTests
{
    private readonly ModelRecommender _recommender = new();

    [Fact]
    public void RecommendModel_BinaryClassification_ReturnsCorrectConfiguration()
    {
        // Arrange
        var report = CreateSampleReport(
            problemType: MLProblemType.BinaryClassification,
            rowCount: 1000,
            uniqueTargetValues: 2);

        // Act
        var recommendation = _recommender.RecommendModel(report);

        // Assert
        Assert.Equal(MLProblemType.BinaryClassification, recommendation.ProblemType);
        Assert.Equal("target", recommendation.TargetColumn);
        Assert.Contains("feature1", recommendation.FeatureColumns);
        Assert.Contains("feature2", recommendation.FeatureColumns);
        Assert.True(recommendation.RecommendedTrainingTimeSeconds > 0);
        Assert.NotEmpty(recommendation.PrimaryMetric);
        Assert.NotEmpty(recommendation.RecommendedTrainers);
        Assert.True(recommendation.Confidence > 0 && recommendation.Confidence <= 1);
    }

    [Fact]
    public void RecommendModel_MulticlassClassification_ReturnsCorrectConfiguration()
    {
        // Arrange
        var report = CreateSampleReport(
            problemType: MLProblemType.MulticlassClassification,
            rowCount: 5000,
            uniqueTargetValues: 5);

        // Act
        var recommendation = _recommender.RecommendModel(report);

        // Assert
        Assert.Equal(MLProblemType.MulticlassClassification, recommendation.ProblemType);
        Assert.Equal("target", recommendation.TargetColumn);
        Assert.NotEmpty(recommendation.AdditionalMetrics);
        Assert.NotEmpty(recommendation.RecommendedTrainers);
    }

    [Fact]
    public void RecommendModel_Regression_ReturnsCorrectConfiguration()
    {
        // Arrange
        var report = CreateSampleReport(
            problemType: MLProblemType.Regression,
            rowCount: 2000,
            uniqueTargetValues: 1500); // High cardinality for regression

        // Act
        var recommendation = _recommender.RecommendModel(report);

        // Assert
        Assert.Equal(MLProblemType.Regression, recommendation.ProblemType);
        Assert.Equal("R2Score", recommendation.PrimaryMetric);
        Assert.Contains("RMSE", recommendation.AdditionalMetrics);
        Assert.Contains("RSquared", recommendation.OptimizationMetric);
    }

    [Fact]
    public void RecommendModel_SmallDataset_ReducesConfidence()
    {
        // Arrange
        var smallReport = CreateSampleReport(
            problemType: MLProblemType.BinaryClassification,
            rowCount: 50, // Very small
            uniqueTargetValues: 2);

        var largeReport = CreateSampleReport(
            problemType: MLProblemType.BinaryClassification,
            rowCount: 10000,
            uniqueTargetValues: 2);

        // Act
        var smallRecommendation = _recommender.RecommendModel(smallReport);
        var largeRecommendation = _recommender.RecommendModel(largeReport);

        // Assert
        Assert.True(smallRecommendation.Confidence < largeRecommendation.Confidence,
            "Small dataset should have lower confidence");
        Assert.Contains(smallRecommendation.Warnings,
            w => w.Contains("Dataset is very small"));
    }

    [Fact]
    public void RecommendModel_ImbalancedDataset_RecommendsFScore()
    {
        // Arrange
        var report = CreateImbalancedReport(imbalanceRatio: 10.0);

        // Act
        var recommendation = _recommender.RecommendModel(report);

        // Assert
        Assert.Equal("F1Score", recommendation.PrimaryMetric);
        Assert.Contains("F1Score", recommendation.OptimizationMetric);
        Assert.Contains(recommendation.Warnings, w => w.Contains("imbalanced"));
    }

    [Fact]
    public void RecommendModel_BalancedDataset_RecommendsAccuracy()
    {
        // Arrange
        var report = CreateSampleReport(
            problemType: MLProblemType.BinaryClassification,
            rowCount: 1000,
            uniqueTargetValues: 2);

        // Act
        var recommendation = _recommender.RecommendModel(report);

        // Assert
        Assert.Equal("Accuracy", recommendation.PrimaryMetric);
        Assert.Contains("Accuracy", recommendation.OptimizationMetric);
    }

    [Fact]
    public void RecommendModel_MissingValues_GeneratesPreprocessingRecommendations()
    {
        // Arrange
        // 35% missing in one column = ~11.67% overall across 3 columns (target + 2 features)
        var report = CreateReportWithMissingValues(missingPercentage: 35.0);

        // Act
        var recommendation = _recommender.RecommendModel(report);

        // Assert
        Assert.Contains(recommendation.PreprocessingRecommendations,
            r => r.Contains("missing", StringComparison.OrdinalIgnoreCase));
        Assert.Contains(recommendation.Warnings,
            w => w.Contains("missing", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void RecommendModel_HighCardinalityFeatures_GeneratesWarning()
    {
        // Arrange
        var report = CreateReportWithHighCardinality();

        // Act
        var recommendation = _recommender.RecommendModel(report);

        // Assert
        Assert.Contains(recommendation.Warnings,
            w => w.Contains("High-cardinality"));
        Assert.Contains(recommendation.PreprocessingRecommendations,
            r => r.Contains("high-cardinality"));
    }

    [Fact]
    public void RecommendModel_LargeDataset_RecommendsLongerTrainingTime()
    {
        // Arrange
        var smallReport = CreateSampleReport(
            problemType: MLProblemType.BinaryClassification,
            rowCount: 500,
            uniqueTargetValues: 2);

        var largeReport = CreateSampleReport(
            problemType: MLProblemType.BinaryClassification,
            rowCount: 50000,
            uniqueTargetValues: 2);

        // Act
        var smallRecommendation = _recommender.RecommendModel(smallReport);
        var largeRecommendation = _recommender.RecommendModel(largeReport);

        // Assert
        Assert.True(largeRecommendation.RecommendedTrainingTimeSeconds >
                   smallRecommendation.RecommendedTrainingTimeSeconds,
            "Larger dataset should have longer training time");
    }

    [Fact]
    public void RecommendModel_RecommendsSuitableTrainers()
    {
        // Arrange
        var report = CreateSampleReport(
            problemType: MLProblemType.BinaryClassification,
            rowCount: 1000,
            uniqueTargetValues: 2);

        // Act
        var recommendation = _recommender.RecommendModel(report);

        // Assert
        Assert.NotEmpty(recommendation.RecommendedTrainers);
        Assert.Contains("LightGbm", recommendation.RecommendedTrainers);
        Assert.Contains("FastTree", recommendation.RecommendedTrainers);
    }

    [Fact]
    public void RecommendModel_NoTargetColumn_ThrowsException()
    {
        // Arrange
        var baseReport = CreateSampleReport(
            problemType: MLProblemType.BinaryClassification,
            rowCount: 1000,
            uniqueTargetValues: 2);

        // Create new report without target
        var report = new DataAnalysisReport
        {
            FilePath = baseReport.FilePath,
            RowCount = baseReport.RowCount,
            ColumnCount = baseReport.ColumnCount,
            Columns = baseReport.Columns,
            RecommendedTarget = null,
            QualityIssues = baseReport.QualityIssues,
            MLReadiness = baseReport.MLReadiness
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _recommender.RecommendModel(report));
    }

    [Fact]
    public void RecommendModel_MoreFeaturesThanSamples_GeneratesWarning()
    {
        // Arrange
        var report = CreateReportWithManyFeatures(
            rowCount: 50,
            featureCount: 100); // More features than samples

        // Act
        var recommendation = _recommender.RecommendModel(report);

        // Assert
        Assert.Contains(recommendation.Warnings,
            w => w.Contains("More features than samples"));
    }

    [Fact]
    public void RecommendModel_CategoricalFeatures_RecommendsEncoding()
    {
        // Arrange
        var report = CreateReportWithCategoricalFeatures();

        // Act
        var recommendation = _recommender.RecommendModel(report);

        // Assert
        Assert.Contains(recommendation.PreprocessingRecommendations,
            r => r.Contains("categorical") || r.Contains("encoding"));
    }

    [Fact]
    public void RecommendModel_NumericFeatures_RecommendsNormalization()
    {
        // Arrange
        var report = CreateSampleReport(
            problemType: MLProblemType.Regression,
            rowCount: 1000,
            uniqueTargetValues: 800);

        // Act
        var recommendation = _recommender.RecommendModel(report);

        // Assert
        Assert.Contains(recommendation.PreprocessingRecommendations,
            r => r.Contains("normalizing") || r.Contains("scaling"));
    }

    // Helper methods
    private DataAnalysisReport CreateSampleReport(
        MLProblemType problemType,
        int rowCount,
        int uniqueTargetValues)
    {
        var targetColumn = new ColumnAnalysis
        {
            Name = "target",
            InferredType = problemType == MLProblemType.Regression
                ? DataType.Numeric
                : DataType.Categorical,
            NonNullCount = rowCount,
            NullCount = 0,
            UniqueCount = uniqueTargetValues,
            CategoricalStats = problemType != MLProblemType.Regression
                ? new CategoricalStatistics
                {
                    MostFrequentValue = "ClassA",
                    MostFrequentCount = rowCount / 2,
                    ValueCounts = new Dictionary<string, int>
                    {
                        ["ClassA"] = rowCount / 2,
                        ["ClassB"] = rowCount / 2
                    }
                }
                : null
        };

        var feature1 = new ColumnAnalysis
        {
            Name = "feature1",
            InferredType = DataType.Numeric,
            NonNullCount = rowCount,
            NullCount = 0,
            UniqueCount = rowCount,
            NumericStats = new NumericStatistics
            {
                Mean = 50,
                Median = 50,
                StandardDeviation = 10,
                Variance = 100,
                Min = 0,
                Max = 100,
                Q1 = 40,
                Q3 = 60
            }
        };

        var feature2 = new ColumnAnalysis
        {
            Name = "feature2",
            InferredType = DataType.Numeric,
            NonNullCount = rowCount,
            NullCount = 0,
            UniqueCount = rowCount,
            NumericStats = new NumericStatistics
            {
                Mean = 75,
                Median = 75,
                StandardDeviation = 15,
                Variance = 225,
                Min = 0,
                Max = 150,
                Q1 = 60,
                Q3 = 90
            }
        };

        return new DataAnalysisReport
        {
            FilePath = "test.csv",
            RowCount = rowCount,
            ColumnCount = 3,
            Columns = new List<ColumnAnalysis> { targetColumn, feature1, feature2 },
            RecommendedTarget = new TargetRecommendation
            {
                ColumnName = "target",
                ProblemType = problemType,
                Confidence = 0.8,
                Reason = "Test target"
            },
            QualityIssues = new DataQualityIssues(),
            MLReadiness = new MLReadinessAssessment
            {
                IsReady = true,
                ReadinessScore = 0.9,
                BlockingIssues = new List<string>(),
                Warnings = new List<string>(),
                Recommendations = new List<string>()
            }
        };
    }

    private DataAnalysisReport CreateImbalancedReport(double imbalanceRatio)
    {
        var majorityCount = (int)(1000 * imbalanceRatio / (imbalanceRatio + 1));
        var minorityCount = 1000 - majorityCount;

        var report = CreateSampleReport(
            MLProblemType.BinaryClassification,
            1000,
            2);

        var targetColumn = report.Columns.First(c => c.Name == "target");
        var updatedTarget = new ColumnAnalysis
        {
            Name = targetColumn.Name,
            InferredType = targetColumn.InferredType,
            NonNullCount = targetColumn.NonNullCount,
            NullCount = targetColumn.NullCount,
            UniqueCount = targetColumn.UniqueCount,
            CategoricalStats = new CategoricalStatistics
            {
                MostFrequentValue = "ClassA",
                MostFrequentCount = majorityCount,
                ValueCounts = new Dictionary<string, int>
                {
                    ["ClassA"] = majorityCount,
                    ["ClassB"] = minorityCount
                }
            }
        };

        var updatedColumns = report.Columns.Where(c => c.Name != "target").ToList();
        updatedColumns.Insert(0, updatedTarget);

        return new DataAnalysisReport
        {
            FilePath = report.FilePath,
            RowCount = report.RowCount,
            ColumnCount = report.ColumnCount,
            Columns = updatedColumns,
            RecommendedTarget = report.RecommendedTarget,
            QualityIssues = report.QualityIssues,
            MLReadiness = report.MLReadiness
        };
    }

    private DataAnalysisReport CreateReportWithMissingValues(double missingPercentage)
    {
        var report = CreateSampleReport(
            MLProblemType.BinaryClassification,
            1000,
            2);

        var missingCount = (int)(1000 * missingPercentage / 100);

        var feature1 = report.Columns.First(c => c.Name == "feature1");
        var updatedFeature = new ColumnAnalysis
        {
            Name = feature1.Name,
            InferredType = feature1.InferredType,
            NonNullCount = 1000 - missingCount,
            NullCount = missingCount,
            UniqueCount = feature1.UniqueCount,
            NumericStats = feature1.NumericStats
        };

        var updatedColumns = report.Columns.Where(c => c.Name != "feature1").ToList();
        updatedColumns.Add(updatedFeature);

        var updatedQualityIssues = new DataQualityIssues
        {
            ColumnsWithMissingValues = new List<string> { "feature1" }
        };

        return new DataAnalysisReport
        {
            FilePath = report.FilePath,
            RowCount = report.RowCount,
            ColumnCount = report.ColumnCount,
            Columns = updatedColumns,
            RecommendedTarget = report.RecommendedTarget,
            QualityIssues = updatedQualityIssues,
            MLReadiness = report.MLReadiness
        };
    }

    private DataAnalysisReport CreateReportWithHighCardinality()
    {
        var report = CreateSampleReport(
            MLProblemType.BinaryClassification,
            1000,
            2);

        var updatedQualityIssues = new DataQualityIssues
        {
            HighCardinalityColumns = new List<string> { "high_cardinality_feature" }
        };

        return new DataAnalysisReport
        {
            FilePath = report.FilePath,
            RowCount = report.RowCount,
            ColumnCount = report.ColumnCount,
            Columns = report.Columns,
            RecommendedTarget = report.RecommendedTarget,
            QualityIssues = updatedQualityIssues,
            MLReadiness = report.MLReadiness
        };
    }

    private DataAnalysisReport CreateReportWithManyFeatures(int rowCount, int featureCount)
    {
        var targetColumn = new ColumnAnalysis
        {
            Name = "target",
            InferredType = DataType.Categorical,
            NonNullCount = rowCount,
            NullCount = 0,
            UniqueCount = 2,
            CategoricalStats = new CategoricalStatistics
            {
                MostFrequentValue = "ClassA",
                MostFrequentCount = rowCount / 2,
                ValueCounts = new Dictionary<string, int>
                {
                    ["ClassA"] = rowCount / 2,
                    ["ClassB"] = rowCount / 2
                }
            }
        };

        var columns = new List<ColumnAnalysis> { targetColumn };

        for (int i = 0; i < featureCount; i++)
        {
            columns.Add(new ColumnAnalysis
            {
                Name = $"feature{i}",
                InferredType = DataType.Numeric,
                NonNullCount = rowCount,
                NullCount = 0,
                UniqueCount = rowCount,
                NumericStats = new NumericStatistics
                {
                    Mean = 50,
                    Median = 50,
                    StandardDeviation = 10,
                    Variance = 100,
                    Min = 0,
                    Max = 100,
                    Q1 = 40,
                    Q3 = 60
                }
            });
        }

        return new DataAnalysisReport
        {
            FilePath = "test.csv",
            RowCount = rowCount,
            ColumnCount = columns.Count,
            Columns = columns,
            RecommendedTarget = new TargetRecommendation
            {
                ColumnName = "target",
                ProblemType = MLProblemType.BinaryClassification,
                Confidence = 0.8,
                Reason = "Test target"
            },
            QualityIssues = new DataQualityIssues(),
            MLReadiness = new MLReadinessAssessment
            {
                IsReady = true,
                ReadinessScore = 0.9,
                BlockingIssues = new List<string>(),
                Warnings = new List<string>(),
                Recommendations = new List<string>()
            }
        };
    }

    private DataAnalysisReport CreateReportWithCategoricalFeatures()
    {
        var report = CreateSampleReport(
            MLProblemType.BinaryClassification,
            1000,
            2);

        var categoricalFeature = new ColumnAnalysis
        {
            Name = "categorical_feature",
            InferredType = DataType.Categorical,
            NonNullCount = 1000,
            NullCount = 0,
            UniqueCount = 5,
            CategoricalStats = new CategoricalStatistics
            {
                MostFrequentValue = "Category1",
                MostFrequentCount = 300,
                ValueCounts = new Dictionary<string, int>
                {
                    ["Category1"] = 300,
                    ["Category2"] = 250,
                    ["Category3"] = 200,
                    ["Category4"] = 150,
                    ["Category5"] = 100
                }
            }
        };

        var updatedColumns = report.Columns.ToList();
        updatedColumns.Add(categoricalFeature);

        return new DataAnalysisReport
        {
            FilePath = report.FilePath,
            RowCount = report.RowCount,
            ColumnCount = updatedColumns.Count,
            Columns = updatedColumns,
            RecommendedTarget = report.RecommendedTarget,
            QualityIssues = report.QualityIssues,
            MLReadiness = report.MLReadiness
        };
    }
}
