using MLoop.Core.Diagnostics;

namespace MLoop.Core.Tests.Diagnostics;

/// <summary>
/// Tests for PerformanceDiagnostics - model performance analysis.
/// </summary>
public class PerformanceDiagnosticsTests
{
    private readonly PerformanceDiagnostics _diagnostics;

    public PerformanceDiagnosticsTests()
    {
        _diagnostics = new PerformanceDiagnostics();
    }

    [Theory]
    [InlineData(0.95, PerformanceLevel.Excellent)]
    [InlineData(0.85, PerformanceLevel.Good)]
    [InlineData(0.65, PerformanceLevel.Moderate)]
    [InlineData(0.40, PerformanceLevel.Low)]
    [InlineData(0.20, PerformanceLevel.Poor)]
    public void Analyze_Regression_ReturnsCorrectLevel(double rSquared, PerformanceLevel expected)
    {
        // Arrange
        var metrics = new Dictionary<string, double> { ["RSquared"] = rSquared };

        // Act
        var result = _diagnostics.Analyze("Regression", metrics);

        // Assert
        Assert.Equal(expected, result.OverallAssessment);
        Assert.Equal("R²", result.PrimaryMetric);
        Assert.Equal(rSquared, result.PrimaryMetricValue);
    }

    [Theory]
    [InlineData(0.98, PerformanceLevel.Excellent)]
    [InlineData(0.90, PerformanceLevel.Good)]
    [InlineData(0.78, PerformanceLevel.Moderate)]
    [InlineData(0.65, PerformanceLevel.Low)]
    [InlineData(0.52, PerformanceLevel.Poor)]
    public void Analyze_BinaryClassification_ReturnsCorrectLevel(double auc, PerformanceLevel expected)
    {
        // Arrange
        var metrics = new Dictionary<string, double> { ["AreaUnderRocCurve"] = auc };

        // Act
        var result = _diagnostics.Analyze("BinaryClassification", metrics);

        // Assert
        Assert.Equal(expected, result.OverallAssessment);
        Assert.Equal("AUC", result.PrimaryMetric);
    }

    [Theory]
    [InlineData(0.92, PerformanceLevel.Excellent)]
    [InlineData(0.80, PerformanceLevel.Good)]
    [InlineData(0.65, PerformanceLevel.Moderate)]
    [InlineData(0.45, PerformanceLevel.Low)]
    [InlineData(0.30, PerformanceLevel.Poor)]
    public void Analyze_MulticlassClassification_ReturnsCorrectLevel(double accuracy, PerformanceLevel expected)
    {
        // Arrange
        var metrics = new Dictionary<string, double> { ["MacroAccuracy"] = accuracy };

        // Act
        var result = _diagnostics.Analyze("MulticlassClassification", metrics);

        // Assert
        Assert.Equal(expected, result.OverallAssessment);
    }

    [Fact]
    public void Analyze_LowPerformance_ProvidesSuggestions()
    {
        // Arrange
        var metrics = new Dictionary<string, double> { ["RSquared"] = 0.35 };

        // Act
        var result = _diagnostics.Analyze("Regression", metrics);

        // Assert
        Assert.True(result.NeedsAttention);
        Assert.NotEmpty(result.Suggestions);
        Assert.NotEmpty(result.Warnings);
    }

    [Fact]
    public void Analyze_ExcellentPerformance_NoSuggestionsNeeded()
    {
        // Arrange
        var metrics = new Dictionary<string, double> { ["RSquared"] = 0.95 };

        // Act
        var result = _diagnostics.Analyze("Regression", metrics);

        // Assert
        Assert.False(result.NeedsAttention);
        Assert.Equal(PerformanceLevel.Excellent, result.OverallAssessment);
    }

    [Fact]
    public void Analyze_UnknownTaskType_ReturnsUnknownLevel()
    {
        // Arrange
        var metrics = new Dictionary<string, double> { ["SomeMetric"] = 0.85 };

        // Act
        var result = _diagnostics.Analyze("UnknownTask", metrics);

        // Assert
        Assert.Equal(PerformanceLevel.Unknown, result.OverallAssessment);
    }

    [Fact]
    public void Analyze_WithDataCharacteristics_AddsDataBasedSuggestions()
    {
        // Arrange - low samples-to-features ratio
        var metrics = new Dictionary<string, double> { ["RSquared"] = 0.70 };

        // Act
        var result = _diagnostics.Analyze("Regression", metrics, featureCount: 100, rowCount: 50);

        // Assert
        Assert.Contains(result.Warnings, w => w.Contains("samples-to-features ratio"));
        Assert.Contains(result.Warnings, w => w.Contains("Small dataset"));
    }

    [Fact]
    public void Analyze_BinaryClassification_HighAccuracyLowAuc_WarnsImbalance()
    {
        // Arrange - high accuracy but low AUC suggests class imbalance
        var metrics = new Dictionary<string, double>
        {
            ["AreaUnderRocCurve"] = 0.65,
            ["Accuracy"] = 0.95
        };

        // Act
        var result = _diagnostics.Analyze("BinaryClassification", metrics);

        // Assert
        Assert.Contains(result.Warnings, w => w.Contains("class imbalance"));
    }

    [Fact]
    public void Analyze_Regression_CollectsSecondaryMetrics()
    {
        // Arrange
        var metrics = new Dictionary<string, double>
        {
            ["RSquared"] = 0.85,
            ["RootMeanSquaredError"] = 1.5,
            ["MeanAbsoluteError"] = 1.2
        };

        // Act
        var result = _diagnostics.Analyze("Regression", metrics);

        // Assert
        Assert.True(result.SecondaryMetrics.ContainsKey("RMSE"));
        Assert.True(result.SecondaryMetrics.ContainsKey("MAE"));
        Assert.Equal(1.5, result.SecondaryMetrics["RMSE"]);
    }

    [Fact]
    public void Analyze_HighFeatureCount_SuggestsFeatureSelection()
    {
        // Arrange
        var metrics = new Dictionary<string, double> { ["RSquared"] = 0.70 };

        // Act
        var result = _diagnostics.Analyze("Regression", metrics, featureCount: 100, rowCount: 1000);

        // Assert
        Assert.Contains(result.Suggestions, s => s.Contains("feature selection") || s.Contains("feature count"));
    }

    #region Degenerate Model Detection

    [Fact]
    public void Analyze_BinaryHighAccuracyZeroF1_DetectsDegenerate()
    {
        var metrics = new Dictionary<string, double>
        {
            ["accuracy"] = 0.95,
            ["f1_score"] = 0.0,
            ["auc"] = 0.52
        };

        var result = _diagnostics.Analyze("binary-classification", metrics);

        Assert.Equal(PerformanceLevel.Poor, result.OverallAssessment);
        Assert.Contains(result.Warnings, w => w.Contains("majority class"));
    }

    [Fact]
    public void Analyze_MulticlassHighAccuracyZeroF1_DetectsDegenerate()
    {
        var metrics = new Dictionary<string, double>
        {
            ["macro_accuracy"] = 0.80,
            ["micro_accuracy"] = 0.80,
            ["macro_f1"] = 0.0,
            ["log_loss"] = 1.5
        };

        var result = _diagnostics.Analyze("multiclass-classification", metrics);

        Assert.Equal(PerformanceLevel.Poor, result.OverallAssessment);
        Assert.Contains(result.Warnings, w => w.Contains("majority class"));
    }

    [Fact]
    public void Analyze_MulticlassGoodF1_NoDegenerate()
    {
        var metrics = new Dictionary<string, double>
        {
            ["macro_accuracy"] = 0.85,
            ["micro_accuracy"] = 0.87,
            ["macro_f1"] = 0.82,
            ["log_loss"] = 0.5
        };

        var result = _diagnostics.Analyze("multiclass-classification", metrics);

        Assert.NotEqual(PerformanceLevel.Poor, result.OverallAssessment);
        Assert.DoesNotContain(result.Warnings, w => w.Contains("majority class"));
    }

    #endregion

    #region Anomaly Detection

    [Fact]
    public void Analyze_AnomalyDetection_HighAuc_ReturnsExcellent()
    {
        var metrics = new Dictionary<string, double> { ["auc"] = 0.97 };

        var result = _diagnostics.Analyze("anomaly-detection", metrics);

        Assert.Equal(PerformanceLevel.Excellent, result.OverallAssessment);
        Assert.Equal("AUC", result.PrimaryMetric);
    }

    [Fact]
    public void Analyze_AnomalyDetection_LowAuc_ReturnsLow()
    {
        var metrics = new Dictionary<string, double> { ["auc"] = 0.55 };

        var result = _diagnostics.Analyze("anomaly-detection", metrics);

        Assert.Equal(PerformanceLevel.Low, result.OverallAssessment);
        Assert.True(result.Suggestions.Count > 0);
    }

    [Fact]
    public void Analyze_AnomalyDetection_NoAuc_UsesDetectionRate()
    {
        var metrics = new Dictionary<string, double>
        {
            ["detection_rate"] = 0.05,
            ["anomaly_count"] = 50,
            ["total_count"] = 1000
        };

        var result = _diagnostics.Analyze("anomaly-detection", metrics);

        Assert.Equal("Detection Rate", result.PrimaryMetric);
        Assert.Equal(PerformanceLevel.Good, result.OverallAssessment);
    }

    [Fact]
    public void Analyze_AnomalyDetection_HighDetectionRate_WarnsOversensitive()
    {
        var metrics = new Dictionary<string, double>
        {
            ["detection_rate"] = 0.65,
            ["anomaly_count"] = 650,
            ["total_count"] = 1000
        };

        var result = _diagnostics.Analyze("anomaly-detection", metrics);

        Assert.Equal(PerformanceLevel.Low, result.OverallAssessment);
        Assert.Contains(result.Warnings, w => w.Contains("anomalous"));
    }

    [Fact]
    public void Analyze_AnomalyDetection_NoMetrics_ReturnsUnknown()
    {
        var metrics = new Dictionary<string, double>();

        var result = _diagnostics.Analyze("anomaly-detection", metrics);

        Assert.Equal(PerformanceLevel.Unknown, result.OverallAssessment);
    }

    #endregion

    #region Clustering

    [Fact]
    public void Analyze_Clustering_LowDBI_ReturnsExcellent()
    {
        var metrics = new Dictionary<string, double>
        {
            ["davies_bouldin_index"] = 0.3,
            ["average_distance"] = 1.5
        };

        var result = _diagnostics.Analyze("clustering", metrics);

        Assert.Equal(PerformanceLevel.Excellent, result.OverallAssessment);
        Assert.Equal("Davies-Bouldin Index", result.PrimaryMetric);
    }

    [Fact]
    public void Analyze_Clustering_ModerateDBI_ReturnsModerate()
    {
        var metrics = new Dictionary<string, double>
        {
            ["davies_bouldin_index"] = 1.5,
            ["average_distance"] = 5.0
        };

        var result = _diagnostics.Analyze("clustering", metrics);

        Assert.Equal(PerformanceLevel.Moderate, result.OverallAssessment);
        Assert.True(result.Suggestions.Count > 0);
    }

    [Fact]
    public void Analyze_Clustering_HighDBI_ReturnsLow()
    {
        var metrics = new Dictionary<string, double>
        {
            ["davies_bouldin_index"] = 3.0,
            ["average_distance"] = 10.0
        };

        var result = _diagnostics.Analyze("clustering", metrics);

        Assert.Equal(PerformanceLevel.Low, result.OverallAssessment);
        Assert.True(result.Suggestions.Count >= 2);
    }

    [Fact]
    public void Analyze_Clustering_NoDBI_UsesAverageDistance()
    {
        var metrics = new Dictionary<string, double>
        {
            ["average_distance"] = 2.0
        };

        var result = _diagnostics.Analyze("clustering", metrics);

        Assert.Equal("Average Distance", result.PrimaryMetric);
        Assert.Equal(PerformanceLevel.Moderate, result.OverallAssessment);
    }

    [Fact]
    public void Analyze_Clustering_NoMetrics_ReturnsUnknown()
    {
        var metrics = new Dictionary<string, double>();

        var result = _diagnostics.Analyze("clustering", metrics);

        Assert.Equal(PerformanceLevel.Unknown, result.OverallAssessment);
    }

    [Fact]
    public void Analyze_Clustering_ImbalancedClusters_WarnsImbalance()
    {
        var metrics = new Dictionary<string, double>
        {
            ["davies_bouldin_index"] = 0.8,
            ["average_distance"] = 2.0,
            ["largest_cluster_ratio"] = 0.9,
            ["cluster_count"] = 3
        };

        var result = _diagnostics.Analyze("clustering", metrics);

        Assert.Equal(PerformanceLevel.Good, result.OverallAssessment);
        Assert.Contains(result.Warnings, w => w.Contains("imbalance"));
    }

    [Fact]
    public void Analyze_Clustering_GoodDBI_ReturnsGood()
    {
        var metrics = new Dictionary<string, double>
        {
            ["davies_bouldin_index"] = 0.7,
            ["average_distance"] = 3.0,
            ["normalized_mutual_information"] = 0.85,
            ["num_clusters"] = 5
        };

        var result = _diagnostics.Analyze("clustering", metrics);

        Assert.Equal(PerformanceLevel.Good, result.OverallAssessment);
        Assert.True(result.SecondaryMetrics.ContainsKey("NMI"));
        Assert.True(result.SecondaryMetrics.ContainsKey("Number of Clusters"));
    }

    #endregion
}
