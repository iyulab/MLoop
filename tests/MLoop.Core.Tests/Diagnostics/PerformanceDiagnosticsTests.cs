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
}
