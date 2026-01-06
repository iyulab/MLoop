using Microsoft.Data.Analysis;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.Core.Preprocessing.Incremental;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Tests.Preprocessing.Incremental.TestData;

namespace MLoop.Core.Tests.Preprocessing.Incremental;

public class SampleAnalyzerTests
{
    [Fact]
    public async Task AnalyzeAsync_WithValidData_ReturnsAnalysis()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);

        // Act
        var analysis = await analyzer.AnalyzeAsync(data, stageNumber: 1);

        // Assert
        Assert.NotNull(analysis);
        Assert.Equal(1, analysis.StageNumber);
        Assert.Equal(1000, analysis.RowCount);
        Assert.Equal(5, analysis.ColumnCount); // Feature1, Feature2, Feature3, Category, Label
        Assert.NotEmpty(analysis.Columns);
    }

    [Fact]
    public async Task AnalyzeAsync_WithNullData_ThrowsArgumentNullException()
    {
        // Arrange
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(
            async () => await analyzer.AnalyzeAsync(null!, stageNumber: 1)
        );
    }

    [Fact]
    public void AnalyzeColumn_WithNumericColumn_ReturnsNumericStats()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);
        var numericColumn = data.Columns["Feature1"];

        // Act
        var analysis = analyzer.AnalyzeColumn(numericColumn, "Feature1", 0);

        // Assert
        Assert.NotNull(analysis);
        Assert.Equal("Feature1", analysis.ColumnName);
        Assert.Equal(0, analysis.ColumnIndex);
        Assert.Equal(DataType.Double, analysis.DataType);
        Assert.NotNull(analysis.NumericStats);
        Assert.Null(analysis.CategoricalStats);
        Assert.True(analysis.NumericStats.Mean >= 0 && analysis.NumericStats.Mean <= 100);
        Assert.True(analysis.NumericStats.StandardDeviation > 0);
    }

    [Fact]
    public void AnalyzeColumn_WithCategoricalColumn_ReturnsCategoricalStats()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);
        var categoricalColumn = data.Columns["Category"];

        // Act
        var analysis = analyzer.AnalyzeColumn(categoricalColumn, "Category", 0);

        // Assert
        Assert.NotNull(analysis);
        Assert.Equal("Category", analysis.ColumnName);
        Assert.Equal(DataType.String, analysis.DataType);
        Assert.Null(analysis.NumericStats);
        Assert.NotNull(analysis.CategoricalStats);
        Assert.Equal(5, analysis.CategoricalStats.UniqueCount); // A, B, C, D, E
        Assert.True(analysis.CategoricalStats.Entropy > 0);
        Assert.NotEmpty(analysis.CategoricalStats.TopValues);
    }

    [Fact]
    public void AnalyzeColumn_WithMissingValues_DetectsMissing()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateDataWithMissing(1000, missingPercentage: 0.3);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);
        var columnWithMissing = data.Columns["Feature1"];

        // Act
        var analysis = analyzer.AnalyzeColumn(columnWithMissing, "Feature1", 0);

        // Assert
        Assert.True(analysis.MissingPercentage > 25 && analysis.MissingPercentage < 35);
        Assert.True(analysis.NullCount > 0);
        Assert.NotEmpty(analysis.QualityIssues);
        Assert.Contains(analysis.QualityIssues, issue => issue.IssueType == "ModerateMissingValues");
    }

    [Fact]
    public void AnalyzeColumn_WithOutliers_DetectsOutliers()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateDataWithOutliers(1000, outlierPercentage: 0.1);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);
        var columnWithOutliers = data.Columns["Feature1"];

        // Act
        var analysis = analyzer.AnalyzeColumn(columnWithOutliers, "Feature1", 0);

        // Assert
        Assert.NotNull(analysis.NumericStats);
        Assert.True(analysis.NumericStats.OutlierCount > 0);
        Assert.True(analysis.NumericStats.OutlierPercentage > 5);
        Assert.Contains(analysis.QualityIssues, issue => issue.IssueType == "HighOutliers");
    }

    [Fact]
    public void AnalyzeColumn_WithHighCardinality_DetectsHighCardinality()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateHighCardinalityData(1000, uniqueValues: 500);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);
        var highCardColumn = data.Columns["HighCardinalityColumn"];

        // Act
        var analysis = analyzer.AnalyzeColumn(highCardColumn, "HighCardinalityColumn", 0);

        // Assert
        Assert.NotNull(analysis.CategoricalStats);
        Assert.True(analysis.CategoricalStats.IsHighCardinality);
        Assert.Contains(analysis.QualityIssues, issue => issue.IssueType == "HighCardinality");
        Assert.Contains(analysis.RecommendedActions, action =>
            action.Contains("target encoding") || action.Contains("embedding"));
    }

    [Fact]
    public void AnalyzeColumn_WithNullColumn_ThrowsArgumentNullException()
    {
        // Arrange
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(
            () => analyzer.AnalyzeColumn(null!, "Test", 0)
        );
    }

    [Fact]
    public void HasConverged_WithSimilarAnalyses_ReturnsTrue()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000, randomSeed: 42);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);

        var analysis1 = analyzer.AnalyzeColumn(data.Columns["Feature1"], "Feature1", 0);
        var analysis2 = analyzer.AnalyzeColumn(data.Columns["Feature1"], "Feature1", 0);

        var sampleAnalysis1 = new SampleAnalysis
        {
            StageNumber = 1,
            SampleRatio = 0.1,
            Timestamp = DateTime.UtcNow,
            RowCount = 1000,
            ColumnCount = 1,
            Columns = new List<ColumnAnalysis> { analysis1 },
            QualityScore = 0.8,
            EstimatedMemoryBytes = 8000
        };

        var sampleAnalysis2 = new SampleAnalysis
        {
            StageNumber = 2,
            SampleRatio = 0.2,
            Timestamp = DateTime.UtcNow,
            RowCount = 1000,
            ColumnCount = 1,
            Columns = new List<ColumnAnalysis> { analysis2 },
            QualityScore = 0.8,
            EstimatedMemoryBytes = 8000
        };

        // Act
        var hasConverged = analyzer.HasConverged(sampleAnalysis1, sampleAnalysis2, threshold: 0.01);

        // Assert
        Assert.True(hasConverged);
    }

    [Fact]
    public void HasConverged_WithDifferentAnalyses_ReturnsFalse()
    {
        // Arrange
        var data1 = SampleDataGenerator.GenerateMixedData(1000, randomSeed: 42);
        var data2 = SampleDataGenerator.GenerateMixedData(1000, randomSeed: 123);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);

        var analysis1 = analyzer.AnalyzeColumn(data1.Columns["Feature1"], "Feature1", 0);
        var analysis2 = analyzer.AnalyzeColumn(data2.Columns["Feature1"], "Feature1", 0);

        var sampleAnalysis1 = new SampleAnalysis
        {
            StageNumber = 1,
            SampleRatio = 0.1,
            Timestamp = DateTime.UtcNow,
            RowCount = 1000,
            ColumnCount = 1,
            Columns = new List<ColumnAnalysis> { analysis1 },
            QualityScore = 0.8,
            EstimatedMemoryBytes = 8000
        };

        var sampleAnalysis2 = new SampleAnalysis
        {
            StageNumber = 2,
            SampleRatio = 0.2,
            Timestamp = DateTime.UtcNow,
            RowCount = 1000,
            ColumnCount = 1,
            Columns = new List<ColumnAnalysis> { analysis2 },
            QualityScore = 0.8,
            EstimatedMemoryBytes = 8000
        };

        // Act
        var hasConverged = analyzer.HasConverged(sampleAnalysis1, sampleAnalysis2, threshold: 0.001);

        // Assert
        Assert.False(hasConverged);
    }

    [Fact]
    public void HasConverged_WithNullPrevious_ThrowsArgumentNullException()
    {
        // Arrange
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);
        var current = new SampleAnalysis
        {
            StageNumber = 1,
            SampleRatio = 0.1,
            Timestamp = DateTime.UtcNow,
            RowCount = 100,
            ColumnCount = 1,
            Columns = new List<ColumnAnalysis>(),
            QualityScore = 0.8,
            EstimatedMemoryBytes = 800
        };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => analyzer.HasConverged(null!, current));
    }

    [Fact]
    public async Task AnalyzeAsync_WithCancellation_ThrowsOperationCanceledException()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(10000);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);
        var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            async () => await analyzer.AnalyzeAsync(data, 1, cancellationToken: cts.Token)
        );
    }

    [Fact]
    public async Task AnalyzeAsync_CalculatesQualityScore()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);

        // Act
        var analysis = await analyzer.AnalyzeAsync(data, stageNumber: 1);

        // Assert
        Assert.True(analysis.QualityScore >= 0.0 && analysis.QualityScore <= 1.0);
        Assert.True(analysis.QualityScore > 0.5); // Clean data should have good quality
    }

    [Fact]
    public async Task AnalyzeAsync_WithHighMissingValues_ReturnsLowerQualityScore()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateDataWithMissing(1000, missingPercentage: 0.6);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);

        // Act
        var analysis = await analyzer.AnalyzeAsync(data, stageNumber: 1);

        // Assert
        Assert.True(analysis.QualityScore < 0.6); // High missing should reduce quality
        var allIssues = analysis.AllQualityIssues;
        Assert.Contains(allIssues, issue => issue.IssueType == "HighMissingValues");
    }

    [Fact]
    public async Task AnalyzeAsync_EstimatesMemoryFootprint()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);

        // Act
        var analysis = await analyzer.AnalyzeAsync(data, stageNumber: 1);

        // Assert
        Assert.True(analysis.EstimatedMemoryBytes > 0);
        // With 5 columns (3 numeric, 2 string) and 1000 rows, should be reasonable
        Assert.True(analysis.EstimatedMemoryBytes > 1000 && analysis.EstimatedMemoryBytes < 1000000);
    }

    [Fact]
    public async Task AnalyzeAsync_GeneratesRecommendations()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateDataWithMissing(1000, missingPercentage: 0.3);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);

        // Act
        var analysis = await analyzer.AnalyzeAsync(data, stageNumber: 1);

        // Assert
        var allRecommendations = analysis.Columns.SelectMany(c => c.RecommendedActions).ToList();
        Assert.NotEmpty(allRecommendations);
        Assert.Contains(allRecommendations, rec =>
            rec.Contains("Fill missing") || rec.Contains("median") || rec.Contains("mode"));
    }

    [Fact]
    public void AnalyzeColumn_WithEmptyColumn_ReturnsEmptyStats()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateEmptyData();
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);
        var emptyColumn = data.Columns["Feature1"];

        // Act
        var analysis = analyzer.AnalyzeColumn(emptyColumn, "Feature1", 0);

        // Assert
        Assert.Equal(0, analysis.TotalRows);
        Assert.Equal(0, analysis.NonNullCount);
    }

    [Fact]
    public void AnalyzeColumn_WithSameValues_DetectsPattern()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateSameValueData(100);
        var analyzer = new SampleAnalyzer(NullLogger<SampleAnalyzer>.Instance);
        var sameValueColumn = data.Columns["Feature1"];

        // Act
        var analysis = analyzer.AnalyzeColumn(sameValueColumn, "Feature1", 0);

        // Assert
        Assert.NotNull(analysis.NumericStats);
        Assert.Equal(0, analysis.NumericStats.StandardDeviation); // All same = no variance
        Assert.Equal(analysis.NumericStats.Mean, analysis.NumericStats.Min);
        Assert.Equal(analysis.NumericStats.Mean, analysis.NumericStats.Max);
    }
}
