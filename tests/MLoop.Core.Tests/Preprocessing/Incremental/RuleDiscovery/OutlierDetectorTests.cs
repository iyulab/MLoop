using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.RuleDiscovery;

public sealed class OutlierDetectorTests
{
    private readonly OutlierDetector _detector = new();

    [Fact]
    public async Task DetectAsync_NoOutliers_ReturnsEmpty()
    {
        // Arrange: Normal distribution, no outliers
        var column = new DoubleDataFrameColumn("test", new[] { 10.0, 11.0, 10.5, 10.2, 10.8 });

        // Act
        var patterns = await _detector.DetectAsync(column, "test");

        // Assert
        Assert.Empty(patterns);
    }

    [Fact]
    public async Task DetectAsync_WithOutliers_DetectsPattern()
    {
        // Arrange: Clear outliers (Z-score > 3)
        var values = Enumerable.Range(1, 100).Select(i => (double)i).ToArray();
        values[0] = 1000.0; // Clear outlier
        values[99] = -1000.0; // Clear outlier

        var column = new DoubleDataFrameColumn("test", values);

        // Act
        var patterns = await _detector.DetectAsync(column, "test");

        // Assert
        Assert.Single(patterns);
        var pattern = patterns[0];
        Assert.Equal(PatternType.OutlierAnomaly, pattern.Type);
        Assert.True(pattern.Occurrences >= 2);
        Assert.True(pattern.Confidence >= 0.85);
    }

    [Fact]
    public async Task DetectAsync_StringColumn_ReturnsEmpty()
    {
        // Arrange
        var column = new StringDataFrameColumn("test", new[] { "a", "b", "c" });

        // Act
        var patterns = await _detector.DetectAsync(column, "test");

        // Assert
        Assert.Empty(patterns);
    }

    [Fact]
    public async Task DetectAsync_TooFewOutliers_ReturnsEmpty()
    {
        // Arrange: <1% outliers = ignored
        var values = Enumerable.Range(1, 1000).Select(i => (double)i).ToArray();
        values[0] = 10000.0; // Single outlier = 0.1%

        var column = new DoubleDataFrameColumn("test", values);

        // Act
        var patterns = await _detector.DetectAsync(column, "test");

        // Assert
        Assert.Empty(patterns); // Too few outliers to report
    }

    [Fact]
    public void IsApplicable_NumericColumn_ReturnsTrue()
    {
        var column = new Int32DataFrameColumn("test", new[] { 1, 2, 3 });
        Assert.True(_detector.IsApplicable(column));
    }

    [Fact]
    public void IsApplicable_StringColumn_ReturnsFalse()
    {
        var column = new StringDataFrameColumn("test", new[] { "a", "b" });
        Assert.False(_detector.IsApplicable(column));
    }

    [Fact]
    public void PatternType_ReturnsExpected()
    {
        Assert.Equal(PatternType.OutlierAnomaly, _detector.PatternType);
    }
}
