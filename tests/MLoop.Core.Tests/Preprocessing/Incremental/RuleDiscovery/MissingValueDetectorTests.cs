using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.RuleDiscovery;

public sealed class MissingValueDetectorTests
{
    private readonly MissingValueDetector _detector = new();

    [Fact]
    public async Task DetectAsync_NoMissingValues_ReturnsEmpty()
    {
        // Arrange
        var column = new StringDataFrameColumn("test", new[] { "value1", "value2", "value3" });

        // Act
        var patterns = await _detector.DetectAsync(column, "test");

        // Assert
        Assert.Empty(patterns);
    }

    [Fact]
    public async Task DetectAsync_WithNullValues_DetectsPattern()
    {
        // Arrange
        var column = new StringDataFrameColumn("test", new string?[] { "value1", null, "value2", null, "value3" });

        // Act
        var patterns = await _detector.DetectAsync(column, "test");

        // Assert
        Assert.Single(patterns);
        var pattern = patterns[0];
        Assert.Equal(PatternType.MissingValue, pattern.Type);
        Assert.Equal(2, pattern.Occurrences);
        Assert.Equal(5, pattern.TotalRows);
        Assert.Equal(1.0, pattern.Confidence);
    }

    [Fact]
    public async Task DetectAsync_WithNAValues_DetectsPattern()
    {
        // Arrange
        var column = new StringDataFrameColumn("test", new[] { "value1", "N/A", "value2", "NULL", "value3" });

        // Act
        var patterns = await _detector.DetectAsync(column, "test");

        // Assert
        Assert.Single(patterns);
        var pattern = patterns[0];
        Assert.Equal(2, pattern.Occurrences);
        Assert.NotNull(pattern.Examples);
        Assert.Contains("N/A", pattern.Examples);
    }

    [Fact]
    public async Task DetectAsync_HighMissingRate_SetsCriticalSeverity()
    {
        // Arrange: >50% missing = Critical
        var values = new string?[100];
        for (int i = 0; i < 60; i++) values[i] = null;
        for (int i = 60; i < 100; i++) values[i] = $"value{i}";

        var column = new StringDataFrameColumn("test", values);

        // Act
        var patterns = await _detector.DetectAsync(column, "test");

        // Assert
        Assert.Single(patterns);
        Assert.Equal(Severity.Critical, patterns[0].Severity);
    }

    [Fact]
    public void IsApplicable_AnyColumnType_ReturnsTrue()
    {
        // Arrange
        var stringColumn = new StringDataFrameColumn("test", new[] { "a", "b" });
        var intColumn = new Int32DataFrameColumn("test", new[] { 1, 2 });

        // Act & Assert
        Assert.True(_detector.IsApplicable(stringColumn));
        Assert.True(_detector.IsApplicable(intColumn));
    }

    [Fact]
    public void PatternType_ReturnsExpected()
    {
        Assert.Equal(PatternType.MissingValue, _detector.PatternType);
    }
}
