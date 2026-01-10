using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.RuleDiscovery;

public sealed class ConfidenceCalculatorTests
{
    private readonly ConfidenceCalculator _calculator = new();

    [Fact]
    public async Task CalculateAsync_IdenticalSamples_HighStability()
    {
        // Arrange
        var rule = CreateTestRule("TestColumn", PatternType.MissingValue);
        var sample = CreateSampleWithMissingValues(100, 0.2);

        // Act
        var score = await _calculator.CalculateAsync(rule, sample, sample);

        // Assert: Identical samples should have perfect stability
        Assert.True(score.Stability >= 0.99, $"Expected stability >= 0.99, got {score.Stability}");
        Assert.True(score.Consistency >= 0.80, $"Expected consistency >= 0.80, got {score.Consistency}");
        Assert.True(score.Overall >= 0.60, $"Expected overall >= 0.60, got {score.Overall}");
    }

    [Fact]
    public async Task CalculateAsync_HighCoverage_ScoresCorrectly()
    {
        // Arrange: 80% of rows have missing values
        var rule = CreateTestRule("TestColumn", PatternType.MissingValue);
        var sample = CreateSampleWithMissingValues(100, 0.8);

        // Act
        var score = await _calculator.CalculateAsync(rule, sample, sample);

        // Assert
        Assert.True(score.Coverage >= 0.75);
        Assert.True(score.Overall >= 0.70);
    }

    [Fact]
    public async Task CalculateAsync_LowCoverage_ScoresCorrectly()
    {
        // Arrange: Only 5% missing values
        var rule = CreateTestRule("TestColumn", PatternType.MissingValue);
        var sample = CreateSampleWithMissingValues(100, 0.05);

        // Act
        var score = await _calculator.CalculateAsync(rule, sample, sample);

        // Assert
        Assert.True(score.Coverage < 0.10);
    }

    [Fact]
    public async Task CalculateAsync_DifferentSamples_LowerStability()
    {
        // Arrange
        var rule = CreateTestRule("TestColumn", PatternType.MissingValue);
        var sample1 = CreateSampleWithMissingValues(100, 0.2);
        var sample2 = CreateSampleWithMissingValues(100, 0.5);

        // Act
        var score = await _calculator.CalculateAsync(rule, sample1, sample2);

        // Assert
        Assert.True(score.Stability < 1.0);
    }

    [Fact]
    public async Task CalculateAsync_WeightedFormula_CalculatesCorrectly()
    {
        // Arrange
        var rule = CreateTestRule("TestColumn", PatternType.MissingValue);
        var sample = CreateSampleWithMissingValues(100, 0.5);

        // Act
        var score = await _calculator.CalculateAsync(rule, sample, sample);

        // Assert: Overall = Consistency * 0.5 + Coverage * 0.3 + Stability * 0.2
        var expected = (score.Consistency * 0.5) + (score.Coverage * 0.3) + (score.Stability * 0.2);
        Assert.Equal(expected, score.Overall, precision: 3);
    }

    [Fact]
    public async Task CalculateAsync_ColumnNotExists_ReturnsLowScore()
    {
        // Arrange
        var rule = CreateTestRule("NonExistentColumn", PatternType.MissingValue);
        var sample = CreateSampleWithMissingValues(100, 0.5, "TestColumn");

        // Act
        var score = await _calculator.CalculateAsync(rule, sample, sample);

        // Assert
        Assert.Equal(0.0, score.Coverage);
        Assert.Equal(0.0, score.Consistency);
    }

    private static PreprocessingRule CreateTestRule(string columnName, PatternType patternType)
    {
        return new PreprocessingRule
        {
            Id = "test-rule",
            Type = PreprocessingRuleType.MissingValueStrategy,
            ColumnNames = new[] { columnName },
            Description = "Test rule",
            PatternType = patternType,
            RequiresHITL = false,
            Priority = 5,
            DiscoveredInStage = 1
        };
    }

    private static DataFrame CreateSampleWithMissingValues(
        int rows,
        double missingRatio,
        string columnName = "TestColumn")
    {
        var values = new string?[rows];
        var missingCount = (int)(rows * missingRatio);

        for (int i = 0; i < missingCount; i++)
        {
            values[i] = null;
        }

        for (int i = missingCount; i < rows; i++)
        {
            values[i] = $"value{i}";
        }

        var df = new DataFrame();
        df.Columns.Add(new StringDataFrameColumn(columnName, values));
        return df;
    }
}
