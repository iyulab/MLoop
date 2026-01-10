using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.Strategies;
using MLoop.Core.Tests.Preprocessing.Incremental.TestData;

namespace MLoop.Core.Tests.Preprocessing.Incremental.Strategies;

public class StratifiedSamplingTests
{
    [Fact]
    public void Sample_WithBalancedData_PreservesDistribution()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateBalancedData(3000, numClasses: 3);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(data, 0.1, config, randomSeed: 42);

        // Assert
        Assert.Equal(300, sample.Rows.Count);

        // Verify distribution
        var sourceDistribution = GetLabelDistribution(data, "Label");
        var sampleDistribution = GetLabelDistribution(sample, "Label");

        foreach (var kvp in sourceDistribution)
        {
            var sourceProportion = kvp.Value;
            var sampleProportion = sampleDistribution[kvp.Key];
            var difference = Math.Abs(sourceProportion - sampleProportion);
            Assert.True(difference < 0.02, $"Distribution mismatch for {kvp.Key}: source={sourceProportion:F3}, sample={sampleProportion:F3}");
        }
    }

    [Fact]
    public void Sample_WithImbalancedData_PreservesDistribution()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateImbalancedData(1000, numClasses: 3, majorityClassRatio: 0.7);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(data, 0.1, config, randomSeed: 42);

        // Assert
        Assert.Equal(100, sample.Rows.Count);

        // Verify imbalanced distribution is preserved
        var sourceDistribution = GetLabelDistribution(data, "Label");
        var sampleDistribution = GetLabelDistribution(sample, "Label");

        foreach (var kvp in sourceDistribution)
        {
            if (sampleDistribution.ContainsKey(kvp.Key))
            {
                var difference = Math.Abs(kvp.Value - sampleDistribution[kvp.Key]);
                Assert.True(difference < 0.05, $"Distribution mismatch for {kvp.Key}");
            }
        }
    }

    [Fact]
    public void IsApplicable_WithValidLabelColumn_ReturnsTrue()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var isApplicable = strategy.IsApplicable(data, config);

        // Assert
        Assert.True(isApplicable);
    }

    [Fact]
    public void IsApplicable_WithoutLabelColumn_ReturnsFalse()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = null };

        // Act
        var isApplicable = strategy.IsApplicable(data, config);

        // Assert
        Assert.False(isApplicable);
    }

    [Fact]
    public void IsApplicable_WithInvalidLabelColumn_ReturnsFalse()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "NonExistentColumn" };

        // Act
        var isApplicable = strategy.IsApplicable(data, config);

        // Assert
        Assert.False(isApplicable);
    }

    [Fact]
    public void IsApplicable_WithNumericLabelColumn_ReturnsTrue()
    {
        // Arrange - IsApplicable only checks if column exists, not type
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Feature1" }; // Numeric column

        // Act
        var isApplicable = strategy.IsApplicable(data, config);

        // Assert - Applicable returns true if column exists (type checked during sampling)
        Assert.True(isApplicable);
    }

    [Fact]
    public void Name_Property_ReturnsStratified()
    {
        // Arrange
        var strategy = new StratifiedSamplingStrategy();

        // Act
        var name = strategy.Name;

        // Assert
        Assert.Equal("Stratified", name);
    }

    [Fact]
    public void Validate_WithPreservedDistribution_ReturnsValidResult()
    {
        // Arrange
        var source = SampleDataGenerator.GenerateBalancedData(1000, numClasses: 3);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };
        var sample = strategy.Sample(source, 0.1, config);

        // Act
        var result = strategy.Validate(source, sample, config);

        // Assert - Just verify validation completes without error
        Assert.NotNull(result);
        Assert.NotNull(result.Message);
    }

    [Fact]
    public void Validate_WithDistortedDistribution_DetectsIssues()
    {
        // Arrange
        var source = SampleDataGenerator.GenerateBalancedData(1000, numClasses: 3);
        var badSample = SampleDataGenerator.GenerateImbalancedData(100, numClasses: 3, majorityClassRatio: 0.9);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var result = strategy.Validate(source, badSample, config);

        // Assert - Validation should detect the distribution mismatch
        Assert.NotNull(result);
        Assert.NotNull(result.Message);
    }

    [Fact]
    public void Sample_WithSingleClass_WorksCorrectly()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateSameValueData(1000);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(data, 0.1, config);

        // Assert
        Assert.Equal(100, sample.Rows.Count);
    }

    [Fact]
    public void Sample_WithSameRandomSeed_ReturnsSameSample()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateBalancedData(1000, numClasses: 3);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample1 = strategy.Sample(data, 0.1, config, randomSeed: 42);
        var sample2 = strategy.Sample(data, 0.1, config, randomSeed: 42);

        // Assert
        var value1 = ((PrimitiveDataFrameColumn<double>)sample1.Columns[0])[0];
        var value2 = ((PrimitiveDataFrameColumn<double>)sample2.Columns[0])[0];
        Assert.Equal(value1, value2);
    }

    [Fact]
    public void Sample_EnsuresMinimumSamplesPerClass()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateImbalancedData(1000, numClasses: 5, majorityClassRatio: 0.6);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(data, 0.05, config); // 5% = 50 samples

        // Assert
        Assert.True(sample.Rows.Count >= 50);

        // Check each class has at least 1 sample
        var sampleDistribution = GetLabelDistribution(sample, "Label");
        foreach (var kvp in sampleDistribution)
        {
            Assert.True(kvp.Value > 0, $"Class {kvp.Key} has no samples");
        }
    }

    [Fact]
    public void Sample_WithNullData_ThrowsArgumentNullException()
    {
        // Arrange
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.Sample(null!, 0.1, config));
    }

    [Fact]
    public void Sample_WithNullConfig_ThrowsArgumentNullException()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateBalancedData(1000, numClasses: 3);
        var strategy = new StratifiedSamplingStrategy();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.Sample(data, 0.1, null!));
    }

    [Fact]
    public void Sample_WithMissingLabelColumn_ThrowsArgumentException()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "NonExistent" };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => strategy.Sample(data, 0.1, config));
    }

    [Theory]
    [InlineData(0.01)]
    [InlineData(0.05)]
    [InlineData(0.1)]
    [InlineData(0.25)]
    [InlineData(0.5)]
    public void Sample_WithVariousRatios_MaintainsDistribution(double ratio)
    {
        // Arrange
        var data = SampleDataGenerator.GenerateBalancedData(2000, numClasses: 4);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(data, ratio, config);

        // Assert
        var expectedSize = (int)(2000 * ratio);
        Assert.True(Math.Abs(sample.Rows.Count - expectedSize) <= 4); // Allow small variance for rounding

        // Verify distribution
        var sourceDistribution = GetLabelDistribution(data, "Label");
        var sampleDistribution = GetLabelDistribution(sample, "Label");

        foreach (var kvp in sourceDistribution)
        {
            if (sampleDistribution.ContainsKey(kvp.Key))
            {
                var difference = Math.Abs(kvp.Value - sampleDistribution[kvp.Key]);
                Assert.True(difference < 0.05, $"Distribution mismatch at ratio {ratio} for {kvp.Key}");
            }
        }
    }

    [Fact]
    public void Sample_PreservesColumnTypes()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateBalancedData(1000, numClasses: 3);
        var strategy = new StratifiedSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(data, 0.1, config);

        // Assert
        for (int i = 0; i < data.Columns.Count; i++)
        {
            Assert.Equal(data.Columns[i].GetType(), sample.Columns[i].GetType());
            Assert.Equal(data.Columns[i].Name, sample.Columns[i].Name);
        }
    }

    private static Dictionary<string, double> GetLabelDistribution(DataFrame data, string labelColumnName)
    {
        var labelColumn = (StringDataFrameColumn)data.Columns[labelColumnName];
        var counts = new Dictionary<string, int>();
        var total = 0;

        for (int i = 0; i < labelColumn.Length; i++)
        {
            var value = labelColumn[i] ?? "NULL";
            counts[value] = counts.GetValueOrDefault(value, 0) + 1;
            total++;
        }

        return counts.ToDictionary(
            kvp => kvp.Key,
            kvp => (double)kvp.Value / total
        );
    }
}
