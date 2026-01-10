using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.Strategies;
using MLoop.Core.Tests.Preprocessing.Incremental.TestData;

namespace MLoop.Core.Tests.Preprocessing.Incremental.Strategies;

public class AdaptiveSamplingTests
{
    [Fact]
    public void Sample_WithGoodStratificationConditions_UsesStratifiedStrategy()
    {
        // Arrange: Balanced data with reasonable number of classes
        var data = SampleDataGenerator.GenerateBalancedData(3000, numClasses: 5);
        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(data, 0.1, config, randomSeed: 42);

        // Assert
        Assert.Equal(300, sample.Rows.Count);

        // Verify distribution is preserved (indicates stratified was used)
        var sourceDistribution = GetLabelDistribution(data, "Label");
        var sampleDistribution = GetLabelDistribution(sample, "Label");

        var maxDifference = sourceDistribution.Max(kvp =>
            Math.Abs(kvp.Value - sampleDistribution.GetValueOrDefault(kvp.Key, 0.0)));

        Assert.True(maxDifference < 0.05, $"Stratified strategy should preserve distribution: max diff = {maxDifference}");
    }

    [Fact]
    public void Sample_WithTooManyClasses_UsesRandomStrategy()
    {
        // Arrange: Data with too many classes (>100)
        var data = SampleDataGenerator.GenerateHighCardinalityData(5000, uniqueValues: 150);
        var dataWithLabel = new DataFrame();
        dataWithLabel.Columns.Add(data.Columns[0]);

        // Add balanced label column with many classes
        var labels = new StringDataFrameColumn("Label",
            Enumerable.Range(0, 5000).Select(i => $"Class{i % 150}").ToArray());
        dataWithLabel.Columns.Add(labels);

        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(dataWithLabel, 0.1, config);

        // Assert
        Assert.Equal(500, sample.Rows.Count);
        // Random strategy doesn't preserve exact distribution
    }

    [Fact]
    public void Sample_WithNoLabelColumn_UsesRandomStrategy()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = null };

        // Act
        var sample = strategy.Sample(data, 0.1, config);

        // Assert
        Assert.Equal(100, sample.Rows.Count);
    }

    [Fact]
    public void Sample_WithSingleClass_UsesRandomStrategy()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateSameValueData(1000);
        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(data, 0.1, config);

        // Assert
        Assert.Equal(100, sample.Rows.Count);
    }

    [Fact]
    public void IsApplicable_WithAnyData_ReturnsTrue()
    {
        // Arrange: Adaptive is always applicable as it falls back to random
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration();

        // Act
        var isApplicable = strategy.IsApplicable(data, config);

        // Assert
        Assert.True(isApplicable);
    }

    [Fact]
    public void Name_Property_ReturnsAdaptive()
    {
        // Arrange
        var strategy = new AdaptiveSamplingStrategy();

        // Act
        var name = strategy.Name;

        // Assert
        Assert.Equal("Adaptive", name);
    }

    [Fact]
    public void Validate_IncludesAdaptiveInformation()
    {
        // Arrange
        var source = SampleDataGenerator.GenerateBalancedData(1000, numClasses: 3);
        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };
        var sample = strategy.Sample(source, 0.1, config);

        // Act
        var result = strategy.Validate(source, sample, config);

        // Assert
        Assert.Contains("Adaptive", result.Message);
        Assert.True(result.Details.ContainsKey("SelectedStrategy"));
        Assert.True(result.Details.ContainsKey("AdaptiveReason"));
    }

    [Fact]
    public void Sample_WithHighCardinalityLabel_UsesRandomStrategy()
    {
        // Arrange: High cardinality (>50% unique)
        var data = new DataFrame();
        var numericColumn = new PrimitiveDataFrameColumn<double>("Feature1",
            Enumerable.Range(0, 1000).Select(i => (double)i).ToArray());
        var labelColumn = new StringDataFrameColumn("Label",
            Enumerable.Range(0, 1000).Select(i => $"Class{i}").ToArray()); // Each row unique

        data.Columns.Add(numericColumn);
        data.Columns.Add(labelColumn);

        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(data, 0.1, config);

        // Assert
        Assert.Equal(100, sample.Rows.Count);
    }

    [Fact]
    public void Sample_WithInsufficientSamplesPerClass_UsesRandomStrategy()
    {
        // Arrange: Too few samples per class (<5)
        var data = SampleDataGenerator.GenerateBalancedData(100, numClasses: 30); // 100/30 = ~3.3 per class
        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(data, 0.5, config);

        // Assert
        Assert.Equal(50, sample.Rows.Count);
    }

    [Fact]
    public void Sample_WithNullData_ThrowsArgumentNullException()
    {
        // Arrange
        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.Sample(null!, 0.1, config));
    }

    [Fact]
    public void Sample_WithNullConfig_ThrowsArgumentNullException()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var strategy = new AdaptiveSamplingStrategy();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => strategy.Sample(data, 0.1, null!));
    }

    [Fact]
    public void Sample_WithSameRandomSeed_ReturnsSameSample()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample1 = strategy.Sample(data, 0.1, config, randomSeed: 42);
        var sample2 = strategy.Sample(data, 0.1, config, randomSeed: 42);

        // Assert
        var value1 = ((PrimitiveDataFrameColumn<double>)sample1.Columns[0])[0];
        var value2 = ((PrimitiveDataFrameColumn<double>)sample2.Columns[0])[0];
        Assert.Equal(value1, value2);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(50)]
    [InlineData(100)]
    public void Sample_SelectsStrategyBasedOnClassCount(int numClasses)
    {
        // Arrange
        var data = SampleDataGenerator.GenerateBalancedData(5000, numClasses: numClasses);
        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        // Act
        var sample = strategy.Sample(data, 0.1, config);

        // Assert
        Assert.Equal(500, sample.Rows.Count);

        // For reasonable class counts (2-100), should use stratified
        if (numClasses >= 2 && numClasses <= 100)
        {
            var sourceDistribution = GetLabelDistribution(data, "Label");
            var sampleDistribution = GetLabelDistribution(sample, "Label");

            var maxDifference = sourceDistribution.Max(kvp =>
                Math.Abs(kvp.Value - sampleDistribution.GetValueOrDefault(kvp.Key, 0.0)));

            Assert.True(maxDifference < 0.1, $"With {numClasses} classes, stratified should be used");
        }
    }

    [Fact]
    public void Sample_PreservesColumnTypes()
    {
        // Arrange
        var data = SampleDataGenerator.GenerateMixedData(1000);
        var strategy = new AdaptiveSamplingStrategy();
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

    [Fact]
    public void Validate_ReturnsReasonForStrategySelection()
    {
        // Arrange
        var source = SampleDataGenerator.GenerateBalancedData(1000, numClasses: 3);
        var strategy = new AdaptiveSamplingStrategy();
        var config = new SamplingConfiguration { LabelColumn = "Label" };
        var sample = strategy.Sample(source, 0.1, config);

        // Act
        var result = strategy.Validate(source, sample, config);

        // Assert
        var reason = result.Details["AdaptiveReason"].ToString();
        Assert.NotNull(reason);
        Assert.NotEmpty(reason);
        Assert.Contains("class", reason, StringComparison.OrdinalIgnoreCase);
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
