using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.Statistics;

namespace MLoop.Core.Tests.Preprocessing.Incremental.Statistics;

public class CategoricalAnalyzerTests
{
    private static AnalysisConfiguration DefaultConfig => new();

    #region Basic Statistics Tests

    [Fact]
    public void Analyze_SimpleValues_CalculatesCount()
    {
        var col = new StringDataFrameColumn("Cat", new[] { "A", "B", "C" });

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(3, stats.Count);
    }

    [Fact]
    public void Analyze_SimpleValues_CalculatesUniqueCount()
    {
        var col = new StringDataFrameColumn("Cat", new[] { "A", "B", "A", "C", "B" });

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(3, stats.UniqueCount);
    }

    [Fact]
    public void Analyze_TopValues_SortedByFrequency()
    {
        var col = new StringDataFrameColumn("Cat", new[] { "A", "B", "B", "C", "C", "C" });

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal("C", stats.TopValues[0].Value);
        Assert.Equal(3, stats.TopValues[0].Frequency);
        Assert.Equal("B", stats.TopValues[1].Value);
        Assert.Equal(2, stats.TopValues[1].Frequency);
        Assert.Equal("A", stats.TopValues[2].Value);
        Assert.Equal(1, stats.TopValues[2].Frequency);
    }

    [Fact]
    public void Analyze_Mode_IsMostFrequent()
    {
        var col = new StringDataFrameColumn("Cat", new[] { "A", "B", "B", "C" });

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal("B", stats.Mode);
        Assert.Equal(2, stats.ModeFrequency);
        Assert.Equal(50.0, stats.ModePercentage);
    }

    #endregion

    #region Entropy Tests

    [Fact]
    public void Analyze_UniformDistribution_MaxEntropy()
    {
        // 4 categories with equal frequency → entropy = log2(4) = 2.0
        var col = new StringDataFrameColumn("Cat", new[] { "A", "B", "C", "D" });

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(2.0, stats.Entropy, 5);
    }

    [Fact]
    public void Analyze_SingleCategory_ZeroEntropy()
    {
        var col = new StringDataFrameColumn("Cat", new[] { "A", "A", "A", "A" });

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(0.0, stats.Entropy);
    }

    [Fact]
    public void Analyze_TwoEqualCategories_EntropyIsOne()
    {
        // 2 equal categories → entropy = log2(2) = 1.0
        var col = new StringDataFrameColumn("Cat", new[] { "A", "B", "A", "B" });

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(1.0, stats.Entropy, 5);
    }

    #endregion

    #region Cardinality Tests

    [Fact]
    public void Analyze_CardinalityRatio_Computed()
    {
        var col = new StringDataFrameColumn("Cat", new[] { "A", "B", "C", "A", "B" });

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        // 3 unique / 5 total = 0.6
        Assert.Equal(0.6, stats.CardinalityRatio, 5);
    }

    [Fact]
    public void Analyze_HighCardinality_Detected()
    {
        // UniqueCount > 100 → IsHighCardinality
        var values = Enumerable.Range(0, 150).Select(i => $"Val_{i}").ToArray();
        var col = new StringDataFrameColumn("Cat", values);

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.True(stats.IsHighCardinality);
        Assert.False(stats.IsLowCardinality);
    }

    [Fact]
    public void Analyze_LowCardinality_Detected()
    {
        // UniqueCount < 20 AND CardinalityRatio < 0.1
        // 3 unique in 100 values → ratio = 0.03
        var values = Enumerable.Range(0, 100)
            .Select(i => i % 3 == 0 ? "A" : i % 3 == 1 ? "B" : "C").ToArray();
        var col = new StringDataFrameColumn("Cat", values);

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.True(stats.IsLowCardinality);
        Assert.False(stats.IsHighCardinality);
    }

    [Fact]
    public void Analyze_IdentifierLike_Detected()
    {
        // CardinalityRatio > 0.95 → IsLikelyIdentifier
        var values = Enumerable.Range(0, 100).Select(i => $"ID-{i}").ToArray();
        var col = new StringDataFrameColumn("Cat", values);

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.True(stats.IsLikelyIdentifier);
    }

    #endregion

    #region Null Handling Tests

    [Fact]
    public void Analyze_NullValues_CountedAsNULLString()
    {
        var col = new StringDataFrameColumn("Cat", new[] { "A", null, "B", null });

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        // null values are converted to "NULL" string
        Assert.Equal(4, stats.Count);
        Assert.Contains(stats.TopValues, tv => tv.Value == "NULL" && tv.Frequency == 2);
    }

    #endregion

    #region MaxCategoricalValues Config Tests

    [Fact]
    public void Analyze_MaxCategoricalValues_TruncatesTopValues()
    {
        var values = Enumerable.Range(0, 50).Select(i => $"Cat_{i}").ToArray();
        var col = new StringDataFrameColumn("Cat", values);
        var config = new AnalysisConfiguration { MaxCategoricalValues = 5 };

        var stats = CategoricalAnalyzer.Analyze(col, config);

        Assert.Equal(5, stats.TopValues.Count);
        Assert.Equal(50, stats.UniqueCount); // UniqueCount still accurate
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Analyze_EmptyColumn_ReturnsZeroStats()
    {
        var col = new StringDataFrameColumn("Cat", Array.Empty<string>());

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(0, stats.Count);
        Assert.Equal(0, stats.UniqueCount);
        Assert.Empty(stats.TopValues);
        Assert.Equal(0.0, stats.Entropy);
    }

    [Fact]
    public void Analyze_SingleValue_CorrectStats()
    {
        var col = new StringDataFrameColumn("Cat", new[] { "Only" });

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(1, stats.Count);
        Assert.Equal(1, stats.UniqueCount);
        Assert.Equal("Only", stats.Mode);
        Assert.Equal(0.0, stats.Entropy);
    }

    [Fact]
    public void Analyze_NullColumn_ThrowsArgumentNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            CategoricalAnalyzer.Analyze(null!, DefaultConfig));
    }

    [Fact]
    public void Analyze_NullConfig_ThrowsArgumentNull()
    {
        var col = new StringDataFrameColumn("Cat", new[] { "A" });

        Assert.Throws<ArgumentNullException>(() =>
            CategoricalAnalyzer.Analyze(col, null!));
    }

    [Fact]
    public void Analyze_EmptyStats_ComputedPropertiesDefault()
    {
        var col = new StringDataFrameColumn("Cat", Array.Empty<string>());

        var stats = CategoricalAnalyzer.Analyze(col, DefaultConfig);

        Assert.Equal(0.0, stats.CardinalityRatio);
        Assert.Null(stats.Mode);
        Assert.Equal(0, stats.ModeFrequency);
        Assert.Equal(0.0, stats.ModePercentage);
        Assert.False(stats.IsLikelyIdentifier);
        Assert.False(stats.IsHighCardinality);
    }

    #endregion
}
