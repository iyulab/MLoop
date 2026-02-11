using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.Models;
using MLoop.Core.Preprocessing.Incremental.Strategies;

namespace MLoop.Core.Tests.Preprocessing.Incremental.Strategies;

public class SamplingStrategyTests
{
    private static DataFrame CreateTestData(int rows = 100)
    {
        var df = new DataFrame();
        df.Columns.Add(new PrimitiveDataFrameColumn<double>("Value",
            Enumerable.Range(0, rows).Select(i => (double)i).ToArray()));
        df.Columns.Add(new StringDataFrameColumn("Label",
            Enumerable.Range(0, rows).Select(i => i % 3 == 0 ? "A" : i % 3 == 1 ? "B" : "C").ToArray()));
        return df;
    }

    #region RandomSamplingStrategy Tests

    [Fact]
    public void Random_Name_IsRandom()
    {
        var strategy = new RandomSamplingStrategy();
        Assert.Equal("Random", strategy.Name);
    }

    [Fact]
    public void Random_IsApplicable_AlwaysTrue()
    {
        var strategy = new RandomSamplingStrategy();
        var df = CreateTestData();
        Assert.True(strategy.IsApplicable(df, new SamplingConfiguration()));
    }

    [Fact]
    public void Random_Sample_ReturnsCorrectSize()
    {
        var strategy = new RandomSamplingStrategy();
        var df = CreateTestData(100);

        var sample = strategy.Sample(df, 0.1, new SamplingConfiguration());

        Assert.Equal(10, sample.Rows.Count);
    }

    [Fact]
    public void Random_Sample_PreservesColumnCount()
    {
        var strategy = new RandomSamplingStrategy();
        var df = CreateTestData();

        var sample = strategy.Sample(df, 0.1, new SamplingConfiguration());

        Assert.Equal(df.Columns.Count, sample.Columns.Count);
    }

    [Fact]
    public void Random_Sample_Deterministic()
    {
        var strategy = new RandomSamplingStrategy();
        var df = CreateTestData();

        var sample1 = strategy.Sample(df, 0.1, new SamplingConfiguration(), randomSeed: 42);
        var sample2 = strategy.Sample(df, 0.1, new SamplingConfiguration(), randomSeed: 42);

        // Same seed → same sample
        Assert.Equal(sample1.Rows.Count, sample2.Rows.Count);
        for (int i = 0; i < sample1.Rows.Count; i++)
        {
            Assert.Equal(sample1.Columns["Value"][i], sample2.Columns["Value"][i]);
        }
    }

    [Fact]
    public void Random_Sample_DifferentSeeds_DifferentResults()
    {
        var strategy = new RandomSamplingStrategy();
        var df = CreateTestData();

        var sample1 = strategy.Sample(df, 0.1, new SamplingConfiguration(), randomSeed: 42);
        var sample2 = strategy.Sample(df, 0.1, new SamplingConfiguration(), randomSeed: 99);

        // Different seeds should (very likely) produce different samples
        bool anyDifference = false;
        for (int i = 0; i < Math.Min((int)sample1.Rows.Count, (int)sample2.Rows.Count); i++)
        {
            if (!Equals(sample1.Columns["Value"][i], sample2.Columns["Value"][i]))
            {
                anyDifference = true;
                break;
            }
        }
        Assert.True(anyDifference);
    }

    [Fact]
    public void Random_Sample_RatioOne_ReturnsAllRows()
    {
        var strategy = new RandomSamplingStrategy();
        var df = CreateTestData(50);

        var sample = strategy.Sample(df, 1.0, new SamplingConfiguration());

        Assert.Equal(df.Rows.Count, sample.Rows.Count);
    }

    [Fact]
    public void Random_Sample_EmptyData_ReturnsEmpty()
    {
        var strategy = new RandomSamplingStrategy();
        var df = new DataFrame();
        df.Columns.Add(new PrimitiveDataFrameColumn<double>("Value", Array.Empty<double>()));

        var sample = strategy.Sample(df, 0.5, new SamplingConfiguration());

        Assert.Equal(0, sample.Rows.Count);
    }

    [Fact]
    public void Random_Sample_InvalidRatio_Throws()
    {
        var strategy = new RandomSamplingStrategy();
        var df = CreateTestData();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            strategy.Sample(df, 0.0, new SamplingConfiguration()));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            strategy.Sample(df, 1.5, new SamplingConfiguration()));
    }

    [Fact]
    public void Random_Sample_NullData_Throws()
    {
        var strategy = new RandomSamplingStrategy();

        Assert.Throws<ArgumentNullException>(() =>
            strategy.Sample(null!, 0.5, new SamplingConfiguration()));
    }

    [Fact]
    public void Random_Sample_MinimumOneRow()
    {
        var strategy = new RandomSamplingStrategy();
        var df = CreateTestData(100);

        // Very small ratio → at least 1 row
        var sample = strategy.Sample(df, 0.001, new SamplingConfiguration());

        Assert.True(sample.Rows.Count >= 1);
    }

    #endregion

    #region StratifiedSamplingStrategy Tests

    [Fact]
    public void Stratified_Name_IsStratified()
    {
        var strategy = new StratifiedSamplingStrategy();
        Assert.Equal("Stratified", strategy.Name);
    }

    [Fact]
    public void Stratified_IsApplicable_WithLabelColumn_True()
    {
        var strategy = new StratifiedSamplingStrategy();
        var df = CreateTestData();
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        Assert.True(strategy.IsApplicable(df, config));
    }

    [Fact]
    public void Stratified_IsApplicable_NoLabelColumn_False()
    {
        var strategy = new StratifiedSamplingStrategy();
        var df = CreateTestData();
        var config = new SamplingConfiguration { LabelColumn = null };

        Assert.False(strategy.IsApplicable(df, config));
    }

    [Fact]
    public void Stratified_IsApplicable_MissingLabelColumn_False()
    {
        var strategy = new StratifiedSamplingStrategy();
        var df = CreateTestData();
        var config = new SamplingConfiguration { LabelColumn = "NonExistent" };

        Assert.False(strategy.IsApplicable(df, config));
    }

    [Fact]
    public void Stratified_Sample_PreservesDistribution()
    {
        var strategy = new StratifiedSamplingStrategy();
        var df = CreateTestData(300); // 100 each of A, B, C
        var config = new SamplingConfiguration { LabelColumn = "Label" };

        var sample = strategy.Sample(df, 0.1, config);

        // Count labels in sample
        var labelCol = sample.Columns["Label"];
        var counts = new Dictionary<string, int>();
        for (int i = 0; i < labelCol.Length; i++)
        {
            var val = labelCol[i]?.ToString() ?? "NULL";
            counts[val] = counts.GetValueOrDefault(val, 0) + 1;
        }

        // Each class should be represented
        Assert.True(counts.ContainsKey("A"));
        Assert.True(counts.ContainsKey("B"));
        Assert.True(counts.ContainsKey("C"));
    }

    [Fact]
    public void Stratified_Sample_NoLabelColumn_Throws()
    {
        var strategy = new StratifiedSamplingStrategy();
        var df = CreateTestData();
        var config = new SamplingConfiguration { LabelColumn = null };

        Assert.Throws<ArgumentException>(() =>
            strategy.Sample(df, 0.1, config));
    }

    [Fact]
    public void Stratified_Validate_GoodDistribution_IsValid()
    {
        var strategy = new StratifiedSamplingStrategy();
        var df = CreateTestData(300);
        var config = new SamplingConfiguration
        {
            LabelColumn = "Label",
            DistributionTolerance = 0.1
        };

        var sample = strategy.Sample(df, 0.3, config);
        var result = strategy.Validate(df, sample, config);

        Assert.True(result.IsValid);
    }

    #endregion
}
