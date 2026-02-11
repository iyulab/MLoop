using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.RuleDiscovery;

public class CategoryVariationDetectorTests
{
    private readonly CategoryVariationDetector _detector;

    public CategoryVariationDetectorTests()
    {
        _detector = new CategoryVariationDetector();
    }

    #region IsApplicable Tests

    [Fact]
    public void IsApplicable_StringColumn_ReturnsTrue()
    {
        var col = new StringDataFrameColumn("Cat", new[] { "A", "B" });
        Assert.True(_detector.IsApplicable(col));
    }

    [Fact]
    public void IsApplicable_NumericColumn_ReturnsFalse()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 1.0, 2.0 });
        Assert.False(_detector.IsApplicable(col));
    }

    #endregion

    #region Case Variation Detection

    [Fact]
    public async Task DetectAsync_CaseVariations_DetectsPattern()
    {
        // "Cat" and "cat" and "CAT" are case variations of the same category
        var col = new StringDataFrameColumn("Category", new[]
        {
            "Cat", "cat", "CAT", "Dog", "dog", "Bird"
        });

        var patterns = await _detector.DetectAsync(col, "Category");

        Assert.Contains(patterns, p =>
            p.Type == PatternType.CategoryVariation &&
            p.Description.Contains("Case variation"));
    }

    [Fact]
    public async Task DetectAsync_NoCaseVariations_NoPattern()
    {
        var col = new StringDataFrameColumn("Category", new[]
        {
            "Cat", "Dog", "Bird", "Fish"
        });

        var patterns = await _detector.DetectAsync(col, "Category");

        // No case variations → no case variation pattern
        Assert.DoesNotContain(patterns, p =>
            p.Description.Contains("Case variation"));
    }

    [Fact]
    public async Task DetectAsync_CaseVariations_ReportsExamples()
    {
        var col = new StringDataFrameColumn("Status", new[]
        {
            "Active", "active", "ACTIVE", "Inactive"
        });

        var patterns = await _detector.DetectAsync(col, "Status");

        var casePattern = patterns.FirstOrDefault(p => p.Description.Contains("Case variation"));
        Assert.NotNull(casePattern);
        Assert.NotNull(casePattern.Examples);
        Assert.NotEmpty(casePattern.Examples);
    }

    #endregion

    #region Similar Category (Typo) Detection

    [Fact]
    public async Task DetectAsync_SimilarCategories_DetectsTypos()
    {
        // "California" (10 chars) vs "Californla" (10 chars) → distance=1, similarity=0.9 >= 0.85
        var col = new StringDataFrameColumn("State", new[]
        {
            "California", "Californla", "Texas", "New York"
        });

        var patterns = await _detector.DetectAsync(col, "State");

        Assert.Contains(patterns, p =>
            p.Description.Contains("similar category pairs"));
    }

    [Fact]
    public async Task DetectAsync_VerySimilarNames_DetectsAsTypo()
    {
        // "Manufacturing" (13 chars) vs "Manufacturinq" (13 chars) → distance=1, similarity=0.923 >= 0.85
        var col = new StringDataFrameColumn("Dept", new[]
        {
            "Manufacturing", "Manufacturinq", "Engineering", "Sales"
        });

        var patterns = await _detector.DetectAsync(col, "Dept");

        Assert.Contains(patterns, p =>
            p.Description.Contains("similar category pairs"));
    }

    [Fact]
    public async Task DetectAsync_VeryDifferentCategories_NoTypoPattern()
    {
        var col = new StringDataFrameColumn("Type", new[]
        {
            "Alpha", "Beta", "Gamma", "Delta"
        });

        var patterns = await _detector.DetectAsync(col, "Type");

        Assert.DoesNotContain(patterns, p =>
            p.Description.Contains("similar category pairs"));
    }

    #endregion

    #region Edge Cases

    [Fact]
    public async Task DetectAsync_NumericColumn_ReturnsEmpty()
    {
        var col = new PrimitiveDataFrameColumn<double>("Num", new[] { 1.0, 2.0, 3.0 });

        var patterns = await _detector.DetectAsync(col, "Num");

        Assert.Empty(patterns);
    }

    [Fact]
    public async Task DetectAsync_HighCardinality_ReturnsEmpty()
    {
        // More than 100 unique values → not treated as categorical
        var values = Enumerable.Range(0, 150).Select(i => $"Value_{i}").ToArray();
        var col = new StringDataFrameColumn("HighCard", values);

        var patterns = await _detector.DetectAsync(col, "HighCard");

        Assert.Empty(patterns);
    }

    [Fact]
    public async Task DetectAsync_NullValues_Skipped()
    {
        var col = new StringDataFrameColumn("Cat", new[]
        {
            "A", null, "a", null, "B"
        });

        // Should not crash and should detect case variation A/a
        var patterns = await _detector.DetectAsync(col, "Cat");

        Assert.Contains(patterns, p => p.Description.Contains("Case variation"));
    }

    [Fact]
    public async Task DetectAsync_MissingValueStrings_Skipped()
    {
        var col = new StringDataFrameColumn("Cat", new[]
        {
            "A", "NULL", "N/A", "a", "B"
        });

        var patterns = await _detector.DetectAsync(col, "Cat");

        // NULL and N/A should be treated as missing, not as categories
        Assert.Contains(patterns, p => p.Description.Contains("Case variation"));
    }

    [Fact]
    public async Task DetectAsync_CancellationRequested_ThrowsOperationCanceled()
    {
        var col = new StringDataFrameColumn("Cat", new[] { "A", "B", "C" });
        var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            _detector.DetectAsync(col, "Cat", cts.Token));
    }

    [Fact]
    public async Task DetectAsync_EmptyColumn_ReturnsEmpty()
    {
        var col = new StringDataFrameColumn("Cat", Array.Empty<string>());

        var patterns = await _detector.DetectAsync(col, "Cat");

        Assert.Empty(patterns);
    }

    [Fact]
    public async Task DetectAsync_PatternHasCorrectColumnName()
    {
        var col = new StringDataFrameColumn("MyColumn", new[]
        {
            "Cat", "cat", "Dog"
        });

        var patterns = await _detector.DetectAsync(col, "MyColumn");

        Assert.All(patterns, p => Assert.Equal("MyColumn", p.ColumnName));
    }

    [Fact]
    public async Task DetectAsync_PatternType_IsCategoryVariation()
    {
        var col = new StringDataFrameColumn("Cat", new[]
        {
            "Alpha", "alpha", "ALPHA"
        });

        var patterns = await _detector.DetectAsync(col, "Cat");

        Assert.All(patterns, p => Assert.Equal(PatternType.CategoryVariation, p.Type));
    }

    #endregion
}
