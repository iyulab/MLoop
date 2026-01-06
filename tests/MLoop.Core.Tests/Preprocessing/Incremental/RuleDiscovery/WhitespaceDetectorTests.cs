using Microsoft.Data.Analysis;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Detectors;
using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Tests.Preprocessing.Incremental.RuleDiscovery;

public sealed class WhitespaceDetectorTests
{
    private readonly WhitespaceDetector _detector = new();

    [Fact]
    public async Task DetectAsync_NoIssues_ReturnsEmpty()
    {
        var column = new StringDataFrameColumn("test", new[] { "clean", "values", "here" });
        var patterns = await _detector.DetectAsync(column, "test");
        Assert.Empty(patterns);
    }

    [Fact]
    public async Task DetectAsync_LeadingSpaces_DetectsPattern()
    {
        var column = new StringDataFrameColumn("test", new[] { " leading", "clean", " another" });
        var patterns = await _detector.DetectAsync(column, "test");
        Assert.Single(patterns);
        Assert.Equal(2, patterns[0].Occurrences);
    }

    [Fact]
    public async Task DetectAsync_TrailingSpaces_DetectsPattern()
    {
        var column = new StringDataFrameColumn("test", new[] { "trailing ", "clean", "another " });
        var patterns = await _detector.DetectAsync(column, "test");
        Assert.Single(patterns);
        Assert.Equal(2, patterns[0].Occurrences);
    }

    [Fact]
    public async Task DetectAsync_MultipleSpaces_DetectsPattern()
    {
        var column = new StringDataFrameColumn("test", new[] { "double  space", "clean", "triple   space" });
        var patterns = await _detector.DetectAsync(column, "test");
        Assert.Single(patterns);
        Assert.Equal(2, patterns[0].Occurrences);
    }

    [Fact]
    public async Task DetectAsync_TabsAndNewlines_DetectsPattern()
    {
        var column = new StringDataFrameColumn("test", new[] { "with\ttab", "clean", "with\nnewline" });
        var patterns = await _detector.DetectAsync(column, "test");
        Assert.Single(patterns);
        Assert.Equal(2, patterns[0].Occurrences);
    }

    [Fact]
    public async Task DetectAsync_AllTypes_CountsAll()
    {
        var column = new StringDataFrameColumn("test", new[]
        {
            " leading",
            "trailing ",
            "multiple  spaces",
            "with\ttab"
        });
        var patterns = await _detector.DetectAsync(column, "test");
        Assert.Single(patterns);
        Assert.Equal(4, patterns[0].Occurrences);
    }

    [Fact]
    public void IsApplicable_StringColumn_ReturnsTrue()
    {
        var column = new StringDataFrameColumn("test", new[] { "a" });
        Assert.True(_detector.IsApplicable(column));
    }

    [Fact]
    public void IsApplicable_NumericColumn_ReturnsFalse()
    {
        var column = new Int32DataFrameColumn("test", new[] { 1 });
        Assert.False(_detector.IsApplicable(column));
    }
}
