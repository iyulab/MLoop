using MLoop.Core.AutoML;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// The invariant these pin is not a preference: ML.NET throws
/// "Must be non-negative and not larger than window size (Parameter
/// 'JudgementWindowSize')" when it is violated, and the caller answers a
/// throwing detector by moving on to the next one. So breaking it does not
/// produce a failure anyone sees — it produces a run that quietly used a
/// different algorithm, which is why it survived unnoticed.
/// </summary>
public class SrCnnWindowSizingTests
{
    [Theory]
    [InlineData(12)]   // the caller's own minimum
    [InlineData(20)]
    [InlineData(30)]   // the size that surfaced this in CI
    [InlineData(50)]
    [InlineData(83)]   // last size the old literal 21 rejected
    [InlineData(84)]   // first size it accepted
    [InlineData(200)]
    [InlineData(10_000)]
    public void EveryWindowFitsInsideTheAnalysisWindow(int rows)
    {
        var s = SrCnnWindowSizes.ForSeries(rows);
        Assert.True(s.JudgementWindowSize <= s.WindowSize,
            $"{rows} rows: judgement {s.JudgementWindowSize} > window {s.WindowSize}");
        Assert.True(s.BackAddWindowSize <= s.WindowSize);
        Assert.True(s.LookaheadWindowSize <= s.WindowSize);
        Assert.True(s.AveragingWindowSize <= s.WindowSize);
        Assert.True(s.JudgementWindowSize > 0);
    }

    [Fact]
    public void ShortSeriesAreTheCaseThatUsedToThrow()
    {
        // 30 rows gives an analysis window of 8. The old code asked for a
        // judgement window of 21 regardless, which is the exact throw.
        var s = SrCnnWindowSizes.ForSeries(30);
        Assert.Equal(8, s.WindowSize);
        Assert.Equal(8, s.JudgementWindowSize);
    }

    [Fact]
    public void LongSeriesKeepMlNetsPreferredJudgementWindow()
    {
        // Nothing changes above 84 rows; the fix must not move behaviour that
        // was already correct.
        foreach (var rows in new[] { 84, 100, 500, 10_000 })
        {
            var s = SrCnnWindowSizes.ForSeries(rows);
            Assert.Equal(SrCnnWindowSizes.PreferredJudgementWindowSize, s.JudgementWindowSize);
        }
    }

    [Fact]
    public void TheAnalysisWindowStaysWithinItsDeclaredBounds()
    {
        Assert.Equal(SrCnnWindowSizes.MinimumWindowSize, SrCnnWindowSizes.ForSeries(1).WindowSize);
        Assert.Equal(SrCnnWindowSizes.MaximumWindowSize, SrCnnWindowSizes.ForSeries(1_000_000).WindowSize);
    }
}
