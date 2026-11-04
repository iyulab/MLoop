using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// The width of a forecast band is a quantile of how many native half-widths forecasts of unseen points missed by — the
/// arithmetic is here, apart from the SSA fits that produce the scores (those need ML.NET's MKL native and are walked in
/// the Linux lifecycle check).
/// </summary>
public sealed class ForecastBandCalibrationTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop_band_{Guid.NewGuid():N}");

    public ForecastBandCalibrationTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    [Fact]
    public void ScoreIsTheMissOverTheHalfWidthOnTheSideThePointFell()
    {
        // band 8..14 about 10: a point at 13 is 3 above, the upper half-width is 4
        Assert.Equal(0.75, ForecastBandCalibration.Score(13, 10, 8, 14)!.Value, 10);
        // a point at 7 is 3 below, the lower half-width is 2
        Assert.Equal(1.5, ForecastBandCalibration.Score(7, 10, 8, 14)!.Value, 10);
        Assert.Equal(0.0, ForecastBandCalibration.Score(10, 10, 8, 14)!.Value, 10);
    }

    [Theory]
    [InlineData(double.NaN, 10, 8, 14)]
    [InlineData(13, double.NaN, 8, 14)]
    [InlineData(13, 10, double.NegativeInfinity, 14)]
    [InlineData(13, 10, 8, 10)]   // no width on the side the point fell
    [InlineData(13, 10, 8, 9)]    // a bound on the wrong side of the forecast
    public void AScoreNeedsABandThatGivesADistance(double actual, double forecast, double lower, double upper)
        => Assert.Null(ForecastBandCalibration.Score(actual, forecast, lower, upper));

    [Fact]
    public void TheQuantileIsTheCeilingOfNPlusOneTimesLevelThSmallest()
    {
        var scores = Enumerable.Range(1, 99).Select(i => (double)i).Reverse().ToArray();   // 1..99, unsorted
        // ceil(100 * 0.95) = 95
        Assert.Equal(95, ForecastBandCalibration.Quantile(scores, 0.95));
        Assert.Equal(90, ForecastBandCalibration.Quantile(scores, 0.90));
    }

    [Fact]
    public void WhenTheRankPassesTheLargestScoreTheLargestIsReturned()
    {
        // ceil(10 * 0.95) = 10 → the largest of nine; ceil(10 * 0.99) = 10 as well
        var nine = Enumerable.Range(1, 9).Select(i => (double)i).ToArray();
        Assert.Equal(9, ForecastBandCalibration.Quantile(nine, 0.95));
        Assert.Equal(9, ForecastBandCalibration.Quantile(nine, 0.99));
        Assert.Equal(1, ForecastBandCalibration.Quantile([1.0], 0.5));
    }

    [Fact]
    public void FewerScoresThanTheMinimumFitNothing()
    {
        var few = Enumerable.Repeat(1.2, ForecastBandCalibration.MinimumScores - 1).ToArray();
        Assert.Null(ForecastBandCalibration.Fit(few, 0.95, origins: 3, horizon: 5, window: 12));

        var enough = Enumerable.Repeat(1.2, ForecastBandCalibration.MinimumScores).ToArray();
        var fit = ForecastBandCalibration.Fit(enough, 0.95, origins: 4, horizon: 5, window: 12)!;
        Assert.Equal(1.2, fit.Scale);
        Assert.Equal(ForecastBandCalibration.MinimumScores, fit.Scores);
        Assert.Equal(4, fit.Origins);
    }

    [Fact]
    public void ScoresThatAreNotFiniteAreNotCounted()
    {
        var scores = Enumerable.Repeat(1.0, ForecastBandCalibration.MinimumScores - 1).Append(double.NaN).Append(double.PositiveInfinity).ToArray();
        Assert.Null(ForecastBandCalibration.Fit(scores, 0.95, 2, 5, 12));
    }

    [Fact]
    public void TheBandIsScaledAboutTheForecastOnEachSide()
    {
        var calibration = new ForecastBandCalibration(0.95, 1.5, 5, 50, 10, 12, DateTimeOffset.UtcNow);
        var (lower, upper) = calibration.Apply(10f, 8f, 14f);
        Assert.Equal(7f, lower);    // 10 − 1.5·2
        Assert.Equal(16f, upper);   // 10 + 1.5·4

        // a scale under one narrows
        var narrower = new ForecastBandCalibration(0.95, 0.5, 5, 50, 10, 12, DateTimeOffset.UtcNow);
        Assert.Equal((9f, 12f), narrower.Apply(10f, 8f, 14f));
    }

    [Fact]
    public void ABandThatIsNotFiniteIsLeftAlone()
    {
        var calibration = new ForecastBandCalibration(0.95, 2, 5, 50, 10, 12, DateTimeOffset.UtcNow);
        var (lower, upper) = calibration.Apply(10f, float.NaN, 14f);
        Assert.True(float.IsNaN(lower));
        Assert.Equal(14f, upper);
    }

    [Fact]
    public void ARoundTripThroughTheFileBesideTheModelKeepsEverything()
    {
        var modelPath = Path.Combine(_dir, "model.zip");
        Assert.Null(ForecastBandCalibration.ReadBeside(modelPath));

        var written = new ForecastBandCalibration(0.95, 1.137, 9, 450, 50, 24, new DateTimeOffset(2026, 11, 4, 0, 0, 0, TimeSpan.Zero));
        written.WriteBeside(modelPath);

        Assert.Equal("forecast-band.json", Path.GetFileName(ForecastBandCalibration.PathBeside(modelPath)));
        Assert.Equal(written, ForecastBandCalibration.ReadBeside(modelPath));
    }

    [Fact]
    public void AConformalScaleHoldsItsLevelOnAFreshDrawFromTheSameDistribution()
    {
        // scores are |error| / half-width; draw them from a normal whose native band is 12 % too narrow, fit the scale,
        // and count how often a fresh draw falls inside the scaled band
        var rng = new Random(11);
        double Draw() => Math.Abs(Gaussian(rng)) / (1.96 * 0.877);   // true error ~ N(0,1), native half-width 1.96·0.877
        var fitScores = Enumerable.Range(0, 400).Select(_ => Draw()).ToArray();
        var fit = ForecastBandCalibration.Fit(fitScores, 0.95, 8, 50, 12)!;

        var inside = Enumerable.Range(0, 20_000).Count(_ => Draw() <= fit.Scale);
        var coverage = inside / 20_000.0;
        Assert.InRange(coverage, 0.93, 0.97);
        // and the unscaled band does not: it states 0.95 and holds P(|Z| ≤ 1.96·0.877) ≈ 0.914
        var unscaled = Enumerable.Range(0, 20_000).Count(_ => Draw() <= 1.0) / 20_000.0;
        Assert.InRange(unscaled, 0.90, 0.93);
    }

    private static double Gaussian(Random rng)
        => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
}
