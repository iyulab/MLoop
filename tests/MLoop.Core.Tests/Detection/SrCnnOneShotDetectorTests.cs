using MLoop.Core.Detection;
using Xunit;

namespace MLoop.Core.Tests.Detection;

public class SrCnnOneShotDetectorTests
{
    /// <summary>SR-CNN's FFT needs the MKL native library — absent on some CI runners (same guard
    /// class as PredictionServiceTests.MklAvailable, but probing the entire-series API this suite
    /// actually exercises).</summary>
    private static readonly Lazy<bool> MklAvailable = new(() =>
    {
        try
        {
            SrCnnOneShotDetector.Detect(
                Enumerable.Range(0, 24).Select(i => 10.0 + (i % 3) * 0.1).ToList(),
                new OneShotAnomalyOptions { Period = 0 });
            return true;
        }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException
                                   || ex.InnerException is DllNotFoundException)
        {
            return false;
        }
    });

    private static List<double> SpikeSeries(int length = 60, int spikeAt = 30, double spikeValue = 100.0)
    {
        var series = Enumerable.Range(0, length).Select(i => 10.0 + (i % 3) * 0.1).ToList();
        series[spikeAt] = spikeValue;
        return series;
    }

    /// <summary>Deterministic pseudo-noise in (-1, 1) — a seeded <c>Random</c> is not guaranteed
    /// stable across runtimes, and these tests assert on dispersion.</summary>
    private static double Jitter(int i) => Math.Sin(i * 12.9898) * 43758.5453 % 1.0;

    /// <summary>
    /// Sine + noise around <paramref name="level"/> with four injected spikes. <paramref name="scale"/>
    /// multiplies every deviation from the level (noise, seasonal swing, and spikes alike), so
    /// <c>level</c> and <c>scale</c> move the series' magnitude and its dispersion independently —
    /// that separation is what the control-limit contract is about.
    /// </summary>
    private static List<double> SeasonalNoisySeries(double level = 25.0, double scale = 1.0, int length = 300)
    {
        var spikes = new HashSet<int> { 50, 120, 200, 250 };
        return Enumerable.Range(0, length)
            .Select(i =>
            {
                var deviation = 2.2 * Math.Sin(2 * Math.PI * i / 63) + 0.4 * Jitter(i) + (spikes.Contains(i) ? 12.0 : 0.0);
                return level + scale * deviation;
            })
            .ToList();
    }

    private static double MedianControlWidth(OneShotAnomalyResult result)
    {
        var widths = result.Points.Select(p => p.ControlUpper - p.ControlLower).OrderBy(w => w).ToList();
        return widths[widths.Count / 2];
    }

    [Fact]
    public void Detect_ControlLimits_TrackDispersionNotMagnitude()
    {
        if (!MklAvailable.Value)
            return;

        // The defect this pins: the SR-CNN *margin* is scaled by CalculateBoundaryUnit, which measures
        // the series' level/trend, not its spread — so it widens ~41x when the level shifts and does
        // not move at all when the noise grows 5x. Control limits must do exactly the opposite.
        var baseline = SrCnnOneShotDetector.Detect(SeasonalNoisySeries());
        var shifted = SrCnnOneShotDetector.Detect(SeasonalNoisySeries(level: 1025.0));
        var noisier = SrCnnOneShotDetector.Detect(SeasonalNoisySeries(scale: 5.0));

        var baseWidth = MedianControlWidth(baseline);
        Assert.True(baseWidth > 0, "baseline control band collapsed to zero width");

        var shiftedRatio = MedianControlWidth(shifted) / baseWidth;
        Assert.InRange(shiftedRatio, 0.8, 1.25); // level +1000, same spread ⇒ same band

        var noisierRatio = MedianControlWidth(noisier) / baseWidth;
        Assert.InRange(noisierRatio, 3.5, 6.5); // spread x5 ⇒ band ~5x
    }

    [Fact]
    public void Detect_NormalPoints_LieWithinControlLimits()
    {
        if (!MklAvailable.Value)
            return;

        // The SPC semantic: a control chart is unusable if normal operation plots as violations.
        var result = SrCnnOneShotDetector.Detect(SeasonalNoisySeries());

        var normal = result.Points.Where(p => !p.IsAnomaly).ToList();
        var inside = normal.Count(p => p.Value >= p.ControlLower && p.Value <= p.ControlUpper);

        Assert.True(inside >= normal.Count * 0.95,
            $"only {inside}/{normal.Count} normal points inside the control band");
    }

    [Fact]
    public void Detect_Anomalies_LieOutsideMargin()
    {
        if (!MklAvailable.Value)
            return;

        // Upstream invariant, pinned so the margin band keeps explaining the verdict: SR-CNN clears
        // the anomaly flag for any point inside the margin, so IsAnomaly ⇒ outside. (The converse
        // does not hold — the margin never raises a flag — which is why it is not a control limit.)
        var result = SrCnnOneShotDetector.Detect(SeasonalNoisySeries());

        Assert.Contains(result.Points, p => p.IsAnomaly);
        Assert.All(result.Points.Where(p => p.IsAnomaly), p =>
            Assert.True(p.Value < p.MarginLower || p.Value > p.MarginUpper,
                $"anomaly at {p.Index} sits inside its margin band"));
    }

    [Fact]
    public void Detect_ConstantSeries_ControlBandDoesNotExcludeNormalPoints()
    {
        if (!MklAvailable.Value)
            return;

        // Zero dispersion ⇒ a zero-width band is the honest answer, but it must not report the
        // flat series as all-violations (inclusive comparison, no NaN).
        var result = SrCnnOneShotDetector.Detect(Enumerable.Repeat(7.0, 48).ToList(),
            new OneShotAnomalyOptions { Period = 0 });

        Assert.All(result.Points, p =>
        {
            Assert.False(double.IsNaN(p.ControlLower) || double.IsNaN(p.ControlUpper));
            Assert.True(p.ControlUpper >= p.ControlLower);
        });
        var violations = result.Points.Count(p => p.Value < p.ControlLower || p.Value > p.ControlUpper);
        Assert.True(violations <= 2, $"constant series produced {violations} control violations");
    }

    [Fact]
    public void Detect_SpikeSeries_FlagsSpikeWithBounds()
    {
        if (!MklAvailable.Value)
            return; // MKL natives absent; coverage runs where MKL exists.

        var series = SpikeSeries();
        var result = SrCnnOneShotDetector.Detect(series);

        Assert.Equal(series.Count, result.Points.Count);

        // The injected spike must be flagged.
        var spike = result.Points[30];
        Assert.True(spike.IsAnomaly, $"spike at index 30 should be anomalous (score={spike.Score:F3})");

        // Both bands must bracket the expected value.
        Assert.All(result.Points, p =>
        {
            Assert.True(p.MarginUpper >= p.ExpectedValue,
                $"margin upper {p.MarginUpper} < expected {p.ExpectedValue} at index {p.Index}");
            Assert.True(p.ExpectedValue >= p.MarginLower,
                $"expected {p.ExpectedValue} < margin lower {p.MarginLower} at index {p.Index}");
            Assert.True(p.ControlUpper >= p.ExpectedValue && p.ExpectedValue >= p.ControlLower,
                $"control band does not bracket expected at index {p.Index}");
        });

        // The detector should not scream anomaly on the flat bulk of the series.
        Assert.True(result.AnomalyCount < series.Count / 4,
            $"too many anomalies for a single-spike series ({result.AnomalyCount}/{series.Count})");
    }

    [Fact]
    public void Detect_FlatSeries_ProducesNoOrFewAnomalies()
    {
        if (!MklAvailable.Value)
            return;

        var series = Enumerable.Range(0, 48).Select(i => 5.0 + (i % 4) * 0.05).ToList();
        var result = SrCnnOneShotDetector.Detect(series, new OneShotAnomalyOptions { Period = 0 });

        Assert.Equal(series.Count, result.Points.Count);
        Assert.Equal(0, result.Period);
        Assert.True(result.AnomalyCount <= 2, $"flat series flagged {result.AnomalyCount} anomalies");
    }

    [Fact]
    public void Detect_IndexAndValueRoundTrip()
    {
        if (!MklAvailable.Value)
            return;

        var series = SpikeSeries(length: 24, spikeAt: 12);
        var result = SrCnnOneShotDetector.Detect(series);

        // Output order must match input order — the Index/Value pair is the caller's join key.
        Assert.All(result.Points, p => Assert.Equal(series[p.Index], p.Value));
        Assert.Equal(Enumerable.Range(0, series.Count), result.Points.Select(p => p.Index));
    }

    [Fact]
    public void Detect_TooFewPoints_ThrowsActionable()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            SrCnnOneShotDetector.Detect(Enumerable.Repeat(1.0, SrCnnOneShotDetector.MinimumPoints - 1).ToList()));
        Assert.Contains($"{SrCnnOneShotDetector.MinimumPoints}", ex.Message);
    }

    [Theory]
    [InlineData(-0.1, 99.0, null)]
    [InlineData(1.1, 99.0, null)]
    [InlineData(0.3, -1.0, null)]
    [InlineData(0.3, 100.5, null)]
    [InlineData(0.3, 99.0, -1)]
    public void Detect_OutOfRangeOptions_Throw(double threshold, double sensitivity, int? period)
    {
        var series = Enumerable.Repeat(1.0, 24).ToList();
        Assert.Throws<ArgumentException>(() => SrCnnOneShotDetector.Detect(series, new OneShotAnomalyOptions
        {
            Threshold = threshold,
            Sensitivity = sensitivity,
            Period = period,
        }));
    }
}
