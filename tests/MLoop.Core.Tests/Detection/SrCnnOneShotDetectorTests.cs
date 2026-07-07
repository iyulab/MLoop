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

        // Margin mode: every point carries a coherent SPC band around its expected value.
        Assert.All(result.Points, p =>
        {
            Assert.True(p.UpperBound >= p.ExpectedValue,
                $"upper {p.UpperBound} < expected {p.ExpectedValue} at index {p.Index}");
            Assert.True(p.ExpectedValue >= p.LowerBound,
                $"expected {p.ExpectedValue} < lower {p.LowerBound} at index {p.Index}");
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
