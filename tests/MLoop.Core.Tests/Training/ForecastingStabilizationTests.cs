using System.Globalization;
using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// ML.NET's SSA stabilizes its forecasting recurrence by finding the roots of a polynomial of the window's degree and
/// moving the ones outside the unit circle in. A trend is a double root at one, and for a series with a trend the root
/// finding of a degree-255 polynomial misplaces roots by an amount that depends on the noise: measured on a straight
/// line plus noise (1,825 points, forecast 100 ahead) the stabilized model missed four series in four by 4.2 to 4.6
/// where a forecaster that knew the line would miss by 0.8; with the stabilization off the same four were forecast to
/// within 0.8 to 1.0. A textbook SSA forecasts every one of them, and across six series shapes the stabilized model
/// failed on a third of the series with a trend, an exponential growth or a yearly season.
/// </summary>
public sealed class ForecastingStabilizationTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop_stabilization_{Guid.NewGuid():N}");

    public ForecastingStabilizationTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private const int Length = 1_825;
    private const int Horizon = 100;

    private string WriteLine(int seed)
    {
        var rng = new Random(seed);
        var path = Path.Combine(_dir, $"line-{seed}.csv");
        File.WriteAllLines(path, new[] { "value" }.Concat(Enumerable.Range(0, Length).Select(i =>
        {
            // Box-Muller: a normal noise with a standard deviation of one
            var noise = Math.Sqrt(-2.0 * Math.Log(1.0 - rng.NextDouble())) * Math.Cos(2.0 * Math.PI * rng.NextDouble());
            return (100.0 + 0.05 * i + noise).ToString("F4", CultureInfo.InvariantCulture);
        })));
        return path;
    }

    [TimeSeriesNativeFact]
    public async Task ATrendPlusNoise_IsForecastToWithinTheNoise_NotToGarbage()
    {
        var misses = new List<string>();
        foreach (var seed in new[] { 1, 2, 3, 4 })
        {
            var ctx = new MLContext(seed: 42);
            var result = await new TrainingRunner(ctx, new CsvDataLoader(ctx)).RunAsync(new TrainingConfig
            {
                ModelName = $"line{seed}",
                DataFile = WriteLine(seed),
                LabelColumn = "value",
                Task = "forecasting",
                Horizon = Horizon,
                TimeLimitSeconds = 30,
            });

            Assert.True(result.Metrics.TryGetValue("mae", out var mae), "the run records a mean absolute error");
            // The noise alone is about 0.8 and the stabilized model missed by 4.2 or more. Repeating the last value would
            // miss by up to 3.6 here, so two is the line between a forecast of the line and anything else.
            if (mae >= 2.0)
                misses.Add($"seed {seed}: mean absolute error {mae:F2}");
        }

        Assert.True(misses.Count == 0, "a straight line plus noise is forecast to within the noise, but " + string.Join("; ", misses));
    }
}
