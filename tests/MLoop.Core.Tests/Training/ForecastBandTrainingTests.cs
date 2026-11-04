using System.Globalization;
using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// A forecasting run fits its band width on forecasts of stretches it has not seen: the held-out horizon and the
/// stretches before it. A series too short to give enough such points keeps ML.NET's band, and says so by having none.
/// (The coverage the width buys is walked on a true future in the Linux lifecycle check — it needs many series.)
/// </summary>
public sealed class ForecastBandTrainingTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop_band_training_{Guid.NewGuid():N}");

    public ForecastBandTrainingTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string WriteSeries(int rows, int seed)
    {
        var rng = new Random(seed);
        var path = Path.Combine(_dir, $"series-{rows}-{seed}.csv");
        File.WriteAllLines(path, new[] { "value" }.Concat(Enumerable.Range(0, rows).Select(i =>
            (50.0 + 0.04 * i + 5.0 * Math.Sin(2 * Math.PI * i / 12.0)
                + Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()))
            .ToString("F4", CultureInfo.InvariantCulture))));
        return path;
    }

    private static async Task<TrainedModel> TrainAsync(string path, int horizon, int windowSize = 0)
    {
        var ctx = new MLContext(seed: 7);
        return await new TrainingRunner(ctx, new CsvDataLoader(ctx)).RunAsync(new TrainingConfig
        {
            ModelName = "series",
            DataFile = path,
            LabelColumn = "value",
            Task = "forecasting",
            Horizon = horizon,
            WindowSize = windowSize,
            TimeLimitSeconds = 30,
        });
    }

    [TimeSeriesNativeFact]
    public async Task ALongEnoughSeriesGetsABandWidthFromTheStretchesItDidNotSee()
    {
        var result = await TrainAsync(WriteSeries(400, seed: 1), horizon: 12);

        var band = result.ForecastBand;
        Assert.NotNull(band);
        Assert.Equal(0.95, band.Level);
        Assert.Equal(1 + ForecastBandBacktest.Origins, band.Origins);   // the held-out stretch and eight before it
        Assert.Equal(band.Origins * 12, band.Scores);
        Assert.True(double.IsFinite(band.Scale) && band.Scale > 0, $"scale {band.Scale}");
    }

    [TimeSeriesNativeFact]
    public async Task ASeriesTooShortForEnoughScoredPointsKeepsTheNativeBand()
    {
        // 30 rows, window 5, horizon 3: the held-out stretch and five backtests are 18 points, under the minimum of 20
        var result = await TrainAsync(WriteSeries(30, seed: 2), horizon: 3, windowSize: 5);

        Assert.Null(result.ForecastBand);
        Assert.True(result.Metrics.ContainsKey("mae"), "the model itself is trained");
    }

    [TimeSeriesNativeFact]
    public async Task TheBackfitsReadAtMostTheCapOfPointsBeforeTheirOrigin()
    {
        // a series longer than the cap: every backtest fit reads the SeriesCap points before its origin, so the run
        // does not fit eight copies of the whole series
        var rows = ForecastBandBacktest.SeriesCap + 2_000;
        var rng = new Random(3);
        var series = Enumerable.Range(0, rows).Select(i => (float)(10 + Math.Sin(i / 6.0) + 0.1 * rng.NextDouble())).ToArray();

        var band = ForecastBandBacktest.Calibrate(new MLContext(seed: 3), series, trainSize: rows - 10, horizon: 10,
            windowSize: 24, configuredSeriesLength: 0, ownForecast: null, CancellationToken.None);

        Assert.NotNull(band);
        Assert.Equal(ForecastBandBacktest.Origins, band.Origins);   // no own forecast given: only the backtests count
        Assert.Equal(ForecastBandBacktest.Origins * 10, band.Scores);
    }
}
