using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// The held-out points of a forecasting run are the last <c>horizon</c> rows, and the forecast that looks at them
/// is the one made at the row before them. The run scored the forecast of the last row — of the points beyond the
/// file — against them, so a series with a season reported its phase as an error: measured on a generated series
/// (period 12, noise 1, horizon 10) the run read 6.3 where the forecast of the points that followed missed by 2.9.
/// </summary>
public sealed class ForecastingHoldoutTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop_holdout_{Guid.NewGuid():N}");

    public ForecastingHoldoutTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    [TimeSeriesNativeFact]
    public async Task ANoiselessSeason_IsScoredNearZero_NotByItsPhase()
    {
        // period 12, horizon 5: a forecast of the points after the held-out stretch sits five steps out of phase
        // with it, and misses a swing of 10 by about 6
        var path = Path.Combine(_dir, "series.csv");
        File.WriteAllLines(path, new[] { "value" }.Concat(Enumerable.Range(0, 144)
            .Select(i => (100.0 + 0.05 * i + 10.0 * Math.Sin(2 * Math.PI * i / 12.0)).ToString("F4", System.Globalization.CultureInfo.InvariantCulture))));

        var ctx = new MLContext(seed: 42);
        var result = await new TrainingRunner(ctx, new CsvDataLoader(ctx)).RunAsync(new TrainingConfig
        {
            ModelName = "series",
            DataFile = path,
            LabelColumn = "value",
            Task = "forecasting",
            Horizon = 5,
            TimeLimitSeconds = 30,
        });

        Assert.True(result.Metrics.TryGetValue("mae", out var mae), "the run records a mean absolute error");
        Assert.True(mae < 1.0, $"mean absolute error {mae}: the season is regular, so the forecast of the held-out points is close");
    }
}
