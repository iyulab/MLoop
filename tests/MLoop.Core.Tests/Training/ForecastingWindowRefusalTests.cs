using System.Globalization;
using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// ML.NET refuses a forecasting window unless the training rows (all but the held-out horizon) are more than twice
/// the window. The run's own check asked for less (more rows than the window) and let ML.NET's message through, so the
/// advice named a row count that was not enough. The check precedes the fit, so it runs where the native does not.
/// </summary>
public sealed class ForecastingWindowRefusalTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop_window_{Guid.NewGuid():N}");

    public ForecastingWindowRefusalTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    [Fact]
    public async Task AWindowOfMoreThanHalfTheTrainingRows_IsRefusedWithTheRowsItNeeds()
    {
        var path = Path.Combine(_dir, "series.csv");
        File.WriteAllLines(path, new[] { "value" }.Concat(Enumerable.Range(0, 100)
            .Select(i => (50.0 + 0.1 * i).ToString("F3", CultureInfo.InvariantCulture))));

        var ctx = new MLContext(seed: 42);
        // 100 rows, horizon 10: 90 training rows, a window of 60 is more than half of them
        var error = await Assert.ThrowsAsync<InvalidOperationException>(() => new TrainingRunner(ctx, new CsvDataLoader(ctx)).RunAsync(new TrainingConfig
        {
            ModelName = "series",
            DataFile = path,
            LabelColumn = "value",
            Task = "forecasting",
            Horizon = 10,
            WindowSize = 60,
            TimeLimitSeconds = 10,
        }));

        Assert.Contains("at least 131 rows", error.Message, StringComparison.Ordinal);
        Assert.Contains("window_size", error.Message, StringComparison.Ordinal);
    }
    [Fact]
    public async Task AHorizonLongerThanTheSeries_IsRefusedNamingTheHorizon_NotTheWindow()
    {
        // "or a smaller window_size" was all it said to a series of 300 points with a horizon of 500 - the one setting that cannot fix it.
        var path = Path.Combine(_dir, "short.csv");
        File.WriteAllLines(path, new[] { "value" }.Concat(Enumerable.Range(0, 300)
            .Select(i => (50.0 + 0.1 * i).ToString("F3", CultureInfo.InvariantCulture))));

        var ctx = new MLContext(seed: 42);
        var error = await Assert.ThrowsAsync<InvalidOperationException>(() => new TrainingRunner(ctx, new CsvDataLoader(ctx)).RunAsync(new TrainingConfig
        {
            ModelName = "series",
            DataFile = path,
            LabelColumn = "value",
            Task = "forecasting",
            Horizon = 500,
            TimeLimitSeconds = 10,
        }));

        Assert.Contains("a horizon of at most 149", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("window_size", error.Message, StringComparison.Ordinal);
    }
}
