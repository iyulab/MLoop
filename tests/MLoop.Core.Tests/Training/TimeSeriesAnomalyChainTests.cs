using System.Globalization;
using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// The detectors of a time-series anomaly run are a fallback chain: SR-CNN, and the SSA spike detector only where SR-CNN
/// could not be fitted. There is no label to rank detectors by, so a run that fitted SR-CNN has its model — fitting the
/// next detector as well was most of the run's time, and recorded a "trial" of a detector that was never a candidate.
/// </summary>
public sealed class TimeSeriesAnomalyChainTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop_anomaly_chain_{Guid.NewGuid():N}");

    public TimeSeriesAnomalyChainTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string WriteSeries(int rows)
    {
        var rng = new Random(5);
        var path = Path.Combine(_dir, "series.csv");
        File.WriteAllLines(path, new[] { "value" }.Concat(Enumerable.Range(0, rows).Select(i =>
            (100.0 + 10.0 * Math.Sin(2 * Math.PI * i / 12.0) + (i % 100 == 50 ? 40.0 : 0.0) + rng.NextDouble() - 0.5)
            .ToString("F4", CultureInfo.InvariantCulture))));
        return path;
    }

    [TimeSeriesNativeFact]
    public async Task AFittedFirstDetectorIsTheModelAndTheOnlyTrial()
    {
        var ctx = new MLContext(seed: 7);
        var result = await new TrainingRunner(ctx, new CsvDataLoader(ctx)).RunAsync(new TrainingConfig
        {
            ModelName = "series",
            DataFile = WriteSeries(1_000),
            LabelColumn = "value",
            Task = "time-series-anomaly",
            TimeLimitSeconds = 30,
        });

        Assert.Equal("SrCnnAnomaly", result.Trainer.Name);
        var trial = Assert.Single(result.Trials);
        Assert.Equal("SrCnnAnomaly", trial.Trainer.Name);
    }
}
