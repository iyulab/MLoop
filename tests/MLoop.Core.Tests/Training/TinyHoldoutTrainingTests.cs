using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// A file so small that its 20 % test partition holds no row, or one: the evaluator still answered — r², rmse and mae all 0
/// (a perfect error beside no explained variance) for none, an r² of 1.8e7 for one — and training reported them as measured.
/// Now they are recorded as undefined (the worst value of their direction), which the quality gate refuses.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class TinyHoldoutTrainingTests : IDisposable
{
    private readonly string _tmpDir = Path.Combine(Path.GetTempPath(), $"mloop_tinyholdout_{Guid.NewGuid():N}");

    public TinyHoldoutTrainingTests() => Directory.CreateDirectory(_tmpDir);

    public void Dispose()
    {
        if (Directory.Exists(_tmpDir))
            Directory.Delete(_tmpDir, recursive: true);
    }

    private async Task<TrainedModel> TrainRegressionAsync(int rows)
    {
        var path = Path.Combine(_tmpDir, $"r{rows}.csv");
        var random = new Random(rows);
        await File.WriteAllLinesAsync(path, new[] { "f0,label" }.Concat(Enumerable.Range(0, rows).Select(_ =>
        {
            var x = random.NextDouble() * 4 - 2;
            return FormattableString.Invariant($"{x:0.0000},{2 * x + 0.3 * (random.NextDouble() - 0.5):0.0000}");
        })));
        var ctx = new MLContext(seed: 42);
        var runner = new TrainingRunner(ctx, new CsvDataLoader(ctx));
        return await runner.RunAsync(new TrainingConfig
        {
            ModelName = "tiny",
            DataFile = path,
            LabelColumn = "label",
            Task = "regression",
            TimeLimitSeconds = 5,
            MaxTrials = 3,
        }, null, CancellationToken.None);
    }

    [Fact]
    public async Task ATestPartitionOfNoRow_RecordsEveryMetricAsUndefined_NotAsZero()
    {
        var result = await TrainRegressionAsync(5);

        Assert.Equal(double.MinValue, result.Metrics["r_squared"]);
        Assert.Equal(double.MaxValue, result.Metrics["rmse"]);
        Assert.Equal(double.MaxValue, result.Metrics["mae"]);
    }

    [Fact]
    public async Task ATestPartitionOfOneRow_RecordsRSquaredAsUndefined_ButKeepsTheErrors()
    {
        var result = await TrainRegressionAsync(6);

        Assert.Equal(double.MinValue, result.Metrics["r_squared"]);
        Assert.InRange(result.Metrics["rmse"], 0, 100);
    }
}
