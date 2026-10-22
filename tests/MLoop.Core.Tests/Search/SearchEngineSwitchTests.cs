using System.Globalization;
using Microsoft.ML;
using MLoop.Core.AutoML;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Tests.Common;
using Xunit.Abstractions;

namespace MLoop.Core.Tests.Search;

/// <summary>
/// The internal switch between AutoML's search and MLoop's: the same binary run on
/// both, the MLoop one giving a result of the same shape — trainer, test metrics, a Probability, the
/// trial records — and a model no worse than the data allows. These are the only place both searches run.
/// The parity gate itself (trial-count-fixed distributions, budget, determinism) is measured on CI, where LightGBM loads.
/// TODO(AutoML removal): drops to the MLoop half when the AutoML package goes.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public sealed class SearchEngineSwitchTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"mloop_engine_{Guid.NewGuid():N}");
    private readonly ITestOutputHelper _output;

    public SearchEngineSwitchTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* temp */ }
    }

    private string WriteCsv(int rows = 600)
    {
        var random = new Random(17);
        var lines = new List<string> { "x,z,colour,churn" };
        string[] colours = ["red", "green", "blue", "grey"];
        for (var i = 0; i < rows; i++)
        {
            var x = random.NextDouble();
            var colour = colours[random.Next(colours.Length)];
            var churn = x + (colour == "red" ? 0.35 : 0) + random.NextDouble() * 0.5 > 0.95;
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{x:F4},{random.NextDouble():F4},{colour},{(churn ? "yes" : "no")}"));
        }
        var path = Path.Combine(_dir, "train.csv");
        File.WriteAllLines(path, lines);
        return path;
    }

    private static async Task<AutoMLResult> TrainAsync(string csv, SearchEngine engine, int seconds = 8)
    {
        var ctx = new MLContext(seed: 42);
        var runner = new AutoMLRunner(ctx, new CsvDataLoader(ctx));
        return await runner.RunAsync(new TrainingConfig
        {
            ModelName = "engine",
            DataFile = csv,
            LabelColumn = "churn",
            Task = "binary-classification",
            TimeLimitSeconds = seconds,
            Metric = "auc",
            Engine = engine,
        }, null, CancellationToken.None);
    }

    [Fact]
    public async Task TheMLoopSearch_GivesAResultOfTheSameShape()
    {
        var csv = WriteCsv();

        var mloop = await TrainAsync(csv, SearchEngine.MLoop);
        var automl = await TrainAsync(csv, SearchEngine.AutoML);

        _output.WriteLine($"AutoML: {automl.Trainer.Name} auc {automl.Metrics["auc"]:F4} · {automl.Trials.Count} trials");
        _output.WriteLine($"MLoop:  {mloop.Trainer.Name} auc {mloop.Metrics["auc"]:F4} · {mloop.Trials.Count} trials");

        Assert.Contains(mloop.Trainer.Name, new[] { "FastTree", "FastForest", "LightGbm", "LbfgsLogisticRegression", "SdcaLogisticRegression" });
        Assert.Equal("auc", mloop.RankingMetric);
        // At least AutoML's metric set on the test rows (AutoML's FastForest pipeline gives no Probability, so it has no ece).
        Assert.Subset(mloop.Metrics.Keys.ToHashSet(), automl.Metrics.Keys.ToHashSet());
        Assert.NotEmpty(mloop.Trials);
        Assert.All(mloop.Trials.Where(t => t.Error is null), t => Assert.Contains("auc", t.Metrics.Keys));
        Assert.All(mloop.Trials.Where(t => t.Error is null), t => Assert.True(t.Validation!.Folds >= 1));
        Assert.Equal(new MLoop.Search.Candidates.SearchSettings(Seed: 1, Threads: 1), mloop.Search);   // what config.json records
        Assert.Null(automl.Search);
        Assert.Contains("ece", mloop.Metrics.Keys);                                  // there is a Probability to calibrate against
        // Not the parity gate — a floor: the data separates well, so a working search clears it.
        Assert.True(mloop.Metrics["auc"] > 0.8, $"MLoop search auc {mloop.Metrics["auc"]}");
    }
}
