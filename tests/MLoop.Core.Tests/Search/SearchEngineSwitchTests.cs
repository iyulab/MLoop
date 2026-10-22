using System.Globalization;
using Microsoft.ML;
using Microsoft.ML.Data;
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

    private static async Task<AutoMLResult> TrainAsync(string csv, SearchEngine engine, int seconds = 8,
        string label = "churn", string task = "binary-classification", string metric = "auc")
    {
        var ctx = new MLContext(seed: 42);
        var runner = new AutoMLRunner(ctx, new CsvDataLoader(ctx));
        return await runner.RunAsync(new TrainingConfig
        {
            ModelName = "engine",
            DataFile = csv,
            LabelColumn = label,
            Task = task,
            TimeLimitSeconds = seconds,
            Metric = metric,
            Engine = engine,
        }, null, CancellationToken.None);
    }

    private string WriteMulticlassCsv(int rows = 600)
    {
        var random = new Random(23);
        var lines = new List<string> { "x,z,colour,tier" };
        string[] colours = ["red", "green", "blue"];
        for (var i = 0; i < rows; i++)
        {
            var x = random.NextDouble();
            var colour = colours[random.Next(colours.Length)];
            var score = x + (colour == "red" ? 0.3 : 0) + random.NextDouble() * 0.3;
            var tier = score < 0.55 ? "bronze" : score < 0.95 ? "silver" : "gold";
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{x:F4},{random.NextDouble():F4},{colour},{tier}"));
        }
        var path = Path.Combine(_dir, "tiers.csv");
        File.WriteAllLines(path, lines);
        return path;
    }

    [Fact]
    public async Task TheMLoopSearch_GivesAMulticlassResultOfTheSameShape()
    {
        var csv = WriteMulticlassCsv();

        var mloop = await TrainAsync(csv, SearchEngine.MLoop, label: "tier", task: "multiclass-classification", metric: "macro_accuracy");
        var automl = await TrainAsync(csv, SearchEngine.AutoML, label: "tier", task: "multiclass-classification", metric: "macro_accuracy");

        _output.WriteLine($"AutoML: {automl.Trainer.Name} macro_accuracy {automl.Metrics["macro_accuracy"]:F4} · {automl.Trials.Count} trials");
        _output.WriteLine($"MLoop:  {mloop.Trainer.Name} macro_accuracy {mloop.Metrics["macro_accuracy"]:F4} · {mloop.Trials.Count} trials");

        Assert.Contains(mloop.Trainer.Name, MLoop.Search.Candidates.CandidateCatalog.Multiclass().Select(c => c.Name));
        Assert.Equal("macro_accuracy", mloop.RankingMetric);
        Assert.Subset(mloop.Metrics.Keys.ToHashSet(), automl.Metrics.Keys.ToHashSet());
        Assert.Contains("macro_f1", mloop.Metrics.Keys);
        Assert.All(mloop.Trials.Where(t => t.Error is null), t => Assert.Contains("macro_accuracy", t.Metrics.Keys));
        // The saved model answers in the label's own values, as the AutoML one does.
        var loadContext = new MLContext(seed: 1);
        var answers = mloop.Model.Transform(new CsvDataLoader(loadContext).LoadData(csv, "tier", "multiclass-classification"));
        Assert.Subset(new HashSet<string> { "bronze", "silver", "gold" },
            answers.GetColumn<ReadOnlyMemory<char>>("PredictedLabel").Select(v => v.ToString()).ToHashSet());
        // Each probability is named after its class, read from the Score column's annotations as for AutoML's models.
        var score = answers.Schema["Score"];
        var names = MulticlassScoreVocabulary.Of(score, ((VectorDataViewType)score.Type).Size);
        Assert.Equal(new HashSet<string> { "bronze", "silver", "gold" }, names!.ToHashSet());
        Assert.True(mloop.Metrics["macro_accuracy"] > 0.7, $"MLoop search macro_accuracy {mloop.Metrics["macro_accuracy"]}");
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

    private string WriteRegressionCsv(int rows = 600)
    {
        var random = new Random(31);
        var lines = new List<string> { "x,z,colour,price" };
        string[] colours = ["red", "green", "blue"];
        for (var i = 0; i < rows; i++)
        {
            var x = random.NextDouble();
            var colour = colours[random.Next(colours.Length)];
            // The noise grows with x, so the per-row band has something to learn.
            var price = 10 + 20 * x + (colour == "red" ? 5 : 0) + (random.NextDouble() - 0.5) * (1 + 6 * x);
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{x:F4},{random.NextDouble():F4},{colour},{price:F3}"));
        }
        var path = Path.Combine(_dir, "prices.csv");
        File.WriteAllLines(path, lines);
        return path;
    }

    [Fact]
    public async Task TheMLoopSearch_GivesARegressionResultOfTheSameShape()
    {
        var csv = WriteRegressionCsv();

        var mloop = await TrainAsync(csv, SearchEngine.MLoop, label: "price", task: "regression", metric: "r_squared");
        var automl = await TrainAsync(csv, SearchEngine.AutoML, label: "price", task: "regression", metric: "r_squared");

        _output.WriteLine($"AutoML: {automl.Trainer.Name} r_squared {automl.Metrics["r_squared"]:F4} · {automl.Trials.Count} trials");
        _output.WriteLine($"MLoop:  {mloop.Trainer.Name} r_squared {mloop.Metrics["r_squared"]:F4} · {mloop.Trials.Count} trials");

        Assert.Contains(mloop.Trainer.Name, MLoop.Search.Candidates.CandidateCatalog.Regression().Select(c => c.Name));
        Assert.Equal("r_squared", mloop.RankingMetric);
        // The same metrics as AutoML's run — the evaluator's, the prediction intervals and, when it fits, the σ-model's.
        Assert.Equal(automl.Metrics.Keys.ToHashSet(), mloop.Metrics.Keys.ToHashSet());
        Assert.Equal(automl.ResidualModel is null, mloop.ResidualModel is null);
        Assert.All(mloop.Trials.Where(t => t.Error is null), t => Assert.Contains("r_squared", t.Metrics.Keys));
        Assert.True(mloop.Metrics["r_squared"] > 0.8, $"MLoop search r_squared {mloop.Metrics["r_squared"]}");
    }
}
