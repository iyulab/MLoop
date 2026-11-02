using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;
using MLoop.Core.Training;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// A run asked for a blend (<see cref="TrainingConfig.Ensemble"/>) answers with one chain of the best trials when the blend
/// scores better out of fold, saves and reads it back as one model, and keeps the single model when it was not asked for.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class EnsembleTrainingTests
{
    private sealed class BinaryRow
    {
        public float X1 { get; set; }
        public float X2 { get; set; }
        public float X3 { get; set; }
        public float X4 { get; set; }
        public bool Label { get; set; }
    }

    private sealed class RegressionRow
    {
        public float X1 { get; set; }
        public float X2 { get; set; }
        public float X3 { get; set; }
        public float X4 { get; set; }
        public float Label { get; set; }
    }

    // A noisy rule on a few columns: no single trainer is the whole answer, so the best few trials differ in how they are wrong.
    private static List<BinaryRow> NoisyBinary(int count = 900)
    {
        var random = new Random(11);
        return [.. Enumerable.Range(0, count).Select(_ =>
        {
            float X() => (float)(random.NextDouble() * 2 - 1);
            var (x1, x2, x3, x4) = (X(), X(), X(), X());
            var logit = 1.6 * x1 - 1.2 * x2 + 1.5 * x3 * x4 + 0.8 * Math.Sin(3 * x1) + (random.NextDouble() - 0.5) * 2.5;
            return new BinaryRow { X1 = x1, X2 = x2, X3 = x3, X4 = x4, Label = logit > 0 };
        })];
    }

    private static List<RegressionRow> NoisyRegression(int count = 900)
    {
        var random = new Random(13);
        return [.. Enumerable.Range(0, count).Select(_ =>
        {
            float X() => (float)(random.NextDouble() * 2 - 1);
            var (x1, x2, x3, x4) = (X(), X(), X(), X());
            return new RegressionRow
            {
                X1 = x1, X2 = x2, X3 = x3, X4 = x4,
                Label = (float)(4 * x1 - 3 * x2 + 3 * x3 * x4 + 2 * Math.Sin(3 * x1) + (random.NextDouble() - 0.5) * 6),
            };
        })];
    }

    private sealed class InMemoryDataProvider(MLContext ctx, IDataView data) : IDataProvider
    {
        public IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
            IEnumerable<string>? preserveColumns = null, IReadOnlyCollection<string>? featureExclusions = null, DataKind? labelKind = null) => data;
        public bool ValidateLabelColumn(IDataView d, string labelColumn) => d.Schema.GetColumnOrNull(labelColumn).HasValue;
        public DataSchema GetSchema(IDataView d) => new() { Columns = [], RowCount = 0 };
        public (IDataView trainSet, IDataView testSet) SplitData(IDataView d, double testFraction = 0.2)
        {
            var split = ctx.Data.TrainTestSplit(d, testFraction: testFraction, seed: 42);
            return (split.TrainSet, split.TestSet);
        }
    }

    private static TrainingConfig Config(string task, string metric, bool ensemble) => new()
    {
        ModelName = "e",
        DataFile = "synthetic",
        LabelColumn = "Label",
        Task = task,
        TimeLimitSeconds = 120,
        Metric = metric,
        MaxTrials = 16,
        Ensemble = ensemble,
    };

    [Fact]
    public async Task ABinaryRun_WithEnsemble_AnswersWithOneChainOfTheBestTrials()
    {
        var ctx = new MLContext(seed: 42);
        var data = ctx.Data.LoadFromEnumerable(NoisyBinary());
        var runner = new TrainingRunner(ctx, new InMemoryDataProvider(ctx, data));

        var result = await runner.RunAsync(Config("binary-classification", "auc", ensemble: true));

        Assert.Equal("Ensemble", result.Trainer.Name);
        Assert.Contains("×", result.Trainer.Parameters!.Single(p => p.Key == "members").Value);
        Assert.NotNull(result.Calibration);
        var scored = result.ScoringModel(ctx).Transform(ctx.Data.TrainTestSplit(data, 0.2, seed: 42).TestSet);
        Assert.Contains(scored.Schema, c => c.Name == "Probability");
        Assert.True(result.Metrics["auc"] > 0.5);
        // The blend's probabilities are calibrated on the blend's own out-of-fold answers (a single trial's mapping is fitted
        // to that trial's probabilities, not to their mean): on the test rows the calibrated blend is no worse than the raw one.
        var testRows = ctx.Data.TrainTestSplit(data, 0.2, seed: 42).TestSet;
        var own = CalibrationError.Binary(result.Model.Transform(testRows), "Label")!.Value;
        var calibrated = CalibrationError.Binary(result.ScoringModel(ctx).Transform(testRows), "Label")!.Value;
        Assert.True(calibrated <= own + 1e-9, $"ECE {own} → {calibrated}");
        Assert.Equal(calibrated, result.Metrics[CalibrationError.MetricName], 6);
    }

    [Fact]
    public async Task ABinaryEnsemble_SavesAsOneModelFile_AndReadsBackToTheSameProbabilities()
    {
        var ctx = new MLContext(seed: 42);
        var data = ctx.Data.LoadFromEnumerable(NoisyBinary());
        var result = await new TrainingRunner(ctx, new InMemoryDataProvider(ctx, data)).RunAsync(Config("binary-classification", "auc", ensemble: true));
        Assert.Equal("Ensemble", result.Trainer.Name);
        var test = ctx.Data.TrainTestSplit(data, 0.2, seed: 42).TestSet;
        var path = Path.Combine(Path.GetTempPath(), $"ensemble-run-{Guid.NewGuid():N}.zip");
        try
        {
            ctx.Model.Save(result.Model, data.Schema, path);
            var fresh = new MLContext();
            var loaded = fresh.Model.Load(path, out _);

            var before = result.Model.Transform(test).GetColumn<float>("Probability").ToArray();
            var after = loaded.Transform(test).GetColumn<float>("Probability").ToArray();
            Assert.Equal(before, after);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task ARegressionRun_WithEnsemble_AnswersWithOneChainOfTheBestTrials()
    {
        var ctx = new MLContext(seed: 42);
        var data = ctx.Data.LoadFromEnumerable(NoisyRegression());
        var runner = new TrainingRunner(ctx, new InMemoryDataProvider(ctx, data));

        var result = await runner.RunAsync(Config("regression", "rmse", ensemble: true));

        Assert.Equal("Ensemble", result.Trainer.Name);
        var scored = result.Model.Transform(ctx.Data.TrainTestSplit(data, 0.2, seed: 42).TestSet);
        Assert.Contains(scored.Schema, c => c.Name == "Score");
        Assert.True(double.IsFinite(result.Metrics["rmse"]));
    }

    [Fact]
    public async Task WithoutTheRequest_TheRunAnswersWithTheSingleBestModel()
    {
        var ctx = new MLContext(seed: 42);
        var data = ctx.Data.LoadFromEnumerable(NoisyBinary());

        var result = await new TrainingRunner(ctx, new InMemoryDataProvider(ctx, data)).RunAsync(Config("binary-classification", "auc", ensemble: false));

        Assert.NotEqual("Ensemble", result.Trainer.Name);
    }

    [Fact]
    public async Task AnotherTask_IgnoresTheRequest()
    {
        var ctx = new MLContext(seed: 42);
        var rows = NoisyBinary().Select(r => new { r.X1, r.X2, r.X3, r.X4, Label = r.X1 > 0.3f ? "a" : r.X2 > 0 ? "b" : "c" });
        var data = ctx.Data.LoadFromEnumerable(rows.Select(r => new MulticlassRow { X1 = r.X1, X2 = r.X2, X3 = r.X3, X4 = r.X4, Label = r.Label }));

        var result = await new TrainingRunner(ctx, new InMemoryDataProvider(ctx, data)).RunAsync(Config("multiclass-classification", "macro_accuracy", ensemble: true));

        Assert.NotEqual("Ensemble", result.Trainer.Name);
    }

    private sealed class MulticlassRow
    {
        public float X1 { get; set; }
        public float X2 { get; set; }
        public float X3 { get; set; }
        public float X4 { get; set; }
        public string Label { get; set; } = "";
    }
}
