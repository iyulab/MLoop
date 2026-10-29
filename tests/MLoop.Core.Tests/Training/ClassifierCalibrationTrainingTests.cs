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
/// A classifier run's probabilities are recalibrated on the search's out-of-fold answers (binary: Platt; multiclass: a temperature): on rows a few shallow trees
/// already rank perfectly, FastTree's own probability is a fixed sigmoid that says about 0.66, and the run reports
/// — and saves — what the calibrated model says, fitted on no test row.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class ClassifierCalibrationTrainingTests
{
    private sealed class Row
    {
        public float X { get; set; }
        public float Noise { get; set; }
        public bool Label { get; set; }
    }

    // The classes sit apart on X: every model ranks the rows right, and how sure it says it is is the only question.
    private static List<Row> Separable()
    {
        var random = new Random(7);
        return [.. Enumerable.Range(0, 600).Select(_ =>
        {
            var positive = random.Next(2) == 1;
            return new Row { X = (float)(random.NextDouble() * 0.4 + (positive ? 0.6 : 0)), Noise = (float)random.NextDouble(), Label = positive };
        })];
    }

    private sealed class InMemoryDataProvider(MLContext ctx, IDataView data) : IDataProvider
    {
        public IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
            IEnumerable<string>? preserveColumns = null, IReadOnlyCollection<string>? featureExclusions = null, Microsoft.ML.Data.DataKind? labelKind = null) => data;
        public bool ValidateLabelColumn(IDataView d, string labelColumn) => d.Schema.GetColumnOrNull(labelColumn).HasValue;
        public DataSchema GetSchema(IDataView d) => new() { Columns = [], RowCount = 0 };
        public (IDataView trainSet, IDataView testSet) SplitData(IDataView d, double testFraction = 0.2)
        {
            var split = ctx.Data.TrainTestSplit(d, testFraction: testFraction, seed: 42);
            return (split.TrainSet, split.TestSet);
        }
    }

    [Fact]
    public async Task ABinaryRun_IsCalibratedOnItsOutOfFoldAnswers_AndReportsTheCalibratedModel()
    {
        var ctx = new MLContext(seed: 42);
        var data = ctx.Data.LoadFromEnumerable(Separable());
        var (trainSet, testSet) = new InMemoryDataProvider(ctx, data).SplitData(data);
        var config = new TrainingConfig
        {
            ModelName = "c",
            DataFile = "synthetic",
            LabelColumn = "Label",
            Task = "binary-classification",
            TimeLimitSeconds = 30,
            Metric = "accuracy",
            MaxTrials = 3,
        };

        var result = await new TrainingRunner(ctx, new InMemoryDataProvider(ctx, data)).RunAsync(config);

        var calibration = Assert.IsType<ProbabilityCalibration>(result.Calibration);
        Assert.Equal("platt", calibration.Method);
        Assert.Equal(TrainingRunner.OutOfFoldSource, calibration.FitOn);
        // Every row of the training partition, each answered by the fold model that had not seen it.
        Assert.Equal(trainSet.GetColumn<bool>("Label").Count(), calibration.Rows);
        // The model's own probabilities are off on this fixture; the calibrated ones are not — and the run reports those.
        var own = CalibrationError.Binary(result.Model.Transform(testSet), "Label")!.Value;
        var calibrated = CalibrationError.Binary(result.ScoringModel(ctx).Transform(testSet), "Label")!.Value;
        Assert.True(calibrated < own, $"ECE {own} → {calibrated}");
        Assert.Equal(calibrated, result.Metrics[CalibrationError.MetricName], 6);
        // The answers the run counts are the calibrated model's.
        Assert.Equal(BinaryCounts.Metrics(result.ScoringModel(ctx).Transform(testSet), "Label")!["accuracy"], result.Metrics["accuracy"], 6);
    }

    [Theory]
    [InlineData("auto", "auc")]
    [InlineData("accuracy", "accuracy")]
    public async Task ABinarySearch_OptimizesASmoothScoreUnlessAMetricWasNamed(string metric, string expected)
    {
        var ctx = new MLContext(seed: 42);
        var data = ctx.Data.LoadFromEnumerable(Separable());
        var config = new TrainingConfig
        {
            ModelName = "o",
            DataFile = "synthetic",
            LabelColumn = "Label",
            Task = "binary-classification",
            TimeLimitSeconds = 30,
            Metric = metric,
            MaxTrials = 3,
        };

        var result = await new TrainingRunner(ctx, new InMemoryDataProvider(ctx, data)).RunAsync(config);

        Assert.Equal(expected, result.RankingMetric);
    }

    private sealed class ClassRow
    {
        public float X { get; set; }
        public float Noise { get; set; }
        public string Label { get; set; } = "";
    }

    [Fact]
    public async Task AMulticlassRun_IsTemperedOnItsOutOfFoldAnswers_AndKeepsItsAnswers()
    {
        var ctx = new MLContext(seed: 42);
        var random = new Random(3);
        var rows = Enumerable.Range(0, 600).Select(_ =>
        {
            var x = (float)random.NextDouble();
            var label = random.NextDouble() < 0.15 ? "abc"[random.Next(3)].ToString() : x < 0.33 ? "a" : x < 0.66 ? "b" : "c";
            return new ClassRow { X = x, Noise = (float)random.NextDouble(), Label = label };
        }).ToList();
        var data = ctx.Data.LoadFromEnumerable(rows);
        var config = new TrainingConfig
        {
            ModelName = "m",
            DataFile = "synthetic",
            LabelColumn = "Label",
            Task = "multiclass-classification",
            TimeLimitSeconds = 30,
            Metric = "macro_accuracy",
            MaxTrials = 3,
        };

        var result = await new TrainingRunner(ctx, new InMemoryDataProvider(ctx, data)).RunAsync(config);

        var calibration = Assert.IsType<ProbabilityCalibration>(result.Calibration);
        Assert.Equal("temperature", calibration.Method);
        Assert.Equal(TrainingRunner.OutOfFoldSource, calibration.FitOn);
        // A temperature minimizes log-loss, not ECE, so on a model already calibrated to within a few hundredths
        // it can move ECE either way; it must not leave a model badly calibrated, which is what it exists to fix.
        Assert.True(calibration.EceAfter <= Math.Max(calibration.EceBefore ?? 0, 0.05), $"ECE {calibration.EceBefore} → {calibration.EceAfter}");
        // A temperature keeps every row's answer.
        var (_, testSet) = new InMemoryDataProvider(ctx, data).SplitData(data);
        var own = result.Model.Transform(testSet).GetColumn<string>("PredictedLabel").ToArray();
        var tempered = result.ScoringModel(ctx).Transform(testSet).GetColumn<string>("PredictedLabel").ToArray();
        Assert.Equal(own, tempered);
    }
}
