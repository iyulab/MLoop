using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// Weighted training reaches the model: the search gives the trainers the weights and scores its folds by
/// them — on a fixture where the weight alone decides the answer, an unweighted fit cannot tell the classes
/// apart and a weighted one can.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class WeightedTrainingTests : IDisposable
{
    private readonly string _tmpDir = Path.Combine(Path.GetTempPath(), $"mloop_wt_{Guid.NewGuid():N}");

    public WeightedTrainingTests() => Directory.CreateDirectory(_tmpDir);

    public void Dispose()
    {
        if (Directory.Exists(_tmpDir))
            Directory.Delete(_tmpDir, recursive: true);
    }

    private sealed class Row
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float W { get; set; }
        public bool Label { get; set; }
    }

    // Each x appears twice, once per label, so unweighted P(true | x) = 0.5 everywhere. The weight
    // makes "true" ten times as frequent above 0.5 and ten times as rare below.
    private static List<Row> Fixture()
    {
        var rng = new Random(11);
        var rows = new List<Row>();
        for (var i = 0; i < 1000; i++)
        {
            var x = (float)rng.NextDouble();
            var y = (float)rng.NextDouble();
            var high = x > 0.5f;
            rows.Add(new Row { X = x, Y = y, W = high ? 10f : 1f, Label = true });
            rows.Add(new Row { X = x, Y = y, W = high ? 1f : 10f, Label = false });
        }
        return rows;
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

    private async Task<(float High, float Low, Dictionary<string, double> Metrics)> TrainAndScore(string? weightColumn)
    {
        var (high, low, result) = await TrainAndScoreOn(weightColumn);
        return (high, low, result.Metrics);
    }

    private async Task<(float High, float Low, TrainedModel Result)> TrainAndScoreOn(string? weightColumn)
    {
        var ctx = new MLContext(seed: 42);
        var data = ctx.Data.LoadFromEnumerable(Fixture());
        var config = new TrainingConfig
        {
            ModelName = "w",
            DataFile = "synthetic",
            LabelColumn = "Label",
            Task = "binary-classification",
            TimeLimitSeconds = 60,
            MaxTrials = 10,
            Metric = "auc",
            WeightColumn = weightColumn,
            // The control keeps W out of the features too, so the only difference is the weighting.
            ColumnOverrides = weightColumn is null ? new() { ["W"] = "ignore" } : null,
        };

        var result = await new TrainingRunner(ctx, new InMemoryDataProvider(ctx, data)).RunAsync(config);

        var probe = ctx.Data.LoadFromEnumerable(new[]
        {
            new Row { X = 0.9f, Y = 0.5f, W = 0, Label = true },
            new Row { X = 0.1f, Y = 0.5f, W = 0, Label = true },
        });
        var probabilities = result.Model.Transform(probe).GetColumn<float>("Probability").ToArray();
        return (probabilities[0], probabilities[1], result);
    }

    [Fact]
    public async Task TheWeightReachesTheModel()
    {
        var (high, low, metrics) = await TrainAndScore("W");

        Assert.True(high > 0.7f, $"P(true | x=0.9) = {high}, expected the weighted 0.9");
        Assert.True(low < 0.3f, $"P(true | x=0.1) = {low}, expected the weighted 0.1");
        // Judged on the weighted test rows: unweighted, every x carries both labels and no model
        // scores above 0.5; weighted, the right side of 0.5 is right ten times in eleven.
        Assert.True(metrics["accuracy"] > 0.8, $"weighted accuracy {metrics["accuracy"]}");
        Assert.True(metrics["auc"] > 0.8, $"weighted auc {metrics["auc"]}");
    }

    /// <summary>
    /// The folds are scored by the weights too: unweighted, every fold's AUC would sit near 0.5 on this
    /// fixture.
    /// </summary>
    [Fact]
    public async Task TheFoldsAreScoredByTheWeight()
    {
        var (high, low, result) = await TrainAndScoreOn("W");

        Assert.True(high > 0.7f, $"P(true | x=0.9) = {high}, expected the weighted 0.9");
        Assert.True(low < 0.3f, $"P(true | x=0.1) = {low}, expected the weighted 0.1");
        Assert.True(result.Metrics["auc"] > 0.8, $"weighted test auc {result.Metrics["auc"]}");
        Assert.NotNull(result.Search);
        var best = result.Trials.Where(t => t.Error is null).Max(t => t.Metrics["auc"]);
        Assert.True(best > 0.8, $"best fold-mean auc {best}, expected weighted scoring");
    }

    [Fact]
    public async Task WithoutTheWeight_TheSameDataTellsNothing()
    {
        var (high, low, _) = await TrainAndScore(null);

        // Unweighted, every x carries both labels, so no model can do much better than 0.5; the
        // search still fits noise (0.701 measured on a CI runner), so the band is about not having
        // learned the weighted 0.9 / 0.1, not about hitting 0.5.
        Assert.InRange(high, 0.25f, 0.75f);
        Assert.InRange(low, 0.25f, 0.75f);
    }

    [Fact]
    public async Task ATaskTheWeightsWouldNotReach_IsRefused()
    {
        var ctx = new MLContext(seed: 42);
        var data = ctx.Data.LoadFromEnumerable(Fixture());
        var config = new TrainingConfig
        {
            ModelName = "w",
            DataFile = "synthetic",
            LabelColumn = "",
            Task = "clustering",
            TimeLimitSeconds = 5,
            WeightColumn = "W",
        };

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => new TrainingRunner(ctx, new InMemoryDataProvider(ctx, data)).RunAsync(config));
        Assert.Contains("would ignore the weights", ex.Message);
    }
}

public class WeightedResamplingTests
{
    private sealed class R
    {
        public float V { get; set; }
        public float W { get; set; }
        public string T { get; set; } = "";
    }

    [Theory]
    [InlineData(-1f, "a negative")]
    [InlineData(float.NaN, "missing or non-finite")]
    public void ABadWeightIsNamed(float bad, string expected)
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(new[] { new R { V = 1, W = 1 }, new R { V = 2, W = bad } });

        var ex = Assert.Throws<InvalidOperationException>(() => RowWeights.Read(data, "W"));
        Assert.Contains(expected, ex.Message);
        Assert.Contains("row 2", ex.Message);
    }

    [Fact]
    public void AllZeroWeights_MissingColumn_AndNonNumericText_AreRefused()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(new[] { new R { V = 1, W = 0, T = "x" } });

        Assert.Contains("is 0", Assert.Throws<InvalidOperationException>(() => RowWeights.Read(data, "W")).Message);
        Assert.Contains("not in the data", Assert.Throws<InvalidOperationException>(() => RowWeights.Read(data, "Q")).Message);
        // A text column whose values are not numbers: named with the row, like any missing weight.
        Assert.Contains("row 1", Assert.Throws<InvalidOperationException>(() => RowWeights.Read(data, "T")).Message);
    }

    [Fact]
    public void NumbersReadAsText_AreWeights()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(new[] { new R { T = "2.5" }, new R { T = "0" } });

        Assert.Equal(new[] { 2.5, 0 }, RowWeights.Read(data, "T"));
    }

    [Fact]
    public void TheWeightIsNeverAFeature()
    {
        var config = new TrainingConfig
        {
            ModelName = "m",
            DataFile = "d",
            LabelColumn = "y",
            Task = "regression",
            WeightColumn = "w",
            ColumnOverrides = new() { ["note"] = "text" },
        };

        Assert.Equal("ignore", config.ColumnOverrides!["W"]);  // case-insensitive, as the schema matches it
        Assert.Equal("text", config.ColumnOverrides["note"]);
        // Not a feature, but claimed for a role: no exclusion heuristic may take it out of the data —
        // a weight column of mostly 1s is "constant" to a rule that judges features.
        Assert.Contains("w", config.ClaimedColumns);
    }
}
