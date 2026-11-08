using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Contracts;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Training;
using MLoop.Tests.Common;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// What a run reports about held-out rows has to mean what it says. Line coverage cannot see this: the
/// defects found in the selection and reporting paths (a ranking trainer chosen on the test partition, a cluster count
/// chosen on the number it is reported by, a refit that did not reproduce the score that chose it) all sat on lines
/// the suites already executed. These tests pin the reported number against a fact about the data instead of against
/// an earlier output:
/// <list type="bullet">
/// <item>Labels with no relation to the features cannot score above chance on held-out rows, beyond sampling error
/// (a run that tuned on its own test partition, or read the label through a feature, would).</item>
/// <item>The order of the rows in a file is not information: a file sorted by label scores like a shuffled one.</item>
/// <item>The same data, configuration and seed give the same reported metrics.</item>
/// </list>
/// What these cannot see: a reported partition that overlaps the fitted rows. Tried, and the search settled on
/// regularised models that score noise at chance in-sample too (0.33 against a 0.39 bound), so the bound stayed
/// quiet; disjointness needs a hook on the partitions themselves, not a score.
/// The bounds are four standard errors of the quantity being bounded, so a pass says "indistinguishable from chance",
/// and a failure says a leak or an order dependence, not bad luck: the data and seeds are fixed, so the outcome is the
/// same on every run.
/// </summary>
[Collection("FileSystem")]
[Trait(TestCategories.Category, TestCategories.Integration)]
public class StatisticalInvariantTests
{
    private const int Rows = 1500;
    private const int Seed = 42;

    private sealed class MulticlassRow
    {
        public float F0 { get; set; }
        public float F1 { get; set; }
        public float F2 { get; set; }
        public float F3 { get; set; }
        public float F4 { get; set; }
        public string Label { get; set; } = "";
    }

    private sealed class BinaryRow
    {
        public float F0 { get; set; }
        public float F1 { get; set; }
        public float F2 { get; set; }
        public float F3 { get; set; }
        public float F4 { get; set; }
        public bool Label { get; set; }
    }

    private sealed class RegressionRow
    {
        public float F0 { get; set; }
        public float F1 { get; set; }
        public float F2 { get; set; }
        public float F3 { get; set; }
        public float F4 { get; set; }
        public float Label { get; set; }
    }

    private sealed class InMemoryDataProvider(MLContext ctx, IDataView data) : IDataProvider
    {
        public int TestRows { get; private set; }

        public IDataView LoadData(string filePath, string? labelColumn = null, string? taskType = null,
            IEnumerable<string>? preserveColumns = null, IReadOnlyCollection<string>? featureExclusions = null, DataKind? labelKind = null) => data;
        public bool ValidateLabelColumn(IDataView d, string labelColumn) => d.Schema.GetColumnOrNull(labelColumn).HasValue;
        public DataSchema GetSchema(IDataView d) => new() { Columns = [], RowCount = 0 };
        public (IDataView trainSet, IDataView testSet) SplitData(IDataView d, double testFraction = 0.2)
        {
            var split = ctx.Data.TrainTestSplit(d, testFraction: testFraction, seed: Seed);
            TestRows = (int)split.TestSet.GetColumn<float>("F0").LongCount();
            return (split.TrainSet, split.TestSet);
        }
    }

    private static float Noise(Random random) => (float)(random.NextDouble() * 2 - 1);

    // Labels drawn without looking at the features.
    private static List<MulticlassRow> NoiseMulticlass(int seed)
    {
        var random = new Random(seed);
        string[] labels = ["p", "q", "r"];
        return [.. Enumerable.Range(0, Rows).Select(_ => new MulticlassRow
        {
            F0 = Noise(random), F1 = Noise(random), F2 = Noise(random), F3 = Noise(random), F4 = Noise(random),
            Label = labels[random.Next(labels.Length)],
        })];
    }

    private static List<BinaryRow> NoiseBinary(int seed)
    {
        var random = new Random(seed);
        return [.. Enumerable.Range(0, Rows).Select(_ => new BinaryRow
        {
            F0 = Noise(random), F1 = Noise(random), F2 = Noise(random), F3 = Noise(random), F4 = Noise(random),
            Label = random.Next(2) == 1,
        })];
    }

    private static List<RegressionRow> NoiseRegression(int seed)
    {
        var random = new Random(seed);
        return [.. Enumerable.Range(0, Rows).Select(_ => new RegressionRow
        {
            F0 = Noise(random), F1 = Noise(random), F2 = Noise(random), F3 = Noise(random), F4 = Noise(random),
            Label = (float)(10 + 3 * (random.NextDouble() * 2 - 1)),
        })];
    }

    // The class follows F0 in three bands (with a few flipped), so a model can learn it well — the question is only
    // whether the order of the rows in the file changes what is reported.
    private static List<MulticlassRow> BandedMulticlass(int seed)
    {
        var random = new Random(seed);
        string[] labels = ["low", "mid", "high"];
        return [.. Enumerable.Range(0, Rows).Select(_ =>
        {
            var f0 = Noise(random);
            var band = f0 < -0.33f ? 0 : f0 < 0.33f ? 1 : 2;
            if (random.NextDouble() < 0.05) band = random.Next(3);
            return new MulticlassRow
            {
                F0 = f0, F1 = Noise(random), F2 = Noise(random), F3 = Noise(random), F4 = Noise(random), Label = labels[band],
            };
        })];
    }

    // The same noise, with the label written into a feature: the shape of a leaked column. A detector that cannot
    // see this proves nothing by staying quiet on the honest rows.
    private static List<MulticlassRow> LeakedMulticlass(int seed)
    {
        var rows = NoiseMulticlass(seed);
        foreach (var row in rows)
            row.F0 = row.Label switch { "p" => -0.8f, "q" => 0f, _ => 0.8f };
        return rows;
    }

    private static List<RegressionRow> LeakedRegression(int seed)
    {
        var rows = NoiseRegression(seed);
        foreach (var row in rows)
            row.F0 = (row.Label - 10) / 3;
        return rows;
    }

    // Predicting the commonest class is the best a model with nothing to learn from can do; four standard errors of a
    // proportion on the test partition above it.
    private static double MulticlassChanceBound(IReadOnlyCollection<MulticlassRow> rows, int testRows)
    {
        var share = rows.GroupBy(r => r.Label).Max(g => g.Count()) / (double)rows.Count;
        return share + 4 * Math.Sqrt(share * (1 - share) / testRows);
    }

    // A prediction unrelated to the label explains at most their sample correlation squared, which is chi-square(1) / n:
    // four standard errors of the correlation is 16 / n.
    private static double RegressionChanceBound(int testRows) => 16.0 / testRows;

    private async Task<(TrainedModel Result, int TestRows)> TrainAsync<T>(List<T> rows, string task, int maxTrials = 4)
        where T : class, new()
    {
        var ctx = new MLContext(seed: Seed);
        var provider = new InMemoryDataProvider(ctx, ctx.Data.LoadFromEnumerable(rows));
        var result = await new TrainingRunner(ctx, provider).RunAsync(new TrainingConfig
        {
            ModelName = "invariant",
            DataFile = "synthetic",
            LabelColumn = "Label",
            Task = task,
            TimeLimitSeconds = 60,
            MaxTrials = maxTrials,
        });
        return (result, provider.TestRows);
    }

    [Fact]
    public async Task MulticlassLabelsDrawnWithoutTheFeatures_ScoreAtChanceOnTheTestPartition()
    {
        var rows = NoiseMulticlass(seed: 1);
        var (result, testRows) = await TrainAsync(rows, "multiclass-classification");

        var bound = MulticlassChanceBound(rows, testRows);
        Assert.True(result.Metrics["micro_accuracy"] <= bound,
            $"micro_accuracy {result.Metrics["micro_accuracy"]:0.0000} on {testRows} test rows of labels with no relation to the features; chance bound {bound:0.0000}");
    }

    [Fact]
    public async Task BinaryLabelsDrawnWithoutTheFeatures_RankAtChanceOnTheTestPartition()
    {
        var rows = NoiseBinary(seed: 2);
        var (result, testRows) = await TrainAsync(rows, "binary-classification");

        // Under no relation the AUC is 0.5 with standard error sqrt((n1 + n2 + 1) / (12 n1 n2)) (Hanley-McNeil's null case).
        var positives = (int)Math.Round(testRows * rows.Count(r => r.Label) / (double)rows.Count);
        var negatives = testRows - positives;
        var bound = 0.5 + 4 * Math.Sqrt((positives + negatives + 1.0) / (12.0 * positives * negatives));
        Assert.True(result.Metrics["auc"] <= bound,
            $"auc {result.Metrics["auc"]:0.0000} on {testRows} test rows of labels with no relation to the features; chance bound {bound:0.0000}");
    }

    [Fact]
    public async Task ANumberDrawnWithoutTheFeatures_IsExplainedNoBetterThanSamplingErrorAllows()
    {
        var rows = NoiseRegression(seed: 3);
        var (result, testRows) = await TrainAsync(rows, "regression");

        var bound = RegressionChanceBound(testRows);
        Assert.True(result.Metrics["r_squared"] <= bound,
            $"r_squared {result.Metrics["r_squared"]:0.0000} on {testRows} test rows of a label with no relation to the features; bound {bound:0.0000}");
    }

    [Fact]
    public async Task AFileSortedByLabelScoresLikeTheSameRowsShuffled()
    {
        var shuffled = BandedMulticlass(seed: 4);
        var sorted = shuffled.OrderBy(r => r.Label, StringComparer.Ordinal).ToList();

        var (fromShuffled, testRows) = await TrainAsync(shuffled, "multiclass-classification");
        var (fromSorted, _) = await TrainAsync(sorted, "multiclass-classification");

        // Two estimates of one accuracy on different test rows differ by a standard error times root two; four of those.
        var p = fromShuffled.Metrics["micro_accuracy"];
        var bound = 4 * Math.Sqrt(2 * Math.Max(p * (1 - p), 0.01) / testRows);
        Assert.True(fromShuffled.Metrics["micro_accuracy"] > 0.8, $"the banded rows are learnable: {fromShuffled.Metrics["micro_accuracy"]:0.0000}");
        Assert.True(Math.Abs(fromShuffled.Metrics["micro_accuracy"] - fromSorted.Metrics["micro_accuracy"]) <= bound,
            $"shuffled {fromShuffled.Metrics["micro_accuracy"]:0.0000} vs sorted {fromSorted.Metrics["micro_accuracy"]:0.0000} (bound {bound:0.0000})");
    }

    [Fact]
    public async Task TheSameDataConfigurationAndSeed_ReportTheSameMetrics()
    {
        var rows = BandedMulticlass(seed: 5);

        var (first, _) = await TrainAsync(rows, "multiclass-classification");
        var (second, _) = await TrainAsync(rows, "multiclass-classification");

        foreach (var (name, value) in first.Metrics)
            Assert.True(second.Metrics.TryGetValue(name, out var again) && value.Equals(again),
                $"{name}: {value:R} then {(second.Metrics.TryGetValue(name, out var v) ? v.ToString("R") : "absent")}");
    }

    [Fact]
    public async Task TheMulticlassChanceBoundSeesALabelWrittenIntoAFeature()
    {
        var rows = LeakedMulticlass(seed: 1);
        var (result, testRows) = await TrainAsync(rows, "multiclass-classification");

        Assert.True(result.Metrics["micro_accuracy"] > MulticlassChanceBound(rows, testRows),
            $"a leaked label scored {result.Metrics["micro_accuracy"]:0.0000}; the bound {MulticlassChanceBound(rows, testRows):0.0000} would not have noticed");
    }

    [Fact]
    public async Task TheRegressionChanceBoundSeesALabelWrittenIntoAFeature()
    {
        var rows = LeakedRegression(seed: 3);
        var (result, testRows) = await TrainAsync(rows, "regression");

        Assert.True(result.Metrics["r_squared"] > RegressionChanceBound(testRows),
            $"a leaked label scored {result.Metrics["r_squared"]:0.0000}; the bound {RegressionChanceBound(testRows):0.0000} would not have noticed");
    }
}
