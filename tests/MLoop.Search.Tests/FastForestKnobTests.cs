using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Search.Candidates;
using MLoop.Search.Tuning;

namespace MLoop.Search.Tests;

/// <summary>
/// The two settings of a forest that its search space exposes beyond the trees, the leaves and the per-tree feature
/// fraction — the minimum examples per leaf and the fraction of features tried at each split — reach the trainer in
/// every FastForest candidate (binary, one-versus-all and regression). A setting the space offers but a builder drops
/// is a dimension the search walks without effect.
/// </summary>
public sealed class FastForestKnobTests
{
    private const int Informative = 1;
    private const int Noise = 7;

    private sealed class ValueRow
    {
        public float Label { get; set; }

        [VectorType(Informative + Noise)]
        public float[] Features { get; set; } = [];
    }

    private sealed class ClassRow
    {
        public string Label { get; set; } = "";

        [VectorType(Informative + Noise)]
        public float[] Features { get; set; } = [];
    }

    private sealed class FlagRow
    {
        public bool Label { get; set; }

        [VectorType(Informative + Noise)]
        public float[] Features { get; set; } = [];
    }

    // One informative column and seven of noise, and a label that is the informative column plus noise of its own:
    // a forest grown to purity memorizes the noise and one held to many examples per leaf cannot.
    private static float[] Features(Random random)
        => [.. Enumerable.Range(0, Informative + Noise).Select(_ => (float)random.NextDouble())];

    private static SearchPoint Forest(TrainerCandidate candidate, int minLeaf, double splitFraction, int leaves = 256)
        => candidate.Space.PointOf(new Dictionary<string, double>
        {
            ["NumberOfTrees"] = 30,
            ["NumberOfLeaves"] = leaves,
            ["FeatureFraction"] = 1,
            ["MinimumExampleCountPerLeaf"] = minLeaf,
            ["FeatureFractionPerSplit"] = splitFraction,
        });

    // The score every row gets from a forest built at a point (the vector of a one-versus-all score is flattened).
    private static float[] Scores(TrainerCandidate candidate, SearchPoint point, IDataView data, MLContext ml)
    {
        var scored = candidate.Build(ml, point, new TrainerColumns("Label", "Features"), new SearchSettings()).Fit(data).Transform(data);
        return scored.Schema["Score"].Type is VectorDataViewType
            ? [.. scored.GetColumn<float[]>("Score").SelectMany(v => v)]
            : [.. scored.GetColumn<float>("Score")];
    }

    // The seed, the trees and the data are the same, so a difference in the answers is the setting reaching the trainer.
    private static double MeanDifference(float[] a, float[] b) => a.Zip(b, (x, y) => Math.Abs(x - y)).Average();

    private static double TrainingFit(TrainerCandidate candidate, SearchPoint point, IDataView data, MLContext ml, Func<IDataView, double> measure)
        => measure(candidate.Build(ml, point, new TrainerColumns("Label", "Features"), new SearchSettings()).Fit(data).Transform(data));

    [Fact]
    public void ARegressionForest_ReadsBothSettings()
    {
        var ml = new MLContext(seed: 1);
        var random = new Random(11);
        var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 400).Select(_ =>
        {
            var features = Features(random);
            return new ValueRow { Label = features[0] * 4 + (float)random.NextDouble() * 2, Features = features };
        }).ToList());
        var candidate = CandidateCatalog.Regression().Single(c => c.Name == "FastForest");
        double RSquared(IDataView scored) => ml.Regression.Evaluate(scored, "Label").RSquared;

        var grown = TrainingFit(candidate, Forest(candidate, minLeaf: 1, splitFraction: 1), data, ml, RSquared);
        var held = TrainingFit(candidate, Forest(candidate, minLeaf: 64, splitFraction: 1), data, ml, RSquared);

        Assert.True(grown > held + 0.1, $"minimum examples per leaf: grown to purity {grown:F3}, held to 64 {held:F3}");
        Assert.True(MeanDifference(Scores(candidate, Forest(candidate, 1, 1), data, ml), Scores(candidate, Forest(candidate, 1, 1.0 / (Informative + Noise)), data, ml)) > 0.05,
            "the fraction of features tried per split changed nothing");
    }

    [Fact]
    public void ABinaryForest_ReadsBothSettings()
    {
        var ml = new MLContext(seed: 1);
        var random = new Random(12);
        var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 400).Select(_ =>
        {
            var features = Features(random);
            return new FlagRow { Label = features[0] + random.NextDouble() * 1.5 > 1.2, Features = features };
        }).ToList());
        var candidate = CandidateCatalog.Binary().Single(c => c.Name == "FastForest");
        double Auc(IDataView scored) => ml.BinaryClassification.EvaluateNonCalibrated(scored).AreaUnderRocCurve;

        var grown = TrainingFit(candidate, Forest(candidate, minLeaf: 1, splitFraction: 1), data, ml, Auc);
        var held = TrainingFit(candidate, Forest(candidate, minLeaf: 64, splitFraction: 1), data, ml, Auc);

        Assert.True(grown > held + 0.05, $"minimum examples per leaf: grown to purity {grown:F3}, held to 64 {held:F3}");
        Assert.True(MeanDifference(Scores(candidate, Forest(candidate, 1, 1), data, ml), Scores(candidate, Forest(candidate, 1, 1.0 / (Informative + Noise)), data, ml)) > 0.01,
            "the fraction of features tried per split changed nothing");
    }

    [Fact]
    public void AOneVersusAllForest_ReadsBothSettings()
    {
        var ml = new MLContext(seed: 1);
        var random = new Random(13);
        string[] classes = ["low", "mid", "high"];
        var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 450).Select(_ =>
        {
            var features = Features(random);
            var level = features[0] * 3 + (float)random.NextDouble() * 1.5f;
            return new ClassRow { Label = classes[Math.Clamp((int)(level * 0.8), 0, 2)], Features = features };
        }).ToList());
        var candidate = CandidateCatalog.Multiclass().Single(c => c.Name == "FastForestOva");
        double Accuracy(IDataView scored) => ml.MulticlassClassification.Evaluate(scored, "Label").MicroAccuracy;

        var grown = TrainingFit(candidate, Forest(candidate, minLeaf: 1, splitFraction: 1), data, ml, Accuracy);
        var held = TrainingFit(candidate, Forest(candidate, minLeaf: 64, splitFraction: 1), data, ml, Accuracy);

        Assert.True(grown > held + 0.05, $"minimum examples per leaf: grown to purity {grown:F3}, held to 64 {held:F3}");
        // Small trees: a forest that memorizes its rows answers 0 or 1 whatever features it looked at.
        var difference = MeanDifference(Scores(candidate, Forest(candidate, 1, 1, leaves: 6), data, ml), Scores(candidate, Forest(candidate, 1, 1.0 / (Informative + Noise), leaves: 6), data, ml));
        Assert.True(difference > 0.01, $"the fraction of features tried per split changed nothing ({difference})");
    }

    [Fact]
    public void AForestStartsAtItsLibrarysDefaults_AndTheSearchCanMoveBothSettings()
    {
        // The start is the library's defaults (ten examples per leaf, seven tenths of the features at each split).
        var start = CandidateCatalog.Regression().Single(c => c.Name == "FastForest").StartingPoints.Last();
        Assert.Equal((10d, 0.7), (start["MinimumExampleCountPerLeaf"], Math.Round(start["FeatureFractionPerSplit"], 6)));

        foreach (var candidate in CandidateCatalog.Binary().Concat(CandidateCatalog.Multiclass()).Concat(CandidateCatalog.Regression()).Where(c => c.Name.StartsWith("FastForest", StringComparison.Ordinal)))
        {
            var names = candidate.Space.Dimensions.Select(d => d.Name).ToList();
            Assert.Contains("MinimumExampleCountPerLeaf", names);
            Assert.Contains("FeatureFractionPerSplit", names);
        }
    }
}
