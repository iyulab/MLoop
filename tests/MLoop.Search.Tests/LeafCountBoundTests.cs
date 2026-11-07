using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using MLoop.Search.Candidates;
using MLoop.Search.Tuning;
using MLoop.Search.Validation;

namespace MLoop.Search.Tests;

/// <summary>
/// A tree cannot have more leaves than the rows it is fitted on, so the leaf count a trainer is built with is bounded by them: the
/// trainer sizes its tree storage by the setting, and a forest of 78 trees of 32,768 leaves took 1.4 GB to fit three thousand rows.
/// The tree itself is the same, so the bound changes memory and time and nothing a trial scores.
/// </summary>
public sealed class LeafCountBoundTests
{
    private static SearchPoint ForestWith(double leaves, double trees = 78)
        => CandidateCatalog.FastForestSpace.PointOf(new Dictionary<string, double>
        {
            ["NumberOfLeaves"] = leaves,
            ["NumberOfTrees"] = trees,
            ["FeatureFraction"] = 0.5,
        });

    [Fact]
    public void ALeafCountPastTheRowsIsBoundedByThem_AndNothingElseMoves()
    {
        var point = ForestWith(leaves: 32768);

        var bounded = TrainerCandidate.FittedToRows(point, rows: 3000);

        Assert.Equal(3000, bounded.Int("NumberOfLeaves"));
        Assert.Equal(78, bounded.Int("NumberOfTrees"));
        Assert.Equal(0.5f, bounded.Single("FeatureFraction"));
        Assert.Equal(32768, point.Int("NumberOfLeaves"));   // the point the tuner proposed is not changed
    }

    [Fact]
    public void ALeafCountWithinTheRowsIsLeftAlone()
    {
        var point = ForestWith(leaves: 200);
        Assert.Same(point, TrainerCandidate.FittedToRows(point, rows: 3000));
        Assert.Same(point, TrainerCandidate.FittedToRows(point, rows: 200));
    }

    [Fact]
    public void ATrainerWithoutALeafCountIsLeftAlone()
    {
        var point = CandidateCatalog.LbfgsSpace.PointOf(new Dictionary<string, double> { ["L1Regularization"] = 1, ["L2Regularization"] = 1 });
        Assert.Same(point, TrainerCandidate.FittedToRows(point, rows: 10));
    }

    [Fact]
    public void FewerRowsThanTheSmallestLeafCountStillGiveTheSmallest()
    {
        // the dimensions start at 4 leaves, so a bound under that would build a trainer the space does not describe
        Assert.Equal(4, TrainerCandidate.FittedToRows(ForestWith(leaves: 32768), rows: 2).Int("NumberOfLeaves"));
    }

    [Fact]
    public void EveryTreeTrainerOfTheCatalogHasTheLeafCountTheBoundReads()
    {
        // the bound is by name; a tree trainer whose dimension were renamed would silently escape it
        var trees = CandidateCatalog.Regression().Concat(CandidateCatalog.Binary()).Concat(CandidateCatalog.Multiclass())
            .Where(c => c.Name is "FastTree" or "FastForest" or "LightGbm")
            .ToList();
        Assert.NotEmpty(trees);
        Assert.All(trees, c => Assert.Contains(c.Space.Dimensions, d => d.Name == "NumberOfLeaves"));
    }

    private sealed class Row
    {
        public bool Label { get; set; }
        public float X { get; set; }
        public float Z { get; set; }
    }

    private sealed record Auc(string Name = "auc", bool HigherIsBetter = true) : IMetricPolicy;

    private sealed class Sink : ITrialSink
    {
        public void Completed(TrialResult trial, string rankingMetric) { }
    }

    [Fact]
    public void ASearchBuildsNoTrainerWithMoreLeavesThanItHasRows_YetRecordsWhatTheTunerProposed()
    {
        var ml = new MLContext(seed: 1);
        var random = new Random(5);
        var rows = Enumerable.Range(0, 240).Select(_ =>
        {
            var x = (float)random.NextDouble();
            return new Row { X = x, Z = (float)random.NextDouble(), Label = x + random.NextDouble() * 0.4 > 0.7 };
        }).ToList();
        var data = ml.Data.LoadFromEnumerable(rows);
        var roles = new ColumnRoles("Label", [new("X", ColumnRole.Numeric), new("Z", ColumnRole.Numeric)]);

        var built = new List<int>();
        var probe = new TrainerCandidate("Probe", CandidateCatalog.FastForestSpace, 0.3, (context, point, columns, _) =>
        {
            built.Add(point.Int("NumberOfLeaves"));
            return context.BinaryClassification.Trainers.FastForest(new FastForestBinaryTrainer.Options
            {
                NumberOfLeaves = 4,
                NumberOfTrees = 4,
                LabelColumnName = columns.Label,
                FeatureColumnName = columns.Features,
            });
        })
        {
            StartingPoints = [new Dictionary<string, double> { ["NumberOfLeaves"] = 32768, ["NumberOfTrees"] = 4, ["FeatureFraction"] = 1 }],
        };
        var request = new SearchRequest(
            [probe],
            new Auc(),
            scored => new Dictionary<string, double> { ["auc"] = ml.BinaryClassification.EvaluateNonCalibrated(scored).AreaUnderRocCurve },
            TimeSpan.FromMinutes(5),
            new SearchSettings(Seed: 3),
            MaxTrials: 4);

        var result = new TrialRunner(request, new Sink()).Run(data, roles);

        Assert.NotEmpty(built);
        Assert.All(built, leaves => Assert.True(leaves <= 240, $"a trainer was built with {leaves} leaves for 240 rows"));
        Assert.Contains(240, built);   // the proposal of 32,768 was bounded, not dropped
        Assert.Contains(result.Trials, t => t.Parameters!.Any(p => p.Key == "NumberOfLeaves" && p.Value == "32768"));
    }
}
