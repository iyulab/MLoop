using MLoop.Search.Tuning;

namespace MLoop.Search.Tests;

/// <summary>
/// The tuner's pure parts: the unit-space mapping of a hyperparameter, the cost-frugal tuner's start,
/// and the schema proposer's ECI. Trial-by-trial agreement with AutoML is <c>TunerTraceTests</c>, against a
/// recorded trace of AutoML's tuner.
/// </summary>
public sealed class TuningTests
{
    [Fact]
    public void AnInteger_MapsToTheUnitCubeAndBack_RoundingDown()
    {
        var leaves = SearchDimension.Integer("NumberOfLeaves", 4, 32768, defaultValue: 4);

        Assert.Equal(0.0, leaves.UnitOf(4));
        Assert.Equal(4, leaves.ValueAt(0));
        Assert.Equal(32768, leaves.ValueAt(1));
        Assert.Equal(16386, leaves.ValueAt(0.5));     // 4 + 32764 · 0.5, floored
    }

    [Fact]
    public void ALogScale_SpendsTheCubeEvenlyOverOrdersOfMagnitude()
    {
        var rate = SearchDimension.Double("LearningRate", 2e-10, 1, logBase: true, defaultValue: 1);

        Assert.Equal(1.0, rate.Default.Single(), 12);
        Assert.Equal(Math.Sqrt(2e-10), rate.ValueAt(0.5), 15);
        Assert.Equal(0.5, rate.UnitOf(Math.Sqrt(2e-10)), 12);
    }

    [Fact]
    public void AChoice_IsAnIndex_AndAChoiceOfOneTakesNoCoordinate()
    {
        var encoding = SearchDimension.Choice("Encoding", ["one-hot", "hash"], defaultChoice: "hash");
        var single = SearchDimension.Choice("Schema", ["only"]);

        Assert.Equal(0.5, encoding.Default.Single());
        Assert.Equal(1, encoding.ValueAt(1));         // the top of the cube is the last choice
        Assert.Equal(0, encoding.ValueAt(0.49));
        Assert.Equal(0, single.FeatureDim);
        var space = new SearchSpace([encoding, single]);
        Assert.Equal(1, space.FeatureDim);
        Assert.Equal("hash", space.PointAt([0.7]).Choice("Encoding"));
        Assert.Equal("only", space.PointAt([0.7]).Choice("Schema"));
    }

    [Fact]
    public void TheTuner_StartsAtTheDefault_TheCheapModel()
    {
        var space = new SearchSpace([
            SearchDimension.Integer("NumberOfTrees", 4, 32768, defaultValue: 4),
            SearchDimension.Integer("NumberOfLeaves", 4, 32768, defaultValue: 4),
        ]);
        var tuner = new CostFrugalTuner(space, seed: 1);

        var first = tuner.Propose(0);
        tuner.Update(0, first, 0.5, 10);
        var second = tuner.Propose(1);

        Assert.Equal((4, 4), (first.Int("NumberOfTrees"), first.Int("NumberOfLeaves")));
        Assert.NotEqual(first.ToString(), second.ToString());       // then it moves
    }

    private static SearchSpace TreeSpace { get; } = new([
        SearchDimension.Integer("NumberOfTrees", 4, 32768, logBase: true, defaultValue: 4),
        SearchDimension.Integer("NumberOfLeaves", 4, 32768, logBase: true, defaultValue: 4),
    ]);

    private static IReadOnlyDictionary<string, double> Trees(int trees, int leaves)
        => new Dictionary<string, double> { ["NumberOfTrees"] = trees, ["NumberOfLeaves"] = leaves };

    [Fact]
    public void TheTuner_TriesItsStartingPointsFirst_ThenTheCheapDefault()
    {
        var tuner = new CostFrugalTuner(TreeSpace, seed: 1, startingPoints: [TreeSpace.PointOf(Trees(100, 20))]);

        var first = tuner.Propose(0);
        tuner.Update(0, first, 0.4, 10);
        var second = tuner.Propose(1);

        Assert.Equal((100, 20), (first.Int("NumberOfTrees"), first.Int("NumberOfLeaves")));
        Assert.Equal((4, 4), (second.Int("NumberOfTrees"), second.Int("NumberOfLeaves")));
    }

    [Fact]
    public void TheWalk_BeginsAtAStartingPointThatBeatTheDefault()
    {
        var tuner = new CostFrugalTuner(TreeSpace, seed: 1, startingPoints: [TreeSpace.PointOf(Trees(100, 20))]);

        tuner.Update(0, tuner.Propose(0), 0.2, 10);                 // the starting point: a good loss
        var cheap = tuner.Propose(1);
        tuner.Update(1, cheap, 0.5, 10);                            // the cheap default: a worse one
        var walked = tuner.Propose(2);

        Assert.Equal((4, 4), (cheap.Int("NumberOfTrees"), cheap.Int("NumberOfLeaves")));
        // The step is a tenth of the log range, so the next point lies near 100 × 20, not near 4 × 4.
        Assert.InRange(walked.Int("NumberOfTrees"), 30, 400);
        Assert.InRange(walked.Int("NumberOfLeaves"), 6, 70);
    }

    [Fact]
    public void TheWalk_StaysAtTheDefault_WhenTheStartingPointLost()
    {
        var tuner = new CostFrugalTuner(TreeSpace, seed: 1, startingPoints: [TreeSpace.PointOf(Trees(100, 20))]);

        tuner.Update(0, tuner.Propose(0), 0.6, 10);
        var cheap = tuner.Propose(1);
        tuner.Update(1, cheap, 0.3, 10);
        var walked = tuner.Propose(2);

        Assert.InRange(walked.Int("NumberOfTrees"), 4, 12);
        Assert.InRange(walked.Int("NumberOfLeaves"), 4, 12);
    }

    [Fact]
    public void AStartingPointThatFailed_IsSkipped_AndTheDefaultStillStartsTheWalk()
    {
        var tuner = new CostFrugalTuner(TreeSpace, seed: 1, startingPoints: [TreeSpace.PointOf(Trees(100, 20))]);

        tuner.Update(0, tuner.Propose(0), double.MaxValue, 10);
        var cheap = tuner.Propose(1);
        tuner.Update(1, cheap, 0.3, 10);

        Assert.Equal((4, 4), (cheap.Int("NumberOfTrees"), cheap.Int("NumberOfLeaves")));
        Assert.NotNull(tuner.Propose(2));
    }

    [Fact]
    public void PointOf_TakesTheValuesAsGiven_AndRefusesWhatIsNotInTheSpace()
    {
        var point = TreeSpace.PointOf(Trees(100, 20));

        Assert.Equal(100, point.Int("NumberOfTrees"));                 // not a cube coordinate rounded down
        Assert.Throws<ArgumentException>(() => TreeSpace.PointOf(new Dictionary<string, double> { ["NumberOfTrees"] = 100 }));
        Assert.Throws<ArgumentException>(() => TreeSpace.PointOf(new Dictionary<string, double> { ["NumberOfTrees"] = 100, ["NumberOfLeaves"] = 20, ["Depth"] = 3 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => TreeSpace.PointOf(Trees(2, 20)));
        Assert.Throws<ArgumentOutOfRangeException>(() => TreeSpace.PointOf(Trees(100, 40000)));
    }

    [Fact]
    public void TheTuner_IsDeterministicUnderASeed()
    {
        static string Run(int seed)
        {
            var space = new SearchSpace([SearchDimension.Double("L2", 0.03125, 32768, logBase: true, defaultValue: 1)]);
            var tuner = new CostFrugalTuner(space, seed);
            var trace = new List<string>();
            for (var i = 0; i < 30; i++)
            {
                var p = tuner.Propose(i);
                trace.Add(p.ToString());
                tuner.Update(i, p, Math.Abs(Math.Log(p["L2"]) - 2), 1);
            }
            return string.Join("|", trace);
        }

        Assert.Equal(Run(3), Run(3));
        Assert.NotEqual(Run(3), Run(4));
    }

    [Fact]
    public void Eci_ProposesTheCheapestSchemaFirst_ThenByOneOverEci()
    {
        var proposer = new EciSchemaProposer([("lbfgs", 11.513), ("fasttree", 3.701), ("lightgbm", 4.765)], seed: 0);

        Assert.Equal("fasttree", proposer.Propose());
        proposer.Update("fasttree", loss: -0.80, cost: 2);
        var counts = Enumerable.Range(0, 2000).Select(_ => proposer.Propose()).GroupBy(s => s).ToDictionary(g => g.Key, g => g.Count());

        Assert.Equal(3, counts.Count);
        Assert.True(counts["fasttree"] > counts["lbfgs"]);   // cheaper to improve is drawn more
    }

    [Fact]
    public void Eci_AFastTrialIsASuccess_NotAFailure()
    {
        // Upstream counted a trial under one second as failed (integer seconds = 0) and doubled the
        // schema's ECI — on small data every fast trainer was pushed away. A finite loss succeeds here.
        var proposer = new EciSchemaProposer([("sdca", 13.16), ("fasttree", 3.701)], seed: 0);
        proposer.Propose();
        proposer.Update("fasttree", loss: -0.9, cost: 0.2);

        Assert.True(proposer.Eci["fasttree"] < proposer.Eci["sdca"]);
    }

    [Fact]
    public void Eci_AdmittingEverySchema_DrawsAsAnUnrestrictedProposal()
    {
        var plain = new EciSchemaProposer([("lbfgs", 11.513), ("fasttree", 3.701), ("lightgbm", 4.765)], seed: 3);
        var admitting = new EciSchemaProposer([("lbfgs", 11.513), ("fasttree", 3.701), ("lightgbm", 4.765)], seed: 3);
        plain.Update("fasttree", loss: -0.80, cost: 2);
        admitting.Update("fasttree", loss: -0.80, cost: 2);

        Assert.Equal(
            Enumerable.Range(0, 200).Select(_ => plain.Propose()),
            Enumerable.Range(0, 200).Select(_ => admitting.Propose(_ => true)));
    }

    [Fact]
    public void Eci_ProposesOnlyAdmittedSchemas_AndNothingWhenNoneIs()
    {
        var proposer = new EciSchemaProposer([("lbfgs", 11.513), ("fasttree", 3.701), ("lightgbm", 4.765)], seed: 0);

        Assert.Equal("lightgbm", proposer.Propose(id => id != "fasttree"));   // the cheapest admitted one first
        proposer.Update("lightgbm", loss: -0.80, cost: 2);
        Assert.All(Enumerable.Range(0, 500).Select(_ => proposer.Propose(id => id != "lbfgs")), id => Assert.NotEqual("lbfgs", id));
        Assert.Null(proposer.Propose(_ => false));
    }

    [Fact]
    public void Eci_AFailedTrial_DoublesTheLargestEci()
    {
        var proposer = new EciSchemaProposer([("a", 1), ("b", 5)], seed: 0);
        proposer.Propose();
        proposer.Update("a", loss: double.NaN, cost: 3);

        Assert.Equal(10, proposer.Eci["a"]);
    }
}
