using MLoop.Search.Candidates;
using MLoop.Search.Tuning;

namespace MLoop.Search.Tests;

/// <summary>
/// A trainer that fits a model for each class costs the more the more classes there are, so the proposer's first pick on a label
/// of many classes is not one of them (measured: a hundred classes, 4,000 rows — 7.9 s for the forest trainer, 0.9 s for maximum entropy).
/// </summary>
public sealed class ClassAwareCostTests
{
    private static string FirstProposed(int classes)
    {
        var schemas = CandidateCatalog.Multiclass(name => name != "LightGbm");
        return new EciSchemaProposer([.. schemas.Select(s => (s.Name, s.CostFor(classes)))], seed: 1).Propose();
    }

    [Fact]
    public void ThePerClassTrainers_AreTheOneVersusAllOnes()
        => Assert.Equal(
            ["FastForestOva", "FastTreeOva", "LbfgsLogisticRegressionOva", "SdcaLogisticRegressionOva"],
            CandidateCatalog.Multiclass().Where(c => c.FitsPerClass).Select(c => c.Name).Order(StringComparer.Ordinal));

    [Fact]
    public void NoBinaryOrRegressionTrainer_FitsPerClass()
        => Assert.DoesNotContain(CandidateCatalog.Binary().Concat(CandidateCatalog.Regression()), c => c.FitsPerClass);

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void OnTwoClassesOrFewer_TheCostIsTheTablesOwn(int classes)
        => Assert.All(CandidateCatalog.Multiclass(), c => Assert.Equal(c.InitialCost, c.CostFor(classes)));

    [Fact]
    public void ThePerClassCost_GrowsWithTheClasses_AndTheOthersDoNot()
    {
        var all = CandidateCatalog.Multiclass().ToDictionary(c => c.Name);
        Assert.Equal(all["FastTreeOva"].InitialCost * 50, all["FastTreeOva"].CostFor(100), 6);
        Assert.Equal(all["LbfgsMaximumEntropy"].InitialCost, all["LbfgsMaximumEntropy"].CostFor(100));
    }

    [Fact]
    public void OnTwoClasses_TheSearchOpensAsItAlwaysDid_WithTheForestOfTrees()
        => Assert.Equal("FastTreeOva", FirstProposed(2));

    [Theory]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(100)]
    public void OnFiveClassesOrMore_TheSearchOpensWithATrainerThatFitsOneModel(int classes)
        => Assert.Equal("LbfgsMaximumEntropy", FirstProposed(classes));

    [Fact]
    public void OnFourClasses_TheTreesStillOpen()
        => Assert.Equal("FastTreeOva", FirstProposed(4));
}
