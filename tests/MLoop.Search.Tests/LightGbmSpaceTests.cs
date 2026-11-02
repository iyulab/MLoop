using MLoop.Search.Candidates;

namespace MLoop.Search.Tests;

/// <summary>
/// The LightGBM space holds only settings the trainer reads. ML.NET bags a fraction of the rows only every
/// <c>SubsampleFrequency</c> iterations, and that defaults to 0, so a subsample fraction the builder passes without a
/// frequency changes nothing: a replay of one search's best settings scored the same at 0.26 and at 1. A dimension of
/// that kind is a direction the search walks for no effect.
/// </summary>
public sealed class LightGbmSpaceTests
{
    [Fact]
    public void TheSpaceHasNoSubsampleFraction_BecauseTheBuildersNeverTurnBaggingOn()
    {
        foreach (var candidate in CandidateCatalog.Binary().Concat(CandidateCatalog.Multiclass()).Concat(CandidateCatalog.Regression()).Where(c => c.Name == "LightGbm"))
        {
            var names = candidate.Space.Dimensions.Select(d => d.Name).ToList();
            Assert.DoesNotContain("SubsampleFraction", names);
            // what the builders read: every name a start of the library's defaults sets is a dimension of the space
            foreach (var start in candidate.StartingPoints)
                Assert.All(start.Keys, key => Assert.Contains(key, names));
        }
        Assert.Equal(8, CandidateCatalog.LightGbmSpace.Dimensions.Count);
    }
}
