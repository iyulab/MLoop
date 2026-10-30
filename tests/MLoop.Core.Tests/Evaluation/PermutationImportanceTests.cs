using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

public class PermutationImportanceTests
{
    [Fact]
    public void Order_IsAPermutation_FixedBySeed()
    {
        var order = PermutationImportance.Order(500, seed: 7);

        Assert.Equal(Enumerable.Range(0, 500), order.Order());
        Assert.Equal(order, PermutationImportance.Order(500, seed: 7));
        Assert.NotEqual(order, PermutationImportance.Order(500, seed: 8));
        Assert.NotEqual(Enumerable.Range(0, 500), order);          // an identity shuffle would shuffle nothing
        Assert.Empty(PermutationImportance.Order(0, 1));
    }

    [Fact]
    public void SeedOf_GivesEveryColumnAndRepeatItsOwnShuffle()
    {
        var seeds = from column in Enumerable.Range(0, 40)
                    from repeat in Enumerable.Range(0, 5)
                    select PermutationImportance.SeedOf(1, column, repeat);

        Assert.Equal(200, seeds.Distinct().Count());
        Assert.Equal(PermutationImportance.SeedOf(1, 3, 2), PermutationImportance.SeedOf(1, 3, 2));
    }

    [Fact]
    public void Summarize_ReadsTheFallInTheDirectionOfTheMetric()
    {
        // Higher is better (auc): shuffling a column that held the score up lowers it.
        var higher = PermutationImportance.Summarize(0.90, higherIsBetter: true,
            [("a", [0.60, 0.62]), ("b", [0.89, 0.91]), ("c", [0.95])]);
        // Lower is better (rmse): shuffling raises it, and the fall is the rise.
        var lower = PermutationImportance.Summarize(1.0, higherIsBetter: false,
            [("a", [1.40, 1.42]), ("b", [0.99, 1.01]), ("c", [0.95])]);

        Assert.Equal(["a", "b", "c"], higher.Select(f => f.Column));
        Assert.Equal(0.29, higher[0].Importance, 12);
        Assert.Equal(0.0, higher[1].Importance, 12);
        Assert.Equal(-0.05, higher[2].Importance, 12);             // shuffling helped by chance: reported, not hidden
        Assert.Equal(["a", "b", "c"], lower.Select(f => f.Column));
        Assert.Equal(0.41, lower[0].Importance, 12);
        Assert.Equal(0.0, lower[1].Importance, 12);
        Assert.Equal(-0.05, lower[2].Importance, 12);
    }

    [Fact]
    public void Summarize_GivesTheSpreadOfTheRepeats_AndOrdersTiesByName()
    {
        var summary = PermutationImportance.Summarize(1.0, higherIsBetter: true,
            [("zeta", [0.5, 0.5, 0.5]), ("alpha", [0.4, 0.6]), ("empty", [])]);

        Assert.Equal(["alpha", "zeta"], summary.Select(f => f.Column));   // the column with no score is left out; equal falls order by name
        Assert.Equal(0.5, summary[0].Importance, 12);
        Assert.Equal(Math.Sqrt(0.02), summary[0].StandardDeviation, 12);  // falls 0.6 and 0.4: mean 0.5, sample sd √0.02
        Assert.Equal(0, summary[1].StandardDeviation);
    }
}
