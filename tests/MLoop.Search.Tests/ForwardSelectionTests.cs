using MLoop.Search.Ensemble;

namespace MLoop.Search.Tests;

/// <summary>
/// <see cref="ForwardSelection"/> — the blend of a few trials' out-of-fold answers: it counts the members that help, keeps the
/// prefix that scored best, picks one member when the others only hurt, and gives the same counts for the same answers.
/// </summary>
public sealed class ForwardSelectionTests
{
    private const int Rows = 400;

    private static readonly float[] Truth = [.. Enumerable.Range(0, Rows).Select(i => (float)Math.Sin(i / 17.0) * 10)];

    private static float[] Noisy(int seed, double noise)
    {
        var random = new Random(seed);
        return [.. Truth.Select(t => t + (float)((random.NextDouble() * 2 - 1) * noise))];
    }

    private static double Rmse(double[] blended)
        => Math.Sqrt(blended.Select((b, i) => (b - Truth[i]) * (b - Truth[i])).Average());

    [Fact]
    public void TwoMembersWithIndependentErrors_AreBothCounted_AndTheBlendBeatsEitherAlone()
    {
        float[][] members = [Noisy(1, 3), Noisy(2, 3)];

        var selection = ForwardSelection.Select(members, Rmse, higherIsBetter: false);

        Assert.Equal(2, selection.Members);
        Assert.False(selection.IsSingle);
        Assert.True(selection.Score < selection.SingleScores.Min(), $"{selection.Score} against {string.Join(", ", selection.SingleScores)}");
        Assert.Equal(1.0, selection.Weights.Sum(), 12);
    }

    [Fact]
    public void AMemberThatOnlyHurts_IsNotCounted_AndASingleMemberIsSaid()
    {
        var useless = new float[Rows];   // always 0: far from the truth
        float[][] members = [useless, Noisy(1, 0.1)];

        var selection = ForwardSelection.Select(members, Rmse, higherIsBetter: false);

        Assert.True(selection.IsSingle);
        Assert.Equal(0, selection.Counts[0]);
        Assert.True(selection.Counts[1] >= 1);
    }

    [Fact]
    public void TheBestPrefixIsKept_NotTheLastStep()
    {
        // Adding the second member always makes the metric worse, so the prefix with one addition wins over the 20-step walk.
        float[][] members = [Noisy(1, 0.1), Noisy(2, 6)];

        var selection = ForwardSelection.Select(members, Rmse, higherIsBetter: false);

        Assert.True(selection.Steps < 20);
        Assert.Equal(selection.SingleScores[0], Rmse([.. members[0].Select(v => (double)v)]), 9);
    }

    [Fact]
    public void AHigherIsBetterMetric_IsHandledTheOtherWayRound()
    {
        float[][] members = [Noisy(1, 3), Noisy(2, 3)];
        double NegativeRmse(double[] blended) => -Rmse(blended);

        var lower = ForwardSelection.Select(members, Rmse, higherIsBetter: false);
        var higher = ForwardSelection.Select(members, NegativeRmse, higherIsBetter: true);

        Assert.Equal(lower.Counts, higher.Counts);
    }

    [Fact]
    public void TheSameAnswers_GiveTheSameCounts_AndATieGoesToTheEarlierMember()
    {
        var same = Noisy(5, 2);
        float[][] members = [same, (float[])same.Clone(), (float[])same.Clone()];

        var first = ForwardSelection.Select(members, Rmse, higherIsBetter: false);
        var again = ForwardSelection.Select(members, Rmse, higherIsBetter: false);

        Assert.Equal(first.Counts, again.Counts);
        Assert.Equal([1, 0, 0], first.Counts);   // the first addition is the earliest of equals, and a later prefix must be strictly better
    }

    [Fact]
    public void ACandidateWithANonFiniteMetric_IsSkipped_AndWhenNoneIsFiniteTheSelectionSaysSo()
    {
        float[][] members = [Noisy(1, 1), Noisy(2, 1)];
        // Not finite exactly when the blend is the second member alone — its first addition, and its score by itself.
        double Choosy(double[] blended) => blended[0] == members[1][0] && blended[7] == members[1][7] ? double.NaN : Rmse(blended);

        var selection = ForwardSelection.Select(members, Choosy, higherIsBetter: false);

        Assert.True(double.IsNaN(selection.SingleScores[1]));
        Assert.True(double.IsFinite(selection.SingleScores[0]));
        Assert.True(selection.Counts[0] >= 1);
        Assert.True(double.IsFinite(selection.Score));

        Assert.Throws<InvalidOperationException>(() => ForwardSelection.Select(members, _ => double.NaN, higherIsBetter: false));
    }

    [Fact]
    public void NoMembers_NoRows_OrMembersOfDifferentLengths_AreRefused()
    {
        Assert.Throws<ArgumentException>(() => ForwardSelection.Select([], Rmse, higherIsBetter: false));
        Assert.Throws<ArgumentException>(() => ForwardSelection.Select([[]], Rmse, higherIsBetter: false));
        Assert.Throws<ArgumentException>(() => ForwardSelection.Select([new float[3], new float[4]], Rmse, higherIsBetter: false));
        Assert.Throws<ArgumentOutOfRangeException>(() => ForwardSelection.Select([new float[3]], Rmse, higherIsBetter: false, steps: 0));
    }
}
