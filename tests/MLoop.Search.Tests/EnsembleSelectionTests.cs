using Microsoft.ML;
using MLoop.Search.Ensemble;
using MLoop.Search.Validation;

namespace MLoop.Search.Tests;

/// <summary>
/// <see cref="EnsembleSelection"/> — a blend chosen from retained trials: the members that help are counted, the blend's own
/// out-of-fold answer is their weighted mean, a lone winner or a lone blendable trial is no blend, and a blend holds at most
/// five distinct members.
/// </summary>
public sealed class EnsembleSelectionTests
{
    private const int Rows = 300;

    private sealed class BinaryScored
    {
        public bool Y { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
        public float W { get; set; } = 1;
    }

    private sealed class RegressionScored
    {
        public float Y { get; set; }
        public float Score { get; set; }
    }

    private static readonly bool[] Labels = [.. Enumerable.Range(0, Rows).Select(i => (i * 7919) % 11 < 5)];

    private static float[] NoisyProbabilities(int seed, double noise)
    {
        var random = new Random(seed);
        return [.. Labels.Select(l => (float)Math.Clamp((l ? 0.75 : 0.25) + (random.NextDouble() * 2 - 1) * noise, 0.01, 0.99))];
    }

    private static RetainedTrial Binary(float[] probabilities, bool[]? labels = null, string trainer = "T", float[]? weights = null, bool withProbability = true)
    {
        var ml = new MLContext(seed: 1);
        var truth = labels ?? Labels;
        var rows = truth.Select((l, i) => new BinaryScored
        {
            Y = l,
            Score = probabilities[i] - 0.5f,
            Probability = probabilities[i],
            W = weights?[i] ?? 1,
        }).ToList();
        var view = ml.Data.LoadFromEnumerable(rows);
        if (!withProbability)
            view = ml.Transforms.DropColumns("Probability").Fit(view).Transform(view);
        var outOfFold = OutOfFoldScores.Read(view, "Y", weights is null ? null : "W")!;
        return new RetainedTrial(new TrialResult { Trainer = trainer }, outOfFold, _ => throw new NotSupportedException());
    }

    // Higher is better: the weighted negative Brier score of the blended probabilities.
    private static double NegativeBrier(OutOfFoldScores rows, double[] probability)
    {
        double sum = 0, weight = 0;
        for (var i = 0; i < probability.Length; i++)
        {
            var miss = probability[i] - (rows.Label[i] ? 1 : 0);
            sum += rows.Weight[i] * miss * miss;
            weight += rows.Weight[i];
        }
        return -sum / weight;
    }

    [Fact]
    public void TwoMembersWithIndependentErrors_BothCount_AndTheBlendBeatsTheBestAlone()
    {
        RetainedTrial[] retained = [Binary(NoisyProbabilities(1, 0.4)), Binary(NoisyProbabilities(2, 0.4))];

        var plan = EnsembleSelection.Choose(retained, NegativeBrier, higherIsBetter: true);

        Assert.NotNull(plan);
        Assert.Equal(2, plan.Members.Count);
        Assert.True(plan.Score > plan.BestSingleScore, $"{plan.Score} should beat {plan.BestSingleScore}.");
        var steps = plan.Counts.Sum();
        Assert.Equal(steps, plan.Steps);
        for (var r = 0; r < plan.OutOfFold.Length; r++)
        {
            var expected = (plan.Counts[0] * retained[0].OutOfFold.Probability![r] + plan.Counts[1] * retained[1].OutOfFold.Probability![r]) / steps;
            Assert.Equal(expected, plan.OutOfFold[r], 5);
        }
    }

    [Fact]
    public void ThePlansBlendedRows_AreTheBlendsOwnAnswersOnTheMembersLabelsAndWeights()
    {
        var weights = new float[Rows];
        Array.Fill(weights, 3f);
        var plan = EnsembleSelection.Choose(
            [Binary(NoisyProbabilities(1, 0.4), weights: weights), Binary(NoisyProbabilities(2, 0.4), weights: weights)], NegativeBrier, higherIsBetter: true)!;

        var rows = plan.Blended;
        Assert.Equal(plan.OutOfFold, rows.Probability);
        Assert.Equal(plan.OutOfFold.Select(p => MathF.Log(p / (1 - p))), rows.Score);
        Assert.Equal(Labels, rows.Label);
        Assert.All(rows.Weight, w => Assert.Equal(3f, w));
    }

    [Fact]
    public void ATrialThatOnlyHurts_LeavesNoBlend()
    {
        var good = Binary(NoisyProbabilities(1, 0.05));
        var flat = Binary([.. Enumerable.Repeat(0.5f, Rows)]);

        Assert.Null(EnsembleSelection.Choose([good, flat], NegativeBrier, higherIsBetter: true));
    }

    [Fact]
    public void AloneBlendableTrial_IsNoBlend_AndATrainerWithNoProbabilityIsLeftOut()
    {
        var withProbability = Binary(NoisyProbabilities(1, 0.3));
        var without = Binary(NoisyProbabilities(2, 0.3), withProbability: false);

        Assert.Null(EnsembleSelection.Choose([withProbability], NegativeBrier, higherIsBetter: true));
        Assert.Null(EnsembleSelection.Choose([withProbability, without], NegativeBrier, higherIsBetter: true));
    }

    [Fact]
    public void ATrialOnOtherRows_IsLeftOut()
    {
        var flipped = Binary(NoisyProbabilities(2, 0.3), labels: [.. Labels.Select(l => !l)]);
        var plan = EnsembleSelection.Choose(
            [Binary(NoisyProbabilities(1, 0.4)), flipped, Binary(NoisyProbabilities(3, 0.4))], NegativeBrier, higherIsBetter: true);

        Assert.NotNull(plan);
        Assert.All(plan.Members, m => Assert.True(m.OutOfFold.Label.SequenceEqual(Labels)));
    }

    [Fact]
    public void MoreCountedMembersThanABlendHolds_AreCutToTheMostCountedFive()
    {
        var retained = Enumerable.Range(1, 9).Select(seed => Binary(NoisyProbabilities(seed, 0.45))).ToArray();

        var plan = EnsembleSelection.Choose(retained, NegativeBrier, higherIsBetter: true, steps: 60);

        Assert.NotNull(plan);
        Assert.InRange(plan.Members.Count, 2, EnsembleSelection.DefaultMaxMembers);
        Assert.True(plan.Members.Select(m => Array.IndexOf(retained, m)).SequenceEqual(plan.Members.Select(m => Array.IndexOf(retained, m)).Order()),
            "The members keep the order they were retained in.");
    }

    [Fact]
    public void TheSameAnswers_GiveTheSamePlan()
    {
        RetainedTrial[] retained = [Binary(NoisyProbabilities(1, 0.4)), Binary(NoisyProbabilities(2, 0.4)), Binary(NoisyProbabilities(3, 0.4))];

        var first = EnsembleSelection.Choose(retained, NegativeBrier, higherIsBetter: true)!;
        var second = EnsembleSelection.Choose(retained, NegativeBrier, higherIsBetter: true)!;

        Assert.Equal(first.Counts, second.Counts);
        Assert.Equal(first.OutOfFold, second.OutOfFold);
    }

    [Fact]
    public void Weights_ReachTheMetric()
    {
        var weights = new float[Rows];
        Array.Fill(weights, 2f);
        double[]? seen = null;
        var retained = new[] { Binary(NoisyProbabilities(1, 0.4), weights: weights), Binary(NoisyProbabilities(2, 0.4), weights: weights) };

        EnsembleSelection.Choose(retained, (rows, probability) => { seen = [.. rows.Weight.Select(w => (double)w)]; return NegativeBrier(rows, probability); }, higherIsBetter: true);

        Assert.NotNull(seen);
        Assert.All(seen, w => Assert.Equal(2d, w));
    }

    [Fact]
    public void ARegression_IsBlendedOnItsPredictedValues()
    {
        var ml = new MLContext(seed: 1);
        var truth = Enumerable.Range(0, Rows).Select(i => (float)Math.Sin(i / 17.0) * 10).ToArray();
        RetainedTrial Regression(int seed)
        {
            var random = new Random(seed);
            var view = ml.Data.LoadFromEnumerable(truth.Select(t => new RegressionScored { Y = t, Score = t + (float)((random.NextDouble() * 2 - 1) * 3) }));
            return new RetainedTrial(new TrialResult { Trainer = "R" }, OutOfFoldScores.Read(view, "Y", null, regression: true)!, _ => throw new NotSupportedException());
        }
        static double NegativeRmse(OutOfFoldScores rows, double[] blended)
            => -Math.Sqrt(blended.Select((b, i) => (b - rows.Target[i]) * (b - rows.Target[i])).Average());

        var plan = EnsembleSelection.Choose([Regression(1), Regression(2)], NegativeRmse, higherIsBetter: true);

        Assert.NotNull(plan);
        Assert.Equal(2, plan.Members.Count);
        Assert.True(plan.Score > plan.BestSingleScore);
    }
}
