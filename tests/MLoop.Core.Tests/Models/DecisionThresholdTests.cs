using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.Core.Tests.Models;

public class DecisionThresholdTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "mloop-threshold-" + Guid.NewGuid().ToString("N"));

    public DecisionThresholdTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* temp */ }
    }

    private static (double, bool, double) R(double p, bool positive, double w = 1) => (p, positive, w);

    [Fact]
    public void TheLowestThresholdThatReachesThePrecision_IsChosen_KeepingTheMostRecall()
    {
        // Descending: 0.9+ 0.8+ 0.7- 0.6+ 0.5- 0.4+ — precision 1, 1, .67, .75, .6, .67
        var rows = new[] { R(0.9, true), R(0.8, true), R(0.7, false), R(0.6, true), R(0.5, false), R(0.4, true) };

        var strict = DecisionThreshold.Choose(rows, 1.0)!.Value;
        Assert.Equal(0.75, strict.Threshold, 9);   // halfway between 0.8 (answered positive) and 0.7
        Assert.Equal((1.0, 0.5), (strict.Precision, strict.Recall));
        var chosen = DecisionThreshold.Choose(rows, 0.75)!.Value;
        Assert.Equal(0.55, chosen.Threshold, 9);   // .75 from 0.6 keeps 3 of 4 positives
        Assert.Equal(0.75, chosen.Recall);
    }

    [Fact]
    public void RowsOfOneProbability_AreAnsweredAlike_SoNoThresholdSplitsThem()
    {
        // At 0.8 one positive and one negative tie: a threshold there takes both (precision .67), never just the positive.
        var rows = new[] { R(0.9, true), R(0.8, true), R(0.8, false) };

        Assert.Equal(0.85, DecisionThreshold.Choose(rows, 0.9)!.Value.Threshold, 9);
    }

    [Fact]
    public void Weights_CountInThePrecision()
    {
        var rows = new[] { R(0.9, true), R(0.8, false, w: 9), R(0.7, true) };

        Assert.Equal(0.85, DecisionThreshold.Choose(rows, 0.5)!.Value.Threshold, 9);   // the heavy negative sinks 0.8 and 0.7
    }

    [Fact]
    public void ForTheNegativeClass_TheHighestThresholdThatReachesItsPrecision_IsChosen()
    {
        // Ascending: 0.1- 0.2- 0.3+ 0.4- 0.5- 0.6+ — negative precision 1, 1, .67, .75, .8, .67
        var rows = new[] { R(0.1, false), R(0.2, false), R(0.3, true), R(0.4, false), R(0.5, false), R(0.6, true) };

        // At 1.0 only 0.1 and 0.2 can be answered negative: the threshold is halfway to the next probability up.
        var strict = DecisionThreshold.Choose(rows, 1.0, ofPositive: false)!.Value;
        Assert.Equal(0.25, strict.Threshold, 9);
        Assert.Equal((1.0, 0.5), (strict.Precision, strict.Recall));
        var chosen = DecisionThreshold.Choose(rows, 0.75, ofPositive: false)!.Value;
        Assert.Equal(0.55, chosen.Threshold, 9);   // .8 below 0.55 keeps all 4 negatives
        Assert.Equal(1.0, chosen.Recall);
        Assert.Equal(0.8, chosen.Precision, 9);
    }

    [Fact]
    public void ForTheNegativeClass_AnsweringEveryRowNegative_PutsTheThresholdAboveTheTop()
    {
        var rows = new[] { R(0.2, false), R(0.7, false) };

        var chosen = DecisionThreshold.Choose(rows, 1.0, ofPositive: false)!.Value;

        Assert.True(chosen.Threshold > 0.7);
        Assert.True((float)0.7 < (float)chosen.Threshold);   // the answer rule compares floats
        Assert.Null(DecisionThreshold.Choose([R(0.9, true)], 0.5, ofPositive: false));   // no negative at all
    }

    [Fact]
    public void TheThreshold_NeverSitsOnARowsProbability_SoAnotherRuntimesLastDigitKeepsEveryAnswer()
    {
        // Many rows on few values, as a tree or a calibration's steps give: a threshold on 0.7177 left
        // those rows to float noise — an ONNX graph answered 39 of 300 of them the other way.
        var rows = Enumerable.Repeat(R(0.9555, true), 70).Concat(Enumerable.Repeat(R(0.7177, true), 40))
            .Concat(Enumerable.Repeat(R(0.7177, false), 5)).Concat(Enumerable.Repeat(R(0.178, false), 60)).ToArray();

        var threshold = (float)DecisionThreshold.Choose(rows, 0.9)!.Value.Threshold;

        foreach (var noise in new[] { -1e-6f, 0f, 1e-6f })
        {
            Assert.True(0.7177f + noise >= threshold);
            Assert.True(0.178f + noise < threshold);
        }
        foreach (var (p, _, _) in rows)
            Assert.NotEqual((float)p, threshold);
    }

    [Fact]
    public void WhenNoThresholdReachesThePrecision_ThereIsNone()
    {
        Assert.Null(DecisionThreshold.Choose([R(0.9, false), R(0.8, true)], 0.9));
        Assert.Null(DecisionThreshold.Choose([R(0.9, false)], 0.5));   // no positive at all
    }

    private sealed class Scored
    {
        public float Probability { get; set; }
        public bool PredictedLabel { get; set; }
    }

    [Fact]
    public void Applied_ItAnswersPositiveFromTheThresholdOn()
    {
        var ml = new MLContext(seed: 1);
        var data = ml.Data.LoadFromEnumerable(new[]
        {
            new Scored { Probability = 0.9f, PredictedLabel = true },
            new Scored { Probability = 0.6f, PredictedLabel = true },
            new Scored { Probability = 0.3f, PredictedLabel = false },
        });
        var identity = ml.Transforms.CopyColumns("Probability2", "Probability").Fit(data);
        var threshold = new DecisionThreshold(0.7, 0.9, 1, 0.5, 3, "f.csv", null, DateTimeOffset.UtcNow);

        var answered = threshold.Apply(ml, identity).Transform(data);

        Assert.Equal([true, false, false], answered.GetColumn<bool>("PredictedLabel").ToArray());
    }

    // Wilson one-sided 95 % (z = 1.645): n right of n certifies n / (n + z²) — 20 → 0.8808, 1 → 0.2699; 950 of 1000 → 0.9374.
    [Theory]
    [InlineData(1.0, 20, 0.8808, 0.0005)]
    [InlineData(0.95, 1000, 0.9374, 0.0005)]
    [InlineData(1.0, 1, 0.2699, 0.0005)]
    public void ThePrecisionBound_IsTheOneSided95PercentWilsonBound(double observed, double rows, double expected, double tolerance)
    {
        Assert.InRange(DecisionThreshold.PrecisionBound(observed, rows), expected - tolerance, expected + tolerance);
    }

    [Fact]
    public void ThePrecisionBound_RisesWithTheRows_AndNeverPassesTheObservedShare()
    {
        var previous = 0.0;
        foreach (var rows in new double[] { 5, 20, 100, 1000, 100000 })
        {
            var bound = DecisionThreshold.PrecisionBound(0.9, rows);
            Assert.True(bound > previous && bound <= 0.9, $"{rows}: {bound}");
            previous = bound;
        }
        Assert.Equal(0, DecisionThreshold.PrecisionBound(0.9, 0));
    }

    // Measured on a real binary set at target 0.9: 95 answered rows leave the bound 0.056 below it, 287 rows 0.030, 964 rows 0.017.
    [Theory]
    [InlineData(0.9, 0.844, true)]
    [InlineData(0.9, 0.870, false)]
    [InlineData(0.9, 0.883, false)]
    [InlineData(0.95, 0.90, false)]   // exactly at the margin is still within it
    [InlineData(0.95, 0.899, true)]
    public void AFileIsTooSmall_WhenTheBoundSitsMoreThanTheMarginBelowTheTarget(double target, double bound, bool uncertified)
    {
        Assert.Equal(uncertified, DecisionThreshold.IsUncertified(target, bound));
    }

    [Fact]
    public void WhatAThresholdRestsOn_IsTheRowsItAnswersAsTheClass()
    {
        // 0.9+ 0.8+ 0.7- 0.6+ 0.5- 0.4+  — a threshold of 0.75 answers the top two, both positive
        var rows = new[] { R(0.9, true), R(0.8, true), R(0.7, false), R(0.6, true), R(0.5, false), R(0.4, true) };

        var (answered, classRows, bound) = DecisionThreshold.EvidenceFor(rows, 0.75, precision: 1.0);

        Assert.Equal((2, 4), (answered, classRows));
        Assert.InRange(bound, 0.3, 0.5);   // two right answers certify little

        // The negative class is answered below the threshold: 0.7-, 0.5- and the positives under them
        var (negativeAnswered, negativeRows, _) = DecisionThreshold.EvidenceFor(rows, 0.45, precision: 1.0, ofPositive: false);
        Assert.Equal((1, 2), (negativeAnswered, negativeRows));
    }

    [Fact]
    public void WeightedRows_CountAsTheirEffectiveNumber()
    {
        // One heavy row among nineteen light ones is not twenty rows of evidence.
        var uniform = Enumerable.Range(0, 20).Select(i => R(0.9 - i * 0.001, true)).ToList();
        var skewed = Enumerable.Range(0, 20).Select(i => R(0.9 - i * 0.001, true, i == 0 ? 100 : 1)).ToList();

        var even = DecisionThreshold.EvidenceFor(uniform, 0.5, precision: 1.0).LowerBound;
        var heavy = DecisionThreshold.EvidenceFor(skewed, 0.5, precision: 1.0).LowerBound;

        Assert.True(heavy < even, $"{heavy} should be below {even}");
    }

    [Fact]
    public void AThresholdWrittenBeforeItRecordedItsEvidence_ReadsAsWithoutIt()
    {
        var model = Path.Combine(_dir, ExperimentLayout.ModelFileName);
        File.WriteAllText(DecisionThreshold.PathBeside(model),
            """{"threshold":0.42,"targetPrecision":0.9,"precision":0.91,"recall":0.6,"rows":120,"chosenOn":"recheck.csv","weightColumn":null,"chosenAt":"2026-10-01T00:00:00+00:00"}""");

        var read = DecisionThreshold.ReadBeside(model)!;

        Assert.Equal(0.42, read.Threshold);
        Assert.Null(read.Answered);
        Assert.Null(read.PrecisionLowerBound);
    }

    [Fact]
    public void TheEvidence_RoundTripsThroughTheFileBesideTheModel()
    {
        var model = Path.Combine(_dir, ExperimentLayout.ModelFileName);
        var threshold = new DecisionThreshold(0.42, 0.9, 0.91, 0.6, 120, "recheck.csv", null, DateTimeOffset.UtcNow,
            Answered: 40, ClassRows: 55, PrecisionLowerBound: 0.8);

        threshold.WriteBeside(model);

        Assert.Equal(threshold, DecisionThreshold.ReadBeside(model));
    }

    [Fact]
    public void ItIsReadFromBesideTheModel_AndAbsentMeansNone()
    {
        var model = Path.Combine(_dir, ExperimentLayout.ModelFileName);
        Assert.Null(DecisionThreshold.ReadBeside(model));
        var threshold = new DecisionThreshold(0.42, 0.9, 0.91, 0.6, 120, "recheck.csv", "w", DateTimeOffset.UtcNow);

        threshold.WriteBeside(model);

        Assert.Equal(threshold, DecisionThreshold.ReadBeside(model));
        Assert.Equal(Path.Combine(_dir, ExperimentLayout.DecisionThresholdFileName), DecisionThreshold.PathBeside(model));
    }
}
