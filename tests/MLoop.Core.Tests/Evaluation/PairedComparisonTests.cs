using Microsoft.ML;
using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

public class PairedComparisonTests
{
    [Fact]
    public void ACandidateRightWhereTheBaselineIsWrong_IsBetter()
    {
        // 200 rows: both right on 150; on 40 only the candidate is right; on 10 only the baseline.
        var candidate = Enumerable.Range(0, 200).Select(i => i >= 190 ? 1.0 : 0.0).ToArray();
        var baseline = Enumerable.Range(0, 200).Select(i => i is >= 150 and < 190 ? 1.0 : 0.0).ToArray();

        var result = PairedComparison.Compare(PairedComparison.ErrorRate, candidate, baseline);

        Assert.Equal(200, result.Rows);
        Assert.Equal(0.05, result.Candidate, 9);
        Assert.Equal(0.20, result.Baseline, 9);
        Assert.Equal(-0.15, result.Difference, 9);
        Assert.True(result.High < 0, $"interval [{result.Low}, {result.High}]");
        Assert.True(result.Low <= result.Difference && result.Difference <= result.High);
        Assert.Equal(PairedVerdict.CandidateBetter, result.Verdict);
    }

    [Fact]
    public void TheSameAnswers_CannotBeToldApart()
    {
        var losses = Enumerable.Range(0, 100).Select(i => i % 7 == 0 ? 1.0 : 0.0).ToArray();

        var result = PairedComparison.Compare(PairedComparison.ErrorRate, losses, losses);

        Assert.Equal(0, result.Difference);
        Assert.Equal(0, result.Low);
        Assert.Equal(0, result.High);
        Assert.Equal(PairedVerdict.Indistinguishable, result.Verdict);
    }

    [Fact]
    public void AFewDisagreements_DoNotDecide()
    {
        var candidate = new double[60];
        var baseline = new double[60];
        candidate[0] = 1; baseline[1] = 1; baseline[2] = 1;   // one miss against two

        var result = PairedComparison.Compare(PairedComparison.ErrorRate, candidate, baseline);

        Assert.Equal(PairedVerdict.Indistinguishable, result.Verdict);
        Assert.True(result.Low < 0 && result.High > 0);
    }

    [Fact]
    public void Weights_ChangeWhatEachRowCountsFor()
    {
        // The candidate misses row 0, the baseline row 1; row 0 counts nine times as much.
        double[] candidate = [1, 0, 0, 0], baseline = [0, 1, 0, 0], weights = [9, 1, 1, 1];

        var result = PairedComparison.Compare(PairedComparison.ErrorRate, candidate, baseline, weights);

        Assert.Equal(9.0 / 12, result.Candidate, 9);
        Assert.Equal(1.0 / 12, result.Baseline, 9);
        Assert.True(result.Difference > 0);
    }

    [Fact]
    public void RowsWithoutALabel_OrWithoutWeight_AreLeftOutOfBoth()
    {
        double[] candidate = [0, double.NaN, 1, 0], baseline = [0, 1, double.NaN, 1], weights = [1, 1, 1, 0];

        var result = PairedComparison.Compare(PairedComparison.ErrorRate, candidate, baseline, weights);

        Assert.Equal(1, result.Rows);
        Assert.Equal(0, result.Difference);
    }

    [Fact]
    public void NoUsableRow_IsReported_NotThrown()
    {
        var result = PairedComparison.Compare(PairedComparison.SquaredError, [double.NaN], [double.NaN]);

        Assert.Equal(0, result.Rows);
        Assert.Equal(PairedVerdict.Indistinguishable, result.Verdict);
    }

    [Fact]
    public void TheSameRows_GiveTheSameInterval()
    {
        var c = Enumerable.Range(0, 80).Select(i => (double)(i % 3)).ToArray();
        var b = Enumerable.Range(0, 80).Select(i => (double)(i % 5)).ToArray();

        Assert.Equal(PairedComparison.Compare(PairedComparison.SquaredError, c, b),
                     PairedComparison.Compare(PairedComparison.SquaredError, c, b));
    }

    [Theory]
    [InlineData("binary-classification", PairedComparison.ErrorRate)]
    [InlineData("multiclass-classification", PairedComparison.ErrorRate)]
    [InlineData("regression", PairedComparison.SquaredError)]
    [InlineData("clustering", null)]
    [InlineData("forecasting", null)]
    public void EachTaskIsComparedByItsOwnRowLoss(string task, string? measure)
        => Assert.Equal(measure, PairedComparison.MeasureOf(task));

    [Fact]
    public void LossesOverDifferentRows_AreRefused()
        => Assert.Throws<ArgumentException>(() => PairedComparison.Compare(PairedComparison.ErrorRate, [0, 1], [0]));

    private sealed class BinaryRow
    {
        public bool Label { get; set; }
        public bool PredictedLabel { get; set; }
    }

    [Fact]
    public void ABinaryClassifiersRowLoss_IsWhetherItsAnswerIsWrong()
    {
        var ml = new MLContext(seed: 1);
        var scored = ml.Data.LoadFromEnumerable(new[]
        {
            new BinaryRow { Label = true, PredictedLabel = true },
            new BinaryRow { Label = true, PredictedLabel = false },
            new BinaryRow { Label = false, PredictedLabel = true },
            new BinaryRow { Label = false, PredictedLabel = false },
        });

        Assert.Equal([0.0, 1.0, 1.0, 0.0], PairedComparison.RowLosses(scored, "binary-classification", "Label")!);
    }

    private sealed class MulticlassRow
    {
        public string Label { get; set; } = "";
        [Microsoft.ML.Data.VectorType(3)]
        public float[] Score { get; set; } = [];
    }

    [Fact]
    public void AMulticlassClassifiersRowLoss_ReadsItsTopScoreAgainstTheLabelKey()
    {
        var ml = new MLContext(seed: 1);
        var raw = ml.Data.LoadFromEnumerable(new[]
        {
            new MulticlassRow { Label = "a", Score = [0.7f, 0.2f, 0.1f] },   // right
            new MulticlassRow { Label = "b", Score = [0.6f, 0.3f, 0.1f] },   // wrong
            new MulticlassRow { Label = "c", Score = [0.1f, 0.1f, 0.8f] },   // right
        });
        var scored = ml.Transforms.Conversion.MapValueToKey("Label").Fit(raw).Transform(raw);

        Assert.Equal([0.0, 1.0, 0.0], PairedComparison.RowLosses(scored, "multiclass-classification", "Label")!);
    }

    [Fact]
    public void WithoutTheColumnsItReads_ThereIsNoRowLoss()
    {
        var ml = new MLContext(seed: 1);
        var scored = ml.Data.LoadFromEnumerable(new[] { new BinaryRow() });

        Assert.Null(PairedComparison.RowLosses(scored, "regression", "Label"));
        Assert.Null(PairedComparison.RowLosses(scored, "clustering", "Label"));
    }
}
