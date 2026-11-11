using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// A multiclass search ranks trials by the quality of the probabilities the run will serve — those after a temperature
/// is fitted — not by the probabilities a trainer happens to produce. A one-versus-all tree's are normalized fixed
/// sigmoids (measured top-label ECE 0.3 to 0.4): accurate rows, a log loss that says otherwise. Ranked raw, it lost to a
/// worse model whose probabilities were honest, and the run then served the worse model.
/// </summary>
public class FoldCalibrationTests
{
    private const int Classes = 3;
    private const int Rows = 3000;

    private sealed class ScoredRow
    {
        [KeyType(Classes)] public uint Label { get; set; }
        [VectorType(Classes)] public float[] Score { get; set; } = [];
        [KeyType(Classes)] public uint PredictedLabel { get; set; }
    }

    // Rows whose class shows in `separation` standard deviations of per-class evidence; `shrink` scales the evidence the
    // model turns into probabilities. With shrink = separation / sigma^2 the probabilities are exactly calibrated
    // (Gaussian evidence of equal variance), and any other shrink leaves the same answers sure of themselves too little or too much.
    private static IDataView Scored(MLContext ml, double separation, double shrink, int seed)
    {
        var random = new Random(seed);
        double Gaussian()
        {
            var u1 = 1 - random.NextDouble();
            return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * random.NextDouble());
        }
        var rows = new List<ScoredRow>(Rows);
        for (var i = 0; i < Rows; i++)
        {
            var truth = random.Next(Classes);
            var evidence = Enumerable.Range(0, Classes).Select(c => (c == truth ? separation : 0) + Gaussian()).ToArray();
            var logits = evidence.Select(e => e * shrink).ToArray();
            var max = logits.Max();
            var exp = logits.Select(l => Math.Exp(l - max)).ToArray();
            var total = exp.Sum();
            rows.Add(new ScoredRow
            {
                Label = (uint)truth,   // a key property is 0-based: ML.NET stores it as value + 1

                Score = [.. exp.Select(e => (float)(e / total))],
                PredictedLabel = (uint)Array.IndexOf(evidence, evidence.Max()),
            });
        }
        return ml.Data.LoadFromEnumerable(rows);
    }

    private static double LogLoss(MLContext ml, IDataView scored)
        => ml.MulticlassClassification.Evaluate(scored, "Label").LogLoss;

    [Fact]
    public void AnAccurateModelWithFlatProbabilities_LosesRawToAWorseHonestOne_AndWinsOnceTempered()
    {
        var ml = new MLContext(seed: 1);
        // accurate but unsure of itself (its evidence turned into probabilities at a twenty-fifth of the calibrated scale) — a one-versus-all tree's habit
        var accurate = Scored(ml, separation: 2.5, shrink: 0.1, seed: 11);
        // less accurate and exactly calibrated (shrink = separation for unit variance)
        var honest = Scored(ml, separation: 1.0, shrink: 1.0, seed: 12);

        // The defect the tempering fixes: ranked raw, the worse model is ahead.
        Assert.True(LogLoss(ml, honest) < LogLoss(ml, accurate),
            $"setup: the raw ranking must prefer the honest model (honest {LogLoss(ml, honest):F3}, accurate {LogLoss(ml, accurate):F3})");

        var accurateTempered = LogLoss(ml, FoldCalibration.Temper(ml, accurate, "Label", null));
        var honestTempered = LogLoss(ml, FoldCalibration.Temper(ml, honest, "Label", null));

        Assert.True(accurateTempered < honestTempered,
            $"tempered, the accurate model must be ahead (accurate {accurateTempered:F3}, honest {honestTempered:F3})");
        Assert.True(accurateTempered < LogLoss(ml, accurate) - 0.3, "the tempering must move an underconfident model's log loss, not nudge it");
    }

    [Fact]
    public void AModelThatIsAlreadyCalibrated_IsLeftAlmostWhereItWas()
    {
        var ml = new MLContext(seed: 1);
        var honest = Scored(ml, separation: 1.0, shrink: 1.0, seed: 12);

        var before = LogLoss(ml, honest);
        var after = LogLoss(ml, FoldCalibration.Temper(ml, honest, "Label", null));

        // Fitted on a sample of the part (FitRows), a temperature is not guaranteed better on every row of it: a model that was already
        // calibrated stays where it was to within the sample's noise.
        Assert.InRange(after, before - 0.02, before + 0.005);
    }

    [Fact]
    public void TheAnswersAreUntouched_OnlyTheProbabilitiesMove()
    {
        var ml = new MLContext(seed: 1);
        var accurate = Scored(ml, separation: 2.5, shrink: 0.1, seed: 11);

        var before = ml.MulticlassClassification.Evaluate(accurate, "Label");
        var after = ml.MulticlassClassification.Evaluate(FoldCalibration.Temper(ml, accurate, "Label", null), "Label");

        Assert.Equal(before.MicroAccuracy, after.MicroAccuracy);
        Assert.Equal(before.MacroAccuracy, after.MacroAccuracy);
    }

    [Fact]
    public void ARowSetThatIsNotMulticlassProbabilities_IsReturnedAsItIs()
    {
        var ml = new MLContext(seed: 1);
        var notScored = ml.Data.LoadFromEnumerable(new[] { new Plain { Label = 1, Score = 0.5f } });

        Assert.Same(notScored, FoldCalibration.Temper(ml, notScored, "Label", null));
    }

    private sealed class Plain
    {
        public float Label { get; set; }
        public float Score { get; set; }
    }
}
