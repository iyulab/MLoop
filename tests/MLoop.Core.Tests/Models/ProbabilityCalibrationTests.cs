using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

public class ProbabilityCalibrationTests
{
    // A model that is right as often as p² says it is sure: overconfident. Rows at p are positive with
    // probability p² — drawn deterministically so the test has no noise to wait out.
    private static List<(double, bool, double)> Overconfident(int perLevel = 200)
    {
        var rows = new List<(double, bool, double)>();
        for (var level = 1; level <= 9; level++)
        {
            var p = level / 10.0;
            var positives = (int)Math.Round(perLevel * p * p);
            for (var i = 0; i < perLevel; i++)
                rows.Add((p, i < positives, 1.0));
        }
        return rows;
    }

    [Theory]
    [InlineData("platt")]
    [InlineData("isotonic")]
    public void ACalibration_BringsTheProbabilitiesToHowOftenTheyAreRight(string method)
    {
        var calibration = ProbabilityCalibration.Fit(Overconfident(), method, "recheck.csv", null, DateTimeOffset.UtcNow)!;

        Assert.True(calibration.EceAfter < calibration.EceBefore / 2, $"ECE {calibration.EceBefore} → {calibration.EceAfter}");
        Assert.InRange(calibration.Map(0.9), 0.75, 0.87);   // right 81% of the time at 0.9
        Assert.Equal(9 * 200, calibration.Rows);
    }

    // Out-of-fold probabilities of a tree on twelve rows (a probability of 0.98 on a negative among them): a full Newton
    // step from the starting point overshoots and the slope runs to ~1e7, after which every probability is 0 or 1 and a
    // model that ranked the rows answers one class (a model with AUC 1.0 answering "no" to every row).
    [Fact]
    public void Platt_OnFewRowsWhereAFullNewtonStepOvershoots_StaysFiniteAndFallsInLoss()
    {
        (double, bool, double)[] rows =
        [
            (0.9674, true, 1), (0.1, false, 1), (0.0083, false, 1), (0.03, false, 1), (0.0416, false, 1), (0.1, false, 1),
            (0.98, false, 1), (0.03, false, 1), (0.98, false, 1), (0.03, false, 1), (0.03, false, 1), (0.025, false, 1),
        ];

        var calibration = ProbabilityCalibration.Fit(rows, "platt", "f", null, DateTimeOffset.UtcNow)!;

        Assert.InRange(calibration.A!.Value, -20, 20);
        Assert.InRange(calibration.B!.Value, -20, 20);
        // Not all-zero: the two ends of the model's own ranking stay apart, and the fit is not worse than leaving p as it is.
        Assert.True(calibration.Map(0.9674) > calibration.Map(0.0083) || Math.Abs(calibration.A.Value) < 1e-9);
        Assert.True(Loss(rows, p => calibration.Map(p)) <= Loss(rows, p => p) + 1e-9,
            $"loss {Loss(rows, p => calibration.Map(p))} vs identity {Loss(rows, p => p)}");
    }

    private static double Loss((double P, bool Y, double W)[] rows, Func<double, double> map)
        => -rows.Sum(r => r.W * Math.Log(Math.Clamp(r.Y ? map(r.P) : 1 - map(r.P), 1e-15, 1)));

    [Fact]
    public void Platt_OnRowsTheModelSeparates_StaysAtPlattsTargets()
    {
        // Every row the model was sure of was right. With targets of 1 and 0 the fit has no finite slope and
        // ends at probabilities of 0 and 1; Platt's targets put it at (N₊ + 1) / (N₊ + 2) and 1 / (N₋ + 2).
        var rows = Enumerable.Range(0, 300).Select(i => (0.66, true, 1.0))
            .Concat(Enumerable.Range(0, 300).Select(i => (0.34, false, 1.0))).ToList();

        var calibration = ProbabilityCalibration.Fit(rows, "platt", "f", null, DateTimeOffset.UtcNow)!;

        Assert.True(double.IsFinite(calibration.A!.Value) && calibration.A < 20, $"slope {calibration.A}");
        Assert.Equal(301.0 / 302, calibration.Map(0.66), 3);
        Assert.Equal(1.0 / 302, calibration.Map(0.34), 3);
    }

    [Fact]
    public void Isotonic_KeepsTheOrder()
    {
        var calibration = ProbabilityCalibration.Fit(Overconfident(), "isotonic", "f", null, DateTimeOffset.UtcNow)!;

        var mapped = Enumerable.Range(0, 101).Select(i => calibration.Map(i / 100.0)).ToList();
        Assert.True(mapped.Zip(mapped.Skip(1)).All(pair => pair.Second >= pair.First));
    }

    [Fact]
    public void Weights_CountInTheFit()
    {
        // At 0.5 one positive and one negative; the negative weighs nine.
        var calibration = ProbabilityCalibration.Fit([(0.5, true, 1.0), (0.5, false, 9.0)], "isotonic", "f", "w", DateTimeOffset.UtcNow)!;

        Assert.Equal(0.1, calibration.Map(0.5), 9);
    }

    [Fact]
    public void OneClassOnly_CarriesNoCalibration()
    {
        Assert.Null(ProbabilityCalibration.Fit([(0.9, true, 1.0), (0.2, true, 1.0)], "platt", "f", null, DateTimeOffset.UtcNow));
    }

    [Fact]
    public void ItIsReadBackAsWritten()
    {
        var dir = Path.Combine(Path.GetTempPath(), "mloop-calibration-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var model = Path.Combine(dir, "model.zip");
            var calibration = ProbabilityCalibration.Fit(Overconfident(), "isotonic", "recheck.csv", "inv_pi", DateTimeOffset.UtcNow)!;
            calibration.WriteBeside(model);

            var read = ProbabilityCalibration.ReadBeside(model)!;
            Assert.Equal(calibration.Map(0.37), read.Map(0.37), 12);
            Assert.Equal("inv_pi", read.WeightColumn);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    // Three classes, a model that puts 0.9 on its answer and is right 60% of the time: overconfident.
    private static List<(float[], int, double)> OverconfidentMulticlass()
    {
        var rows = new List<(float[], int, double)>();
        for (var i = 0; i < 300; i++)
        {
            var answer = i % 3;
            var p = new float[3];
            for (var k = 0; k < 3; k++) p[k] = k == answer ? 0.9f : 0.05f;
            var label = i % 10 < 6 ? answer : (answer + 1) % 3;
            rows.Add((p, label, 1.0));
        }
        return rows;
    }

    [Fact]
    public void Temperature_SoftensAnOverconfidentModel_AndKeepsItsAnswers()
    {
        var calibration = ProbabilityCalibration.FitTemperature(OverconfidentMulticlass(), "recheck.csv", null, DateTimeOffset.UtcNow)!;

        Assert.True(calibration.Temperature > 1);
        Assert.True(calibration.EceAfter < calibration.EceBefore / 2, $"ECE {calibration.EceBefore} → {calibration.EceAfter}");
        var tempered = calibration.Temper([0.9f, 0.05f, 0.05f]);
        Assert.Equal(1.0, tempered.Sum(), 5);
        Assert.Equal(0, Array.IndexOf(tempered, tempered.Max()));
        Assert.InRange(tempered[0], 0.5, 0.7);   // right 60% of the time
    }

    private sealed class Flower
    {
        public float X1 { get; set; }
        public float X2 { get; set; }
        public string Species { get; set; } = "";
    }

    /// <summary>
    /// The tempered Score keeps the class names: the vocabulary reads them off Score's annotations, and a
    /// replacement column without them would key every probability by position.
    /// </summary>
    [Fact]
    public void Temperature_KeepsTheClassNamesOnTheScore()
    {
        var ml = new MLContext(seed: 1);
        var flowers = new List<Flower>();
        foreach (var (name, c) in new[] { ("dog", 9f), ("cat", 5f), ("bird", 1f) })
            for (var i = 0; i < 20; i++)
                flowers.Add(new Flower { X1 = c + (i % 5 - 2) * 0.1f, X2 = c, Species = name });
        var data = ml.Data.LoadFromEnumerable(flowers);
        var model = ml.Transforms.Conversion.MapValueToKey("Label", "Species")
            .Append(ml.Transforms.Concatenate("Features", "X1", "X2"))
            .Append(ml.MulticlassClassification.Trainers.LbfgsMaximumEntropy())
            .Fit(data);
        var calibration = ProbabilityCalibration.FitTemperature(OverconfidentMulticlass(), "f", null, DateTimeOffset.UtcNow)!;

        var tempered = calibration.Apply(ml, model).Transform(data);

        var score = tempered.Schema["Score"];
        Assert.Equal(["dog", "cat", "bird"], MulticlassScoreVocabulary.Of(score, 3));
        var first = tempered.GetColumn<VBuffer<float>>(score).First().DenseValues().ToArray();
        Assert.Equal(1.0, first.Sum(), 4);
    }
}
