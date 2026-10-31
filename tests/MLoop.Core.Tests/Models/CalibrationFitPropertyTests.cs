using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// What every probability-calibration fit owes whatever rows it is given — few of them, separated perfectly, all at one
/// probability, a single positive among many negatives, weights spanning twelve orders of magnitude. The rows are drawn
/// from fixed seeds, so a failure is a row set that can be named, not a flake.
/// </summary>
public class CalibrationFitPropertyTests
{
    private static List<(double P, bool Y, double W)> BinaryRows(Random rng, int shape)
    {
        var n = rng.Next(6, 41);
        var rows = new List<(double, bool, double)>();
        for (var i = 0; i < n; i++)
        {
            var y = rng.NextDouble() < (shape == 3 ? 0.08 : 0.35);
            var p = shape switch
            {
                0 => Math.Clamp((y ? 0.9 : 0.1) + (rng.NextDouble() - 0.5) * 0.1, 0, 1),           // separated
                1 => 0.5,                                                                           // one probability for every row
                2 => rng.NextDouble() < 0.5 ? 0.0 : 1.0,                                            // exact 0 and 1
                3 => rng.NextDouble(),                                                              // a rare positive
                _ => rng.NextDouble() < 0.75 ? (y ? 0.97 : 0.03) : rng.NextDouble(),                // mostly sure, some wrong
            };
            var w = shape == 5 ? Math.Pow(10, rng.Next(-6, 7)) : 1.0;
            rows.Add((p, y, w));
        }
        return rows;
    }

    private static double PlattTargetsLoss(List<(double P, bool Y, double W)> rows, Func<double, double> map)
    {
        var pos = rows.Where(r => r.Y).Sum(r => r.W);
        var neg = rows.Where(r => !r.Y).Sum(r => r.W);
        double tp = (pos + 1) / (pos + 2), tn = 1 / (neg + 2);
        return rows.Sum(r =>
        {
            var q = Math.Clamp(map(r.P), 1e-15, 1 - 1e-15);
            var t = r.Y ? tp : tn;
            return -r.W * (t * Math.Log(q) + (1 - t) * Math.Log(1 - q));
        });
    }

    public static TheoryData<int> Shapes => new() { 0, 1, 2, 3, 4, 5 };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Platt_OnAnyRows_IsFinite_AndNoWorseThanLeavingTheProbabilitiesAlone(int shape)
    {
        var rng = new Random(1000 + shape);
        for (var trial = 0; trial < 300; trial++)
        {
            var rows = BinaryRows(rng, shape);
            var calibration = ProbabilityCalibration.Fit(rows, "platt", "f", null, DateTimeOffset.UtcNow);
            if (calibration is null) { Assert.True(rows.All(r => r.Y) || rows.All(r => !r.Y)); continue; }

            Assert.True(double.IsFinite(calibration.A!.Value) && double.IsFinite(calibration.B!.Value), $"shape {shape} trial {trial}: {calibration.A}, {calibration.B}");
            // Weights twelve orders of magnitude apart leave one or two rows deciding the fit, and a fit that separates two close
            // heavy rows has a large slope that is the data's answer; every other shape pins both numbers.
            if (shape != 5)
                Assert.True(Math.Abs(calibration.A.Value) < 60 && Math.Abs(calibration.B.Value) < 60, $"shape {shape} trial {trial}: slope {calibration.A} intercept {calibration.B}");
            Assert.True(PlattTargetsLoss(rows, calibration.Map) <= PlattTargetsLoss(rows, p => p) + 1e-6,
                $"shape {shape} trial {trial}: the fit is worse than the identity");
            foreach (var p in new[] { 0.0, 0.2, 0.5, 0.8, 1.0 })
                Assert.InRange(calibration.Map(p), 0, 1);
        }
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Isotonic_OnAnyRows_IsMonotone_AndStaysInsideZeroAndOne(int shape)
    {
        var rng = new Random(2000 + shape);
        for (var trial = 0; trial < 300; trial++)
        {
            var rows = BinaryRows(rng, shape);
            var calibration = ProbabilityCalibration.Fit(rows, "isotonic", "f", null, DateTimeOffset.UtcNow);
            if (calibration is null) continue;

            var previous = double.NegativeInfinity;
            for (var i = 0; i <= 20; i++)
            {
                var mapped = calibration.Map(i / 20.0);
                Assert.True(double.IsFinite(mapped) && mapped >= 0 && mapped <= 1, $"shape {shape} trial {trial}: {mapped} at {i / 20.0}");
                Assert.True(mapped >= previous - 1e-12, $"shape {shape} trial {trial}: not monotone at {i / 20.0}");
                previous = mapped;
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void Temperature_OnAnyRows_StaysInsideItsRange_AndNoWorseThanTemperatureOne(int shape)
    {
        var rng = new Random(3000 + shape);
        for (var trial = 0; trial < 300; trial++)
        {
            var classes = rng.Next(3, 6);
            var rows = new List<(float[] Probabilities, int Label, double Weight)>();
            for (var i = 0; i < rng.Next(6, 41); i++)
            {
                var label = rng.Next(classes);
                var probabilities = new float[classes];
                switch (shape)
                {
                    case 0: probabilities[label] = 0.97f; break;                                            // sure and right
                    case 1: Array.Fill(probabilities, 1f / classes); break;                                 // uniform
                    case 2: probabilities[rng.Next(classes)] = 1f; break;                                    // one-hot, often wrong
                    default: for (var c = 0; c < classes; c++) probabilities[c] = (float)rng.NextDouble(); break;
                }
                var total = probabilities.Sum();
                for (var c = 0; c < classes; c++) probabilities[c] = total > 0 ? probabilities[c] / total : 1f / classes;
                rows.Add((probabilities, label, 1.0));
            }

            var calibration = ProbabilityCalibration.FitTemperature(rows, "f", null, DateTimeOffset.UtcNow)!;

            Assert.InRange(calibration.Temperature!.Value, 0.05 - 1e-6, 20 + 1e-6);
            double Loss(Func<float[], float[]> map) => rows.Sum(r => -r.Weight * Math.Log(Math.Max(map(r.Probabilities)[r.Label], 1e-12)));
            Assert.True(Loss(p => calibration.Temper(p)) <= Loss(p => p) + 1e-6, $"shape {shape} trial {trial}: T {calibration.Temperature} is worse than 1");
            Assert.All(calibration.Temper(rows[0].Probabilities), q => Assert.True(float.IsFinite(q) && q >= 0 && q <= 1));
        }
    }
}
