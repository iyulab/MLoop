using MLoop.Core.Models;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// The certified threshold (<see cref="PrecisionGuarantee"/>): the sample size the arithmetic demands, the exact
/// binomial numbers behind it, the fixed-sequence walk, and — measured, not argued — how often the usual choice and
/// the certified one end below the target they were asked for.
/// </summary>
public sealed class PrecisionGuaranteeTests
{
    private static (double Probability, bool Positive, double Weight) R(double p, bool positive, double weight = 1) => (p, positive, weight);

    /// <summary>Descending probabilities 0.99, 0.98 … with the labels given; a label of <c>false</c> is a wrong answer.</summary>
    private static List<(double Probability, bool Positive, double Weight)> Sequence(params bool[] labels)
        => labels.Select((positive, i) => R(0.99 - i * 0.005, positive)).ToList();

    [Theory]
    [InlineData(0.90, 0.05, 29)]   // 0.9^29 = 0.0471 ≤ 0.05 < 0.9^28 = 0.0523
    [InlineData(0.90, 0.10, 22)]   // 0.9^22 = 0.0985 ≤ 0.10 < 0.9^21 = 0.1094
    [InlineData(0.95, 0.05, 59)]
    [InlineData(0.99, 0.01, 459)]
    public void TheFewestRowsACertificateTakes_IsWhatTheArithmeticSays(double target, double delta, int expected)
        => Assert.Equal(expected, PrecisionGuarantee.MinimumRows(target, delta));

    [Fact]
    public void APrecisionOfOne_IsNeverCertified()
    {
        Assert.Null(PrecisionGuarantee.MinimumRows(1.0, 0.05));
        Assert.Null(PrecisionGuarantee.Choose(Sequence([.. Enumerable.Repeat(true, 500)]), 1.0, 0.05));
    }

    // P(X ≥ k | n, p) by summation — the number the incomplete beta must reproduce.
    private static double Tail(int k, int n, double p)
    {
        double total = 0;
        for (var i = k; i <= n; i++)
            total += Math.Exp(LogChoose(n, i) + i * Math.Log(p) + (n - i) * Math.Log(1 - p));
        return total;
    }

    private static double LogChoose(int n, int k)
    {
        double sum = 0;
        for (var i = 1; i <= k; i++) sum += Math.Log(n - k + i) - Math.Log(i);
        return sum;
    }

    [Theory]
    [InlineData(36, 40, 0.9)]
    [InlineData(29, 29, 0.9)]
    [InlineData(95, 100, 0.9)]
    [InlineData(5, 40, 0.1)]
    [InlineData(480, 500, 0.95)]
    public void TheTail_IsTheExactBinomialOne(int k, int n, double p)
        => Assert.Equal(Tail(k, n, p), PrecisionGuarantee.TailAtLeast(k, n, p), 10);

    [Fact]
    public void TheLowerBoundOfAllRight_IsTheClosedForm_AndTheTailThereIsDelta()
    {
        // n right of n: P(X ≥ n | p) = p^n, so the limit at risk δ is δ^(1/n).
        var bound = PrecisionGuarantee.LowerBound(29, 29, 0.05);
        Assert.Equal(Math.Pow(0.05, 1.0 / 29), bound, 9);
        Assert.Equal(0.05, PrecisionGuarantee.TailAtLeast(29, 29, bound), 9);
        Assert.Equal(0, PrecisionGuarantee.LowerBound(0, 29, 0.05));
    }

    [Fact]
    public void TwentyNineRightAnswers_CertifyNinety_AtFivePercent_AndTwentyEightDoNot()
    {
        Assert.NotNull(PrecisionGuarantee.Choose(Sequence([.. Enumerable.Repeat(true, 29)]), 0.90, 0.05));
        Assert.Null(PrecisionGuarantee.Choose(Sequence([.. Enumerable.Repeat(true, 28)]), 0.90, 0.05));
        Assert.NotNull(PrecisionGuarantee.Choose(Sequence([.. Enumerable.Repeat(true, 22)]), 0.90, 0.10));
        Assert.Null(PrecisionGuarantee.Choose(Sequence([.. Enumerable.Repeat(true, 21)]), 0.90, 0.10));
    }

    [Fact]
    public void TheWalkStopsAtTheFirstThresholdItCannotReject_AndAnswersTheOneBefore()
    {
        // 29 right, one wrong, then thirty right: 29 of 30 is not enough at 0.90 / 0.05 (P = 0.18), and the walk does not look past it.
        var rows = Sequence([.. Enumerable.Repeat(true, 29), false, .. Enumerable.Repeat(true, 30)]);

        var chosen = PrecisionGuarantee.Choose(rows, 0.90, 0.05)!;

        Assert.Equal((0.99 - 28 * 0.005 + (0.99 - 29 * 0.005)) / 2, chosen.Threshold, 9);   // halfway below the 29th row
        Assert.Equal(1.0, chosen.Precision);
        Assert.Equal(29.0 / 59, chosen.Recall, 9);
        Assert.Equal((0.05, 29.0, 29.0), (chosen.Guarantee.Delta, chosen.Guarantee.Hits, chosen.Guarantee.Rows));
        Assert.True(chosen.Guarantee.LowerBound >= 0.90);
    }

    [Fact]
    public void ACertifiedThreshold_IsNeverLooserThanTheUsualOne_AndItsBoundNeverBelowTheTarget()
    {
        var rng = new Random(7);
        var rows = Enumerable.Range(0, 600).Select(_ => { var s = 0.5 + rng.NextDouble() / 2; return R(s, rng.NextDouble() < s); }).ToList();

        var usual = DecisionThreshold.Choose(rows, 0.85)!.Value;
        var certified = PrecisionGuarantee.Choose(rows, 0.85, 0.05);

        Assert.NotNull(certified);
        Assert.True(certified!.Threshold >= usual.Threshold);
        Assert.True(certified.Guarantee.LowerBound >= 0.85);
    }

    [Fact]
    public void RowsOfOneWeight_AreWhatUnweightedRowsAre()
    {
        var plain = Sequence([.. Enumerable.Repeat(true, 40), false, .. Enumerable.Repeat(true, 20)]);
        var doubled = plain.Select(r => (r.Probability, r.Positive, 2.0)).ToList();

        var a = PrecisionGuarantee.Choose(plain, 0.9, 0.05)!;
        var b = PrecisionGuarantee.Choose(doubled, 0.9, 0.05)!;

        Assert.Equal(a.Threshold, b.Threshold, 12);
        Assert.Equal(a.Guarantee.Rows, b.Guarantee.Rows, 9);   // the effective number, not the weight
    }

    [Fact]
    public void TheNegativeClass_IsCertifiedFromTheBottom()
    {
        // The mirror image: probabilities ascending, the class «negative» answered below the threshold.
        var rows = Enumerable.Range(0, 29).Select(i => R(0.01 + i * 0.005, false)).Concat([R(0.9, true)]).ToList();

        var chosen = PrecisionGuarantee.Choose(rows, 0.90, 0.05, ofPositive: false)!;

        Assert.Equal((0.01 + 28 * 0.005 + 0.9) / 2, chosen.Threshold, 9);   // between the last negative and the positive above
        Assert.Equal(1.0, chosen.Precision);
        Assert.Equal(29.0, chosen.Guarantee.Rows);
    }

    [Fact]
    public void AClassWithNoRows_HasNothingToCertify()
        => Assert.Null(PrecisionGuarantee.Choose([R(0.9, false), R(0.8, false)], 0.9, 0.05));

    /// <summary>
    /// The reason the mode exists, measured on scores that are exactly calibrated (P(yes | score) = score, score uniform on
    /// [0.5, 1]): a threshold t then has a true precision of (t + 1) / 2, so «below 0.9» is «t below 0.8». The usual choice
    /// lands below the target in a large share of files; the certified one in at most about δ of them.
    /// </summary>
    [Fact]
    public void TheCertifiedThreshold_MissesItsTargetRarely_WhereTheUsualOneMissesOftenOnTheSameFiles()
    {
        const double target = 0.9, delta = 0.10;
        const int files = 300, rowsPerFile = 1000;
        var rng = new Random(20261029);
        int usualBelow = 0, usualChosen = 0, certifiedBelow = 0, certifiedChosen = 0;
        for (var f = 0; f < files; f++)
        {
            var rows = new List<(double Probability, bool Positive, double Weight)>(rowsPerFile);
            for (var i = 0; i < rowsPerFile; i++)
            {
                var s = 0.5 + rng.NextDouble() / 2;
                rows.Add(R(s, rng.NextDouble() < s));
            }
            if (DecisionThreshold.Choose(rows, target) is { } usual)
            {
                usualChosen++;
                if ((usual.Threshold + 1) / 2 < target) usualBelow++;
            }
            if (PrecisionGuarantee.Choose(rows, target, delta) is { } certified)
            {
                certifiedChosen++;
                if ((certified.Threshold + 1) / 2 < target) certifiedBelow++;
            }
        }

        Assert.True(usualChosen > 250, $"usual chose {usualChosen}");
        Assert.True(certifiedChosen > 100, $"certified chose {certifiedChosen}");   // a guarantee that is always refused would pass the next line for nothing
        Assert.True(usualBelow / (double)usualChosen > 0.30, $"usual below target in {usualBelow}/{usualChosen}");
        Assert.True(certifiedBelow / (double)files <= delta + 0.05, $"certified below target in {certifiedBelow}/{files}");
    }
}
