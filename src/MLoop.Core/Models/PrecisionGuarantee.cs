namespace MLoop.Core.Models;

/// <summary>
/// What a guaranteed threshold stands on: the risk it was certified at and the evidence that certified it.
/// Recorded beside the threshold (<see cref="DecisionThreshold.Guarantee"/>).
/// </summary>
/// <param name="Delta">The chance, at most, that the class's true precision is below the target although the threshold was chosen.</param>
/// <param name="Hits">How many of the answered rows were of the class (a weighted file counts its effective share).</param>
/// <param name="Rows">How many rows the threshold answers as the class (effective number when weighted).</param>
/// <param name="LowerBound">
/// The one-sided Clopper–Pearson limit of the true precision at risk <paramref name="Delta"/>: the true precision is at least this,
/// except in at most <paramref name="Delta"/> of the files the same procedure could be run on. It is never below the target.
/// </param>
public sealed record DecisionGuarantee(double Delta, double Hits, double Rows, double LowerBound);

/// <summary>One threshold <see cref="PrecisionGuarantee.Choose"/> certified.</summary>
public sealed record GuaranteedChoice(double Threshold, double Precision, double Recall, DecisionGuarantee Guarantee);

/// <summary>
/// Choosing a decision threshold whose precision is <i>certified</i> rather than merely observed.
/// </summary>
/// <remarks>
/// <see cref="DecisionThreshold.Choose"/> takes the lowest threshold whose precision reaches the target on the file — the
/// luckiest reading of the file, so the true precision of what it chose is below the target about half the time (150 to
/// 5,000 rows, measured on synthetic scores). This is the same walk with a test at every step: candidate thresholds are
/// taken from the strictest down, in the order of their score values (decided before any label is read), and each is
/// tested against «the true precision is below the target» with a one-sided exact binomial test at risk δ. The walk stops
/// at the first candidate the test cannot reject, and the answer is the last one it rejected. A fixed order of tests that
/// stops at the first failure needs no correction for the number of candidates looked at (fixed-sequence testing, as in
/// Learn then Test): with probability at least 1 − δ every threshold it can return has a true precision of at least the target.
/// <para>
/// A candidate that answers too few rows cannot be rejected whatever its labels: even every answer right, a precision
/// of <c>t</c> is rejected only from <c>⌈ln δ / ln t⌉</c> rows (29 at 0.90 and δ = 0.05, 22 at δ = 0.10). The walk starts at the
/// first candidate that answers that many — a start that depends on scores and weights only, so it is as fixed as the order.
/// A target of 1 is never certified.
/// </para>
/// Weighted rows count as their effective number (Kish), as the Wilson bound beside the threshold does.
/// </remarks>
public static class PrecisionGuarantee
{
    /// <summary>
    /// The fewest rows a threshold must answer before a precision of <paramref name="target"/> can be certified at risk
    /// <paramref name="delta"/> — all of them right. <c>null</c> when none can (a target of 1).
    /// </summary>
    public static int? MinimumRows(double target, double delta)
    {
        if (!(target > 0 && target < 1 && delta > 0 && delta < 1))
            return null;
        // The smallest n with target^n ≤ δ. Computed by stepping, not by the logarithm, so a boundary like 0.9^29 is exact.
        for (var n = 1; n < 100_000_000; n++)
            if (Math.Pow(target, n) <= delta)
                return n;
        return null;
    }

    /// <summary>P(X ≥ <paramref name="hits"/>) for X ~ Binomial(<paramref name="rows"/>, <paramref name="p"/>); counts may be fractional (effective numbers).</summary>
    public static double TailAtLeast(double hits, double rows, double p)
    {
        if (hits <= 0) return 1;
        if (hits > rows) return 0;
        if (p <= 0) return 0;
        if (p >= 1) return 1;
        return RegularizedIncompleteBeta(hits, rows - hits + 1, p);
    }

    /// <summary>The one-sided lower confidence limit (Clopper–Pearson) of a probability that showed <paramref name="hits"/> in <paramref name="rows"/> at risk <paramref name="delta"/>.</summary>
    public static double LowerBound(double hits, double rows, double delta)
    {
        if (hits <= 0 || rows <= 0) return 0;
        // P(X ≥ hits | p) rises with p: the limit is the p where it equals δ.
        double lo = 0, hi = 1;
        for (var i = 0; i < 200; i++)
        {
            var mid = (lo + hi) / 2;
            if (TailAtLeast(hits, rows, mid) > delta) hi = mid; else lo = mid;
        }
        return (lo + hi) / 2;
    }

    /// <summary>
    /// The lowest threshold of one class whose precision is certified at least <paramref name="target"/> at risk
    /// <paramref name="delta"/>, or <c>null</c> when none is — the fixed-sequence walk described on this type. Rows are
    /// (probability, label, weight); the threshold sits halfway between the last probability answered as the class and the
    /// next, as <see cref="DecisionThreshold.Choose"/> places it.
    /// </summary>
    public static GuaranteedChoice? Choose(
        IReadOnlyList<(double Probability, bool Positive, double Weight)> rows, double target, double delta, bool ofPositive = true)
    {
        ArgumentNullException.ThrowIfNull(rows);
        if (!(target > 0 && target < 1 && delta > 0 && delta < 1))
            return null;
        var finite = rows.Where(r => double.IsFinite(r.Probability) && r.Weight > 0);
        var usable = (ofPositive ? finite.OrderByDescending(r => r.Probability) : finite.OrderBy(r => r.Probability)).ToList();
        var ofClass = usable.Where(r => r.Positive == ofPositive).Sum(r => r.Weight);
        if (ofClass <= 0)
            return null;

        GuaranteedChoice? chosen = null;
        double hit = 0, miss = 0, sumSquares = 0;
        var started = false;
        for (var i = 0; i < usable.Count; i++)
        {
            var w = usable[i].Weight;
            if (usable[i].Positive == ofPositive) hit += w; else miss += w;
            sumSquares += w * w;
            if (i + 1 < usable.Count && usable[i + 1].Probability == usable[i].Probability)
                continue;

            var sum = hit + miss;
            var effective = sum * sum / sumSquares;
            var precision = hit / sum;
            var effectiveHits = precision * effective;
            // Before the walk can start, a candidate that cannot be rejected even with every answer right is passed over.
            if (!started)
            {
                if (Math.Pow(target, effective) > delta)
                    continue;
                started = true;
            }
            if (TailAtLeast(effectiveHits, effective, target) > delta)
                break;

            var next = i + 1 < usable.Count ? usable[i + 1].Probability : ofPositive ? 0.0 : 1.0;
            chosen = new GuaranteedChoice((usable[i].Probability + next) / 2, precision, hit / ofClass,
                new DecisionGuarantee(delta, effectiveHits, effective, LowerBound(effectiveHits, effective, delta)));
        }
        return chosen;
    }

    // Regularized incomplete beta I_x(a, b) — Lentz's continued fraction (Numerical Recipes §6.4), the symmetry
    // relation choosing the side that converges.
    private static double RegularizedIncompleteBeta(double a, double b, double x)
    {
        if (x <= 0) return 0;
        if (x >= 1) return 1;
        var front = Math.Exp(LogGamma(a + b) - LogGamma(a) - LogGamma(b) + a * Math.Log(x) + b * Math.Log(1 - x));
        return x < (a + 1) / (a + b + 2)
            ? front * ContinuedFraction(a, b, x) / a
            : 1 - front * ContinuedFraction(b, a, 1 - x) / b;
    }

    private static double ContinuedFraction(double a, double b, double x)
    {
        const double tiny = 1e-300;
        var qab = a + b;
        var qap = a + 1;
        var qam = a - 1;
        var c = 1.0;
        var d = 1 - qab * x / qap;
        if (Math.Abs(d) < tiny) d = tiny;
        d = 1 / d;
        var h = d;
        for (var m = 1; m <= 10_000; m++)
        {
            var m2 = 2 * m;
            var aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d; if (Math.Abs(d) < tiny) d = tiny;
            c = 1 + aa / c; if (Math.Abs(c) < tiny) c = tiny;
            d = 1 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d; if (Math.Abs(d) < tiny) d = tiny;
            c = 1 + aa / c; if (Math.Abs(c) < tiny) c = tiny;
            d = 1 / d;
            var delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1) < 1e-15)
                break;
        }
        return h;
    }

    // Lanczos approximation of ln Γ(x), x > 0.
    private static double LogGamma(double x)
    {
        double[] coefficients =
        [
            57.1562356658629235, -59.5979603554754912, 14.1360979747417471, -0.491913816097620199,
            .339946499848118887e-4, .465236289270485756e-4, -.983744753048795646e-4, .158088703224912494e-3,
            -.210264441724104883e-3, .217439618115212643e-3, -.164318106536763890e-3, .844182239838527433e-4,
            -.261908384015814087e-4, .368991826595316234e-5
        ];
        var y = x;
        var tmp = x + 5.24218750000000000;
        tmp = (x + 0.5) * Math.Log(tmp) - tmp;
        var ser = 0.999999999999997092;
        foreach (var coefficient in coefficients)
            ser += coefficient / ++y;
        return tmp + Math.Log(2.5066282746310005 * ser / x);
    }
}
