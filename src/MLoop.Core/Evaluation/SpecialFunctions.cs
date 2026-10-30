namespace MLoop.Core.Evaluation;

/// <summary>
/// The few special functions the statistics here stand on, written once: ln Γ, the regularized upper incomplete gamma function
/// and, from it, the upper quantile of the χ² distribution.
/// </summary>
internal static class SpecialFunctions
{
    /// <summary>Lanczos approximation of ln Γ(x), x &gt; 0.</summary>
    internal static double LogGamma(double x)
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

    /// <summary>
    /// Q(a, x) = Γ(a, x) ÷ Γ(a), the regularized upper incomplete gamma function — the share of a gamma distribution of shape
    /// <paramref name="a"/> above <paramref name="x"/>. The series below a + 1, the continued fraction above (Numerical Recipes §6.2).
    /// </summary>
    internal static double UpperIncompleteGamma(double a, double x)
    {
        if (x <= 0)
            return 1;
        if (x < a + 1)
        {
            double sum = 1 / a, term = sum;
            for (var n = 1; n < 10_000; n++)
            {
                term *= x / (a + n);
                sum += term;
                if (Math.Abs(term) < Math.Abs(sum) * 1e-16)
                    break;
            }
            return Math.Max(0, 1 - sum * Math.Exp(-x + a * Math.Log(x) - LogGamma(a)));
        }

        const double tiny = 1e-300;
        var b = x + 1 - a;
        var c = 1 / tiny;
        var d = 1 / b;
        var h = d;
        for (var i = 1; i < 10_000; i++)
        {
            var an = -i * (i - a);
            b += 2;
            d = an * d + b;
            if (Math.Abs(d) < tiny) d = tiny;
            c = b + an / c;
            if (Math.Abs(c) < tiny) c = tiny;
            d = 1 / d;
            var delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1) < 1e-16)
                break;
        }
        return Math.Exp(-x + a * Math.Log(x) - LogGamma(a)) * h;
    }

    /// <summary>The value a χ² variable of <paramref name="degreesOfFreedom"/> exceeds with probability <paramref name="tail"/>.</summary>
    internal static double ChiSquareUpperQuantile(int degreesOfFreedom, double tail)
    {
        if (degreesOfFreedom < 1)
            throw new ArgumentOutOfRangeException(nameof(degreesOfFreedom), degreesOfFreedom, "At least one degree of freedom.");
        if (!(tail > 0 && tail < 1))
            throw new ArgumentOutOfRangeException(nameof(tail), tail, "A tail probability lies strictly between 0 and 1.");
        double low = 0, high = degreesOfFreedom + 12 * Math.Sqrt(2.0 * degreesOfFreedom) + 60;
        while (UpperIncompleteGamma(degreesOfFreedom / 2.0, high / 2) > tail)
            high *= 2;
        for (var i = 0; i < 200 && high - low > 1e-12 * high; i++)
        {
            var mid = (low + high) / 2;
            if (UpperIncompleteGamma(degreesOfFreedom / 2.0, mid / 2) > tail) low = mid; else high = mid;
        }
        return (low + high) / 2;
    }
}
