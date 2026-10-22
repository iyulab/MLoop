namespace MLoop.Search.Tuning;

/// <summary>
/// The tuner's random numbers — uniform, normal (Box–Muller) and categorical draws over
/// <see cref="Random"/>, consumed in exactly the order ML.NET AutoML's <c>RandomNumberGenerator</c>
/// consumes them, so a seeded search proposes what the upstream one did.
/// </summary>
public sealed class TunerRandom
{
    private readonly Random _random;

    /// <param name="seed">The seed; <see langword="null"/> draws a different sequence every run.</param>
    public TunerRandom(int? seed = null) => _random = seed is { } s ? new Random(s) : new Random();

    /// <summary>A uniform draw in [<paramref name="low"/>, <paramref name="high"/>).</summary>
    public double Uniform(double low, double high) => (_random.NextDouble() * (high - low)) + low;

    /// <summary>A normal draw with <paramref name="location"/> and <paramref name="scale"/>.</summary>
    public double Normal(double location, double scale)
    {
        var u = 1 - Uniform(0, 1);
        var v = 1 - Uniform(0, 1);
        var std = Math.Sqrt(-2.0 * Math.Log(u)) * Math.Sin(2.0 * Math.PI * v);
        return location + (std * scale);
    }

    /// <summary><paramref name="size"/> normal draws.</summary>
    public double[] Normal(double location, double scale, int size)
    {
        var values = new double[size];
        for (var i = 0; i < size; i++)
            values[i] = Normal(location, scale);
        return values;
    }
}
