using System.Text.Json;
using MLoop.Core.Storage;

namespace MLoop.Core.Models;

/// <summary>
/// The width a forecasting model's prediction band has to be given to hold the level it states, measured on forecasts the
/// model's recipe made of points it had not seen. ML.NET's singular spectrum analysis sets the band from the spread of
/// the model's own fit, which is smaller than the error on points that follow the series, so a band asked for at 0.95
/// held about 0.88 of them. The scale is a split-conformal quantile of how many half-widths each of those forecasts
/// missed by: the band is the native one, widened (or narrowed) about the forecast by that factor.
/// </summary>
/// <remarks>
/// Lives beside the model (<see cref="ExperimentLayout.ForecastBandFileName"/>), is applied by the one place that
/// replays a forecast (<c>ForecastReplayService</c>), and is copied on promote like the other layers. The scale is of the
/// native half-width, so the band keeps the shape SSA gives it (it widens with the step); one number is fitted, not one
/// per step — a step has too few scores for a quantile of its own. Absent (a model trained before it, or a series too
/// short for backtests): the native band.
/// </remarks>
/// <param name="Level">The coverage the scale was fitted to — the level the band states.</param>
/// <param name="Scale">Factor on the native half-width.</param>
/// <param name="Origins">Forecast origins the scores come from: the model's own held-out stretch plus the backtests before it.</param>
/// <param name="Scores">Scored points (origins × steps that had a usable native band).</param>
/// <param name="Horizon">Steps each origin looked ahead.</param>
/// <param name="Window">SSA window of the fits.</param>
/// <param name="FitAt">When.</param>
public sealed record ForecastBandCalibration(
    double Level,
    double Scale,
    int Origins,
    int Scores,
    int Horizon,
    int Window,
    DateTimeOffset FitAt)
{
    /// <summary>
    /// Fewest scored points a scale is fitted from. A quantile at 0.95 of fewer lies on the largest score or two — it
    /// would state a width from noise — so the native band stands.
    /// </summary>
    public const int MinimumScores = 20;

    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };

    /// <summary>
    /// How many native half-widths a forecast missed the point it forecast by: <c>|actual − forecast|</c> over the
    /// distance from the forecast to the bound on the side the point fell. Null where the band gives no distance
    /// (a bound that is not finite, or not on its side of the forecast) — such a point says nothing about the width.
    /// </summary>
    public static double? Score(double actual, double forecast, double lower, double upper)
    {
        if (!double.IsFinite(actual) || !double.IsFinite(forecast) || !double.IsFinite(lower) || !double.IsFinite(upper))
            return null;
        var half = actual >= forecast ? upper - forecast : forecast - lower;
        if (!(half > 0))
            return null;
        return Math.Abs(actual - forecast) / half;
    }

    /// <summary>
    /// The split-conformal quantile of <paramref name="scores"/> at <paramref name="level"/>: the ⌈(n+1)·level⌉-th smallest.
    /// When that rank passes the largest score the level cannot be certified from this many scores, and the largest is
    /// returned — the widest the evidence supports.
    /// </summary>
    public static double Quantile(IReadOnlyList<double> scores, double level)
    {
        ArgumentNullException.ThrowIfNull(scores);
        if (scores.Count == 0)
            throw new ArgumentException("At least one score is needed.", nameof(scores));
        var sorted = scores.OrderBy(s => s).ToArray();
        var rank = (int)Math.Ceiling((sorted.Length + 1) * level);
        return sorted[Math.Clamp(rank, 1, sorted.Length) - 1];
    }

    /// <summary>
    /// The calibration <paramref name="scores"/> make, or null when they are fewer than <see cref="MinimumScores"/> —
    /// the native band stands.
    /// </summary>
    public static ForecastBandCalibration? Fit(IReadOnlyList<double> scores, double level, int origins, int horizon, int window)
    {
        ArgumentNullException.ThrowIfNull(scores);
        var usable = scores.Where(double.IsFinite).ToArray();
        if (usable.Length < MinimumScores)
            return null;
        return new ForecastBandCalibration(level, Quantile(usable, level), origins, usable.Length, horizon, window, DateTimeOffset.UtcNow);
    }

    /// <summary>The band about <paramref name="forecast"/> with its distances to the native bounds scaled.</summary>
    public (float Lower, float Upper) Apply(float forecast, float lower, float upper)
    {
        if (!float.IsFinite(forecast) || !float.IsFinite(lower) || !float.IsFinite(upper))
            return (lower, upper);
        return ((float)(forecast - Scale * (forecast - lower)), (float)(forecast + Scale * (upper - forecast)));
    }

    /// <summary><paramref name="forecast"/> with each step's band scaled; the forecast itself is unchanged.</summary>
    public Training.ForecastOutput Apply(Training.ForecastOutput forecast)
    {
        ArgumentNullException.ThrowIfNull(forecast);
        var steps = Math.Min(forecast.ForecastedValues.Length, Math.Min(forecast.LowerBound.Length, forecast.UpperBound.Length));
        var lower = (float[])forecast.LowerBound.Clone();
        var upper = (float[])forecast.UpperBound.Clone();
        for (var i = 0; i < steps; i++)
            (lower[i], upper[i]) = Apply(forecast.ForecastedValues[i], lower[i], upper[i]);
        return new Training.ForecastOutput { ForecastedValues = forecast.ForecastedValues, LowerBound = lower, UpperBound = upper };
    }

    /// <summary>The calibration beside <paramref name="modelPath"/>, or null when there is none.</summary>
    public static ForecastBandCalibration? ReadBeside(string modelPath)
    {
        var path = PathBeside(modelPath);
        return File.Exists(path) ? JsonSerializer.Deserialize<ForecastBandCalibration>(File.ReadAllText(path), Options) : null;
    }

    /// <summary>Writes this calibration beside <paramref name="modelPath"/>.</summary>
    public void WriteBeside(string modelPath) => AtomicFile.WriteAllText(PathBeside(modelPath), JsonSerializer.Serialize(this, Options));

    /// <summary>Where the calibration beside <paramref name="modelPath"/> lives.</summary>
    public static string PathBeside(string modelPath)
        => Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".", ExperimentLayout.ForecastBandFileName);
}
