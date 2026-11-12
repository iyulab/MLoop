namespace MLoop.Core.Training;

/// <summary>
/// The window of a forecasting model's singular spectrum analysis (SSA) when the project does not name one.
///
/// The window used to be a quarter of the series, whatever its length. SSA's cost grows with the cube of the window
/// (a decomposition of a window-by-window matrix) and its memory with the square, so the default made the cost of
/// forecasting grow faster than the series: a series of 10,000 points took 38 s, one of 100,000 had not finished
/// after 30 minutes, and one of a million failed at once.
///
/// So the default is a quarter of the series, up to <see cref="MaximumDefault"/>: a series short enough for the
/// quarter to be that small is read as before. The ceiling is the cost, not the accuracy: a window longer than
/// the season needs is no better but is not worse either (the forecasts that diverged with a long window were the
/// stabilization of ML.NET's SSA, which the forecaster now leaves off; with it off windows of 365 to 1,024 forecast
/// daily data with a yearly season as well as the default does), and a season longer than the window is forecast
/// to within the noise with the default. A project that wants a longer window names it in <c>mloop.yaml</c>
/// (<c>window_size</c>), at the cost of time that grows with its cube.
/// </summary>
public static class ForecastingWindow
{
    /// <summary>The largest window taken when none is named; the measurement above is its reason.</summary>
    public const int MaximumDefault = 256;

    /// <summary>The smallest window SSA accepts.</summary>
    public const int Minimum = 2;

    /// <summary>
    /// The fewest rows a series needs to train a window and hold out the horizon: ML.NET wants the training rows (all
    /// but the held-out horizon) to be more than twice the window.
    /// </summary>
    public static int MinimumRows(int windowSize, int horizon) => 2 * windowSize + horizon + 1;

    /// <summary>
    /// What to change when a series of <paramref name="rows"/> points is too short for <paramref name="windowSize"/> and
    /// <paramref name="horizon"/>: the largest horizon this window allows and the largest window this horizon allows, each only when
    /// it exists — or <see langword="null"/> when only more points help. A series shorter than the horizon cannot be helped by a smaller window — saying only that sent a user to the one
    /// setting that could not fix it.
    /// </summary>
    public static string? Remedy(int rows, int windowSize, int horizon)
    {
        var options = new List<string>();
        var longestHorizon = rows - 2 * windowSize - 1;
        if (longestHorizon >= 1)
            options.Add($"a horizon of at most {longestHorizon}");
        var longestWindow = (rows - horizon - 1) / 2;
        if (rows - horizon - 1 >= 2 * Minimum && longestWindow >= Minimum)
            options.Add($"a window_size of at most {longestWindow}");
        if (options.Count == 0)
        {
            var smallestWindowHorizon = rows - 2 * Minimum - 1;
            if (smallestWindowHorizon < 1)
                return null;   // not even the smallest window and a horizon of one fit: only more points help
            options.Add($"both smaller (even the smallest window, {Minimum}, needs a horizon of at most {smallestWindowHorizon})");
        }

        return string.Join(" or ", options);
    }

    /// <summary>The window for a series of <paramref name="seriesLength"/> points when the project names none.</summary>
    public static int Default(int seriesLength)
        => Math.Clamp(seriesLength / 4, Minimum, MaximumDefault);
}
