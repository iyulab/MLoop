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

    /// <summary>The window for a series of <paramref name="seriesLength"/> points when the project names none.</summary>
    public static int Default(int seriesLength)
        => Math.Clamp(seriesLength / 4, Minimum, MaximumDefault);
}
