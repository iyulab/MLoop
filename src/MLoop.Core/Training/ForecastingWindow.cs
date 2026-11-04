namespace MLoop.Core.Training;

/// <summary>
/// The window of a forecasting model's singular spectrum analysis (SSA) when the project does not name one.
///
/// The window used to be a quarter of the series, whatever its length. SSA's cost grows with the cube of the window
/// (a decomposition of a window-by-window matrix) and its memory with the square, so the default made the cost of
/// forecasting grow faster than the series: a series of 10,000 points took 38 s, one of 100,000 had not finished
/// after 30 minutes, and one of a million failed at once. It was also not a better window. A longer window than the
/// season needs adds components that are noise, and the recurrence SSA forecasts with can then blow up: on a series
/// of 10,000 points with a season of 12, a window of 128 or 256 forecast all five test series to within the
/// noise, a window of 512 missed the season on two of five, and a window of 1,024 diverged on three of five (errors
/// from thousands to hundreds of billions).
///
/// So the default is a quarter of the series, up to <see cref="MaximumDefault"/>: a series short enough for the
/// quarter to be that small is read as before. A series with a longer season than the window can hold names its
/// window in <c>mloop.yaml</c> (<c>window_size</c>).
/// </summary>
public static class ForecastingWindow
{
    /// <summary>The largest window taken when none is named; the measurement above is its reason.</summary>
    public const int MaximumDefault = 256;

    /// <summary>The smallest window SSA accepts.</summary>
    public const int Minimum = 2;

    /// <summary>The window for a series of <paramref name="seriesLength"/> points when the project names none.</summary>
    public static int Default(int seriesLength)
        => Math.Clamp(seriesLength / 4, Minimum, MaximumDefault);
}
