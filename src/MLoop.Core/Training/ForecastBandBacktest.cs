using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Training;

/// <summary>
/// Scores a forecasting recipe on stretches of the training series it has not seen, to give the band the width it states
/// (<see cref="ForecastBandCalibration"/>). The model itself has one such stretch — the horizon it holds out — which is
/// too few to say how wide a band must be; the same recipe (window, horizon, no stabilization) is fitted again on the
/// series before each of several earlier origins and asked to forecast what followed, and every point it forecast is
/// scored by how many native half-widths it missed by.
/// </summary>
internal static class ForecastBandBacktest
{
    /// <summary>Origins before the model's own held-out stretch. The stretches do not overlap: each is one horizon before the next.</summary>
    internal static int Origins => int.TryParse(Environment.GetEnvironmentVariable("MLOOP_BAND_ORIGINS"), out var o) ? o : 8;

    /// <summary>
    /// The most points a backtest fit reads: the ones just before its origin. A fit costs time in proportion to its length
    /// (a million points took minutes), and eight of them on the full series would multiply the run; the recent points are
    /// the ones the forecast is made from.
    /// </summary>
    internal static int SeriesCap => int.TryParse(Environment.GetEnvironmentVariable("MLOOP_BAND_CAP"), out var c) ? c : 20_000;

    private sealed class Point
    {
        public float Value { get; set; }
    }

    /// <summary>One forecast and the band it came with.</summary>
    internal readonly record struct Forecast(float[] Values, float[] Lower, float[] Upper);

    /// <summary>
    /// The calibration of the band, or null where the series cannot give enough scored points (then the native band stands).
    /// </summary>
    /// <param name="mlContext">Context the backtest fits run in.</param>
    /// <param name="series">Every value of the training series, in order.</param>
    /// <param name="trainSize">The points the model was fitted on; the horizon after them is its held-out stretch.</param>
    /// <param name="horizon">Steps each forecast looks ahead.</param>
    /// <param name="windowSize">The SSA window of the run.</param>
    /// <param name="ownForecast">The model's own forecast at <paramref name="trainSize"/>, of the held-out stretch (null: it gave no band).</param>
    /// <param name="configuredSeriesLength">The series length the run was asked to use (0: the fit's own length).</param>
    /// <param name="cancellationToken">Checked between fits.</param>
    internal static ForecastBandCalibration? Calibrate(
        MLContext mlContext,
        IReadOnlyList<float> series,
        int trainSize,
        int horizon,
        int windowSize,
        int configuredSeriesLength,
        Forecast? ownForecast,
        CancellationToken cancellationToken)
    {
        var scores = new List<double>();
        var origins = 0;

        if (ownForecast is { } own && trainSize + horizon <= series.Count)
        {
            origins++;
            AddScores(scores, series, trainSize, horizon, own);
        }

        for (var k = 1; k <= Origins; k++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var origin = trainSize - k * horizon;
            var start = Math.Max(0, origin - SeriesCap);
            var length = origin - start;
            if (length <= 2 * windowSize)
                break;   // the earlier origins are shorter still

            var forecast = FitAndForecast(mlContext, series, start, length, horizon, windowSize, configuredSeriesLength);
            origins++;
            AddScores(scores, series, origin, horizon, forecast);
        }

        return ForecastBandCalibration.Fit(scores, ForecastOutput.ConfidenceLevel, origins, horizon, windowSize);
    }

    private static void AddScores(List<double> scores, IReadOnlyList<float> series, int origin, int horizon, Forecast forecast)
    {
        var steps = Math.Min(horizon, Math.Min(forecast.Values.Length, Math.Min(forecast.Lower.Length, forecast.Upper.Length)));
        for (var h = 0; h < steps; h++)
        {
            if (ForecastBandCalibration.Score(series[origin + h], forecast.Values[h], forecast.Lower[h], forecast.Upper[h]) is { } score)
                scores.Add(score);
        }
    }

    /// <summary>
    /// The recipe the run fitted — the same window and horizon, with the series length the run kept ahead of its fit by a
    /// horizon — fitted on <c>series[start .. start + length)</c>, and its forecast at the last of those points.
    /// </summary>
    private static Forecast FitAndForecast(
        MLContext mlContext, IReadOnlyList<float> series, int start, int length, int horizon, int windowSize, int configuredSeriesLength)
    {
        var seriesLength = length + horizon;
        if (configuredSeriesLength > 0)
            seriesLength = Math.Max(windowSize + 1, Math.Min(configuredSeriesLength, seriesLength));

        var points = new Point[length];
        for (var i = 0; i < length; i++)
            points[i] = new Point { Value = series[start + i] };
        var view = mlContext.Data.LoadFromEnumerable(points);

        var pipeline = mlContext.Forecasting.ForecastBySsa(
            outputColumnName: ForecastOutput.ForecastColumnName,
            inputColumnName: nameof(Point.Value),
            windowSize: windowSize,
            seriesLength: seriesLength,
            trainSize: length,
            horizon: horizon,
            shouldStabilize: false,
            confidenceLowerBoundColumn: ForecastOutput.LowerBoundColumnName,
            confidenceUpperBoundColumn: ForecastOutput.UpperBoundColumnName,
            confidenceLevel: (float)ForecastOutput.ConfidenceLevel);

        var model = pipeline.Fit(view);
        return LastForecast(model.Transform(view));
    }

    /// <summary>The forecast, and its band, of the last row of <paramref name="scored"/>.</summary>
    internal static Forecast LastForecast(IDataView scored)
    {
        var forecastCol = scored.Schema[ForecastOutput.ForecastColumnName];
        var lowerCol = scored.Schema.GetColumnOrNull(ForecastOutput.LowerBoundColumnName);
        var upperCol = scored.Schema.GetColumnOrNull(ForecastOutput.UpperBoundColumnName);
        var wanted = new List<DataViewSchema.Column> { forecastCol };
        if (lowerCol.HasValue) wanted.Add(lowerCol.Value);
        if (upperCol.HasValue) wanted.Add(upperCol.Value);

        using var cursor = scored.GetRowCursor(wanted);
        var forecastGetter = cursor.GetGetter<VBuffer<float>>(forecastCol);
        var lowerGetter = lowerCol.HasValue ? cursor.GetGetter<VBuffer<float>>(lowerCol.Value) : null;
        var upperGetter = upperCol.HasValue ? cursor.GetGetter<VBuffer<float>>(upperCol.Value) : null;

        VBuffer<float> f = default, l = default, u = default;
        while (cursor.MoveNext())
        {
            forecastGetter(ref f);
            lowerGetter?.Invoke(ref l);
            upperGetter?.Invoke(ref u);
        }

        return new Forecast(f.DenseValues().ToArray(), l.DenseValues().ToArray(), u.DenseValues().ToArray());
    }
}
