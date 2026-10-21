namespace MLoop.Core.Prediction;

/// <summary>
/// The one value a prediction log records as a row's answer: the predicted class, else the predicted
/// value, else the cluster, else the anomaly verdict. Served predictions are logged with it, and the
/// answer profile a model is trained with is built with it — the two must read the answer the same
/// way, or every answer lands in a bin the reference never had and reads as drift.
/// </summary>
/// <remarks>
/// <c>mloop predict --log</c> reads the same answer from the CSV it wrote (<c>PredictedLabel</c>,
/// else <c>Score</c>) — a file, not a <see cref="PredictionRow"/>. The two agree because both paths
/// spell a class and a value the same way, which the cross-path rendering tests pin.
/// </remarks>
public static class PredictionLogOutput
{
    /// <summary>The answer <paramref name="row"/> is logged as.</summary>
    public static object Of(PredictionRow row)
    {
        ArgumentNullException.ThrowIfNull(row);
        return (object?)row.PredictedLabel ?? (object?)row.Score ?? (object?)row.ClusterId ?? (object?)row.IsAnomaly ?? string.Empty;
    }
}
