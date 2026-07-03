namespace MLoop.Core.Prediction;

public class PredictionResult
{
    public required string TaskType { get; init; }
    public required List<PredictionRow> Rows { get; init; }
    public PredictionMetadata? Metadata { get; init; }
    public List<string>? Warnings { get; init; }
}

public class PredictionRow
{
    public string? PredictedLabel { get; init; }
    public Dictionary<string, double>? Probabilities { get; init; }
    public double? Score { get; init; }
    public int? ClusterId { get; init; }
    public double[]? Distances { get; init; }
    public bool? IsAnomaly { get; init; }
    public double? AnomalyScore { get; init; }

    // ② regression wave: split-conformal prediction interval around Score. Null unless the model
    // carries a conformal band (regression only, computed at train time — see AutoMLRunner.ComputeConformalIntervals).
    // Gives the regression trust-loop signal the confidence probability gives classification.
    public double? ScoreLowerBound { get; init; }
    public double? ScoreUpperBound { get; init; }
    public double? IntervalConfidence { get; init; }
}

/// <summary>
/// A split-conformal regression prediction interval half-width and its coverage level, threaded
/// from the stored metrics (<c>interval_half_width_{pct}</c>) into <c>PredictionService.Predict</c>
/// so the regression rows can surface <c>[Score - HalfWidth, Score + HalfWidth]</c>.
/// </summary>
public record RegressionInterval(double HalfWidth, double Confidence)
{
    /// <summary>
    /// The primary coverage level exposed at predict time. Training stores the 80/90/95 bands
    /// (<see cref="AutoML.AutoMLRunner.ComputeConformalIntervals"/>); predict surfaces this one.
    /// </summary>
    public const double PrimaryConfidence = 0.90;

    /// <summary>
    /// Builds the primary prediction interval from stored experiment metrics, or <c>null</c> when the
    /// task is not regression or the model carries no conformal band. Single source for both the serve
    /// <c>/predict</c> endpoint and the CLI <c>mloop predict</c> path, so the two never drift on which
    /// band key/level they expose.
    /// </summary>
    public static RegressionInterval? FromMetrics(string? taskType, IReadOnlyDictionary<string, double>? metrics)
    {
        if (taskType != "regression" || metrics is null)
            return null;

        int pct = (int)Math.Round(PrimaryConfidence * 100);
        return metrics.TryGetValue($"interval_half_width_{pct}", out var halfWidth)
            ? new RegressionInterval(halfWidth, PrimaryConfidence)
            : null;
    }
}

public record PredictionMetadata(string ModelName, string ExperimentId, DateTime Timestamp);
