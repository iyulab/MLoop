namespace MLoop.Core.Prediction;

public class PredictionResult
{
    public required string TaskType { get; init; }
    public required List<PredictionRow> Rows { get; init; }
    public PredictionMetadata? Metadata { get; init; }
    public List<string>? Warnings { get; init; }
}

public record PredictionRow
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
    // Coverage LEVEL of the band (e.g. 0.90), NOT a per-row confidence — see Confidence below for that.
    public double? IntervalConfidence { get; init; }

    /// <summary>
    /// Normalized per-row confidence ∈ [0,1], derived by <see cref="ConfidencePolicy"/> from this row's
    /// task-family uncertainty signal (classification → max class probability, anomaly → decision-boundary
    /// distance, regression → conformal band width vs residual scale). MLoop owns this rule so consumers
    /// stop re-deriving it — do not confuse with <see cref="IntervalConfidence"/> (the band's coverage
    /// level). Null when the model exposes no usable confidence signal.
    /// </summary>
    public double? Confidence { get; init; }
}

/// <summary>
/// A regression prediction interval threaded from the stored metrics into
/// <c>PredictionService.Predict</c> so the regression rows can surface <c>[Score ± width]</c>.
/// Two modes, decided by <see cref="FromMetrics"/>:
/// <list type="bullet">
/// <item><b>Constant-width</b> (split-conformal, <c>interval_half_width_{pct}</c>): every row gets the
/// same <see cref="HalfWidth"/>.</item>
/// <item><b>Heteroscedastic</b> (normalized conformal, <c>norm_interval_q_{pct}</c> + <c>interval_beta</c>
/// plus a saved <c>residual-model.zip</c>): the width varies per row as
/// <c>q · (max(σ(x), 0) + β)</c> — wide where the σ-model predicts a large residual — so the band width
/// is itself the regression escalate signal (cycle-134 M-13). Falls back to constant-width when no
/// residual model is available for a row.</item>
/// </list>
/// </summary>
public record RegressionInterval(double HalfWidth, double Confidence, double? NormalizedQ = null, double? Beta = null, double? ResidualStd = null)
{
    /// <summary>
    /// The primary coverage level exposed at predict time. Training stores the 80/90/95 bands
    /// (<see cref="AutoML.AutoMLRunner.ComputeConformalIntervals"/>); predict surfaces this one.
    /// </summary>
    public const double PrimaryConfidence = 0.90;

    /// <summary>True when this interval carries the normalized-conformal parameters for per-row widths
    /// (a <c>residual-model.zip</c> must also be loaded to produce σ(x)).</summary>
    public bool IsHeteroscedastic => NormalizedQ.HasValue && Beta.HasValue;

    /// <summary>The half-width for one row given its σ-model raw output. Heteroscedastic:
    /// <c>q · (max(σ_raw, 0) + β)</c>; otherwise the constant <see cref="HalfWidth"/>.</summary>
    public double WidthFor(double sigmaRaw) =>
        IsHeteroscedastic ? NormalizedQ!.Value * (Math.Max(sigmaRaw, 0.0) + Beta!.Value) : HalfWidth;

    /// <summary>
    /// Builds the primary prediction interval from stored experiment metrics, or <c>null</c> when the
    /// task is not regression or the model carries no conformal band. Prefers the heteroscedastic band
    /// (normalized quantile + β) when present, else the constant-width band. Single source for both the
    /// serve <c>/predict</c> endpoint and the CLI <c>mloop predict</c> path, so the two never drift on
    /// which band key/level they expose.
    /// </summary>
    public static RegressionInterval? FromMetrics(string? taskType, IReadOnlyDictionary<string, double>? metrics)
    {
        if (taskType != "regression" || metrics is null)
            return null;

        int pct = (int)Math.Round(PrimaryConfidence * 100);

        // residual_std (typical error scale) normalizes the regression band into a [0,1] confidence in
        // ConfidencePolicy — MLoop owns it, consumers don't easily have it. Absent on old models → null.
        double? residualStd = metrics.TryGetValue("residual_std", out var rs) ? rs : null;

        // Heteroscedastic band takes precedence: it needs the constant-width half-width as its fallback
        // (rows the σ-model can't score) plus the normalized quantile and β.
        if (metrics.TryGetValue($"interval_half_width_{pct}", out var halfWidth))
        {
            if (metrics.TryGetValue($"norm_interval_q_{pct}", out var q) &&
                metrics.TryGetValue("interval_beta", out var beta))
                return new RegressionInterval(halfWidth, PrimaryConfidence, q, beta, residualStd);

            return new RegressionInterval(halfWidth, PrimaryConfidence, ResidualStd: residualStd);
        }

        return null;
    }
}

public record PredictionMetadata(string ModelName, string ExperimentId, DateTime Timestamp);
