namespace MLoop.Core.Prediction;

/// <summary>
/// Single authority for the normalized per-row prediction confidence ∈ [0,1]. Which output field carries
/// the uncertainty signal for a task family — and how it maps to a scalar confidence — is MLoop domain
/// knowledge, so MLoop owns it here rather than each consumer (U-Vision, HoneAI, mloop-mcp) re-deriving it
/// from raw <c>probabilities</c>/<c>score</c> and drifting apart. <see cref="PredictionService"/> fills
/// <see cref="PredictionRow.Confidence"/> with this; the raw signals stay on the row so callers that want a
/// different mapping still can.
///
/// Task-family rules:
/// <list type="bullet">
/// <item><b>Classification</b> → the maximum class probability (argmax confidence).</item>
/// <item><b>Anomaly detection</b> → distance from the 0.5 decision boundary, <c>|anomalyScore − 0.5|·2</c>
/// (RandomizedPca Score ∈ [0,1], threshold 0.5 — Microsoft.ML). Confidence is in the anomaly/normal
/// DECISION, so a strong anomaly and a strong inlier are both confident; only boundary scores are not.</item>
/// <item><b>Regression</b> with a conformal band → <c>1 − clamp(halfWidth / (k·residualStd))</c>: a narrow
/// band (model locally certain) is high-confidence, a wide band low. Normalizing by <c>residual_std</c>
/// (the model's typical error scale, from <c>metrics.json</c>) makes this dataset-scale-stable, which is
/// why MLoop is better placed to own it than a consumer using a <c>|Score|</c> heuristic. Regression point
/// estimates without a band carry no confidence signal → null.</item>
/// <item><b>Ranking / recommendation</b> → null. Their Score is a relative ranking score /
/// predicted rating on the label's own scale — clamping it into [0,1] fabricated a constant
/// "fully confident" 1.0 for every recommendation row (a 1–5 rating always clamps to 1) and an
/// arbitrary value for ranking (D25). Honest null until a real order-uncertainty signal
/// (top-k score margin) is designed and measured.</item>
/// </list>
/// </summary>
/// <remarks>
/// <b>Usage boundary — this confidence measures MAGNITUDE uncertainty, not decision-boundary risk.</b>
/// For regression, a row's band width (and hence this scalar confidence) is empirically correlated with
/// the prediction's <i>distance from any fixed decision threshold</i> T, not its <i>proximity</i> to it —
/// on a live KAMP anodizing-thickness set the Pearson correlation between band width and |score − T| was
/// ≈ +0.88 (heteroscedastic variance scales with magnitude, and large-magnitude rows sit far from a mid-range T).
/// So do <b>not</b> rank rows by lowest confidence / widest band to escalate a pass/fail (threshold) decision:
/// that routes confidently-far-from-line rows and misses the near-boundary rows where decision errors live.
/// The correct decision-escalation signal is whether the prediction <i>band straddles the decision threshold</i>
/// T (lower bound &lt; T &lt; upper bound), which requires the domain threshold T as input and is not derivable
/// from this scalar. Use this confidence for triaging <i>how uncertain the value is</i>; use band-straddles-T
/// for <i>which side of a decision line</i>. (Measured: honeai-sim R-7 M-17, cycle-140.)
/// </remarks>
public static class ConfidencePolicy
{
    /// <summary>
    /// Band half-width, as a multiple of <c>residual_std</c>, at which regression confidence reaches 0.
    /// A 90% conformal band is ≈1.64·residual_std for near-normal residuals, so a typical regression row
    /// lands near <c>1 − 1.64/3 ≈ 0.45</c>; narrower/wider heteroscedastic rows spread around it.
    /// </summary>
    public const double RegressionBandSigmaMultiple = 3.0;

    /// <summary>
    /// Computes the normalized confidence for a scored row, or null when no usable signal is present.
    /// </summary>
    /// <param name="row">The materialized prediction row (raw signals already populated).</param>
    /// <param name="taskType">Canonical task type (e.g. "regression", "binary-classification").</param>
    /// <param name="residualStd">The model's residual standard deviation (regression only, from metrics).</param>
    public static double? Compute(PredictionRow row, string? taskType, double? residualStd = null)
    {
        // Classification: the winning class probability.
        if (row.Probabilities is { Count: > 0 } probs)
            return Clamp(probs.Values.Max());

        // Anomaly detection: distance from the 0.5 decision boundary. Gated to anomaly-detection:
        // time-series-anomaly rows also carry AnomalyScore, but it is the detector's raw score
        // (SrCnn spectral residual / SSA raw score), not a [0,1] probability with a 0.5 boundary —
        // running it through this rule would fabricate a confidence, so TS-anomaly stays null until
        // its own signal mapping is designed and measured (P3-3).
        if (taskType is "anomaly-detection" && row.AnomalyScore is double anomaly)
            return Clamp(Math.Abs(anomaly - 0.5) * 2.0);

        // Regression with a conformal band: narrower band = higher confidence.
        if (row.ScoreUpperBound is double upper && row.ScoreLowerBound is double lower)
        {
            double half = Math.Abs(upper - lower) / 2.0;
            if (residualStd is double std && std > 0)
                return Clamp(1.0 - half / (RegressionBandSigmaMultiple * std));
            // No residual scale (older model) → normalize by the prediction magnitude as a fallback.
            if (row.Score is double s && s != 0)
                return Clamp(1.0 - Math.Min(half / Math.Abs(s), 1.0));
            return null;
        }

        // No other task family exposes a usable [0,1] confidence signal. Ranking/recommendation
        // scores in particular are NOT confidences: a predicted rating (1-5) clamps to a constant
        // 1.0 and a ranking score to an arbitrary value — a fabricated "fully confident" signal
        // for trust-loop consumers (D25). Null is the honest answer until the order-uncertainty
        // signal (top-k score margin) is designed and measured.
        return null;
    }

    private static double Clamp(double v) =>
        double.IsFinite(v) ? Math.Clamp(v, 0.0, 1.0) : 0.0;
}
