namespace MLoop.Core.AutoML;

/// <summary>
/// Estimates optimal training time based on dataset characteristics.
/// Two-phase approach: static heuristic followed by reactive adjustment from probe results.
/// </summary>
public static class TimeEstimator
{
    public const int MinSeconds = 30;
    public const int MaxSeconds = 1800;
    private const int BaseSeconds = 30;

    /// <summary>
    /// Phase 1: Static estimate based on data dimensions and task type.
    /// Formula: base(30) x row_factor x col_factor x task_factor, clamped to [30, 1800].
    /// </summary>
    public static int EstimateStatic(
        int rows,
        int columns,
        string task,
        int classCount,
        bool hasTextFeatures = false)
    {
        // Row scaling: log-linear
        double rowFactor = Math.Max(1.0, Math.Log10(Math.Max(rows, 1) / 1000.0) + 1.0);

        // Column scaling: sub-linear
        double colFactor = Math.Max(1.0, Math.Sqrt(Math.Max(columns, 1) / 10.0));

        // Task multiplier
        double taskFactor = task.ToLowerInvariant() switch
        {
            "regression" => 1.1,
            "multiclass-classification" => 1.5 * Math.Max(1.0, Math.Log2(Math.Max(classCount, 2) / 4.0)),
            _ => 1.0 // binary-classification
        };

        // Text feature boost
        if (hasTextFeatures)
            colFactor *= 2.0;

        double estimated = BaseSeconds * rowFactor * colFactor * taskFactor;

        return Math.Clamp((int)Math.Round(estimated), MinSeconds, MaxSeconds);
    }

    /// <summary>
    /// Calculate probe time as a fraction of static estimate.
    /// Returns a value between 10 and 30 seconds.
    /// </summary>
    public static int GetProbeTime(int staticEstimate)
    {
        return Math.Max(10, Math.Min(30, staticEstimate / 3));
    }

    /// <summary>
    /// Phase 2: Reactive adjustment based on probe training results.
    /// High metric (>0.95) reduces time, low metric (&lt;0.5) doubles it.
    /// </summary>
    public static int EstimateReactive(ProbeResult probe, int staticEstimate)
    {
        double multiplier;

        if (probe.BestMetric > 0.95)
            multiplier = (double)probe.ProbeTimeSeconds * 2 / staticEstimate; // fast convergence
        else if (probe.BestMetric > 0.85)
            multiplier = 1.0; // use static estimate as-is
        else if (probe.BestMetric >= 0.5)
            multiplier = 1.5; // needs more time
        else
            multiplier = 2.0; // difficult dataset

        int reactive = (int)Math.Round(staticEstimate * multiplier);
        return Math.Clamp(reactive, MinSeconds, MaxSeconds);
    }
}

/// <summary>
/// Result from a quick probe training run
/// </summary>
public class ProbeResult
{
    public double BestMetric { get; init; }
    public int ProbeTimeSeconds { get; init; }
    public int TrialsCompleted { get; init; }
}
