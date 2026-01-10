namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Represents a single stage in the progressive sampling process.
/// </summary>
public sealed class SamplingStage
{
    /// <summary>
    /// Gets the stage number (1-based).
    /// </summary>
    public required int StageNumber { get; init; }

    /// <summary>
    /// Gets the sample ratio for this stage (0.0 to 1.0).
    /// </summary>
    public required double SampleRatio { get; init; }

    /// <summary>
    /// Gets the actual number of rows sampled.
    /// </summary>
    public required long RowCount { get; init; }

    /// <summary>
    /// Gets the analysis performed on this stage's sample.
    /// </summary>
    public SampleAnalysis? Analysis { get; init; }

    /// <summary>
    /// Gets the stage execution status.
    /// </summary>
    public required StageStatus Status { get; init; }

    /// <summary>
    /// Gets the timestamp when this stage started.
    /// </summary>
    public required DateTime StartTime { get; init; }

    /// <summary>
    /// Gets the timestamp when this stage completed.
    /// </summary>
    public DateTime? EndTime { get; init; }

    /// <summary>
    /// Gets the duration of this stage.
    /// </summary>
    public TimeSpan? Duration => EndTime.HasValue ? EndTime.Value - StartTime : null;

    /// <summary>
    /// Gets any error that occurred during this stage.
    /// </summary>
    public string? Error { get; init; }

    /// <summary>
    /// Gets whether convergence was detected at this stage.
    /// </summary>
    /// <remarks>
    /// If true, subsequent stages may be skipped (early stopping).
    /// </remarks>
    public bool ConvergenceDetected { get; init; }

    /// <summary>
    /// Gets the variance from the previous stage (if applicable).
    /// </summary>
    /// <remarks>
    /// Used for convergence detection. Lower variance indicates stability.
    /// </remarks>
    public double? VarianceFromPrevious { get; init; }

    /// <summary>
    /// Gets stage-specific metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; init; } = new();

    /// <summary>
    /// Gets a summary string for logging.
    /// </summary>
    /// <returns>Human-readable summary.</returns>
    public string GetSummary()
    {
        var durationStr = Duration.HasValue ? $"{Duration.Value.TotalSeconds:F2}s" : "in progress";
        var convergenceStr = ConvergenceDetected ? " (converged)" : "";

        return $"Stage {StageNumber}: {SampleRatio:P} ({RowCount:N0} rows) - " +
               $"{Status} in {durationStr}{convergenceStr}";
    }
}

/// <summary>
/// Status of a sampling stage.
/// </summary>
public enum StageStatus
{
    /// <summary>
    /// Stage is pending execution.
    /// </summary>
    Pending,

    /// <summary>
    /// Stage is currently executing.
    /// </summary>
    Running,

    /// <summary>
    /// Stage completed successfully.
    /// </summary>
    Completed,

    /// <summary>
    /// Stage was skipped (e.g., due to early stopping).
    /// </summary>
    Skipped,

    /// <summary>
    /// Stage failed with an error.
    /// </summary>
    Failed
}
