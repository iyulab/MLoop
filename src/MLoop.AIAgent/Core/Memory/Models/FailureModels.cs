namespace MLoop.AIAgent.Core.Memory.Models;

/// <summary>
/// Context of a failure occurrence for pattern learning.
/// </summary>
public sealed class FailureContext
{
    /// <summary>
    /// Type of error (e.g., "InvalidDataException", "ColumnMismatchException").
    /// </summary>
    public string ErrorType { get; set; } = default!;

    /// <summary>
    /// Error message text.
    /// </summary>
    public string ErrorMessage { get; set; } = default!;

    /// <summary>
    /// Stack trace if available.
    /// </summary>
    public string? StackTrace { get; set; }

    /// <summary>
    /// Phase where failure occurred (DataLoading, Preprocessing, Training, Evaluation).
    /// </summary>
    public string Phase { get; set; } = default!;

    /// <summary>
    /// Dataset characteristics when failure occurred.
    /// </summary>
    public DatasetFingerprint? DatasetContext { get; set; }

    /// <summary>
    /// Preprocessing state when failure occurred.
    /// </summary>
    public List<PreprocessingStep>? PreprocessingState { get; set; }

    /// <summary>
    /// Environment information.
    /// </summary>
    public Dictionary<string, string>? Environment { get; set; }

    /// <summary>
    /// Timestamp of failure.
    /// </summary>
    public DateTime OccurredAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Generates a query string for semantic search.
    /// </summary>
    public string ToQueryString()
    {
        var parts = new List<string>
        {
            $"Error: {ErrorType}",
            $"Message: {ErrorMessage}",
            $"Phase: {Phase}"
        };

        if (DatasetContext != null)
            parts.Add($"Dataset: {DatasetContext.Describe()}");

        return string.Join(". ", parts);
    }

    /// <summary>
    /// Generates a description for semantic embedding.
    /// </summary>
    public string Describe()
    {
        var desc = $"Failure in {Phase} phase: {ErrorType}. {ErrorMessage}";

        if (DatasetContext != null)
            desc += $" Dataset: {DatasetContext.SizeCategory} with {DatasetContext.ColumnNames.Count} columns.";

        if (PreprocessingState?.Count > 0)
            desc += $" After {PreprocessingState.Count} preprocessing steps.";

        return desc;
    }
}

/// <summary>
/// Resolution applied to fix a failure.
/// </summary>
public sealed class Resolution
{
    /// <summary>
    /// Root cause identified.
    /// </summary>
    public string RootCause { get; set; } = default!;

    /// <summary>
    /// Fix description.
    /// </summary>
    public string FixDescription { get; set; } = default!;

    /// <summary>
    /// Code changes or configuration changes made.
    /// </summary>
    public List<string>? Changes { get; set; }

    /// <summary>
    /// Advice to prevent similar failures.
    /// </summary>
    public string? PreventionAdvice { get; set; }

    /// <summary>
    /// Whether the fix was verified as successful.
    /// </summary>
    public bool Verified { get; set; }

    /// <summary>
    /// Timestamp of resolution.
    /// </summary>
    public DateTime ResolvedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Generates a description for semantic embedding.
    /// </summary>
    public string Describe()
    {
        var desc = $"Root cause: {RootCause}. Fix: {FixDescription}.";

        if (!string.IsNullOrEmpty(PreventionAdvice))
            desc += $" Prevention: {PreventionAdvice}";

        return desc;
    }
}

/// <summary>
/// Warning based on similar past failures.
/// </summary>
public sealed class FailureWarning
{
    /// <summary>
    /// Original failure context.
    /// </summary>
    public FailureContext? Context { get; set; }

    /// <summary>
    /// Resolution that fixed the original failure.
    /// </summary>
    public Resolution? Resolution { get; set; }

    /// <summary>
    /// Similarity score to current situation (0.0 to 1.0).
    /// </summary>
    public float SimilarityScore { get; set; }

    /// <summary>
    /// Risk level based on similarity and severity.
    /// </summary>
    public WarningLevel Level => SimilarityScore switch
    {
        >= 0.9f => WarningLevel.High,
        >= 0.7f => WarningLevel.Medium,
        _ => WarningLevel.Low
    };

    /// <summary>
    /// Warning message to display.
    /// </summary>
    public string Message => $"Potential failure risk ({SimilarityScore * 100:F0}% match): {Context?.ErrorType}. " +
                            $"Suggestion: {Resolution?.PreventionAdvice ?? Resolution?.FixDescription}";
}

/// <summary>
/// Warning severity level.
/// </summary>
public enum WarningLevel
{
    Low,
    Medium,
    High
}

/// <summary>
/// Information about a dataset for failure checking.
/// </summary>
public sealed class DatasetInfo
{
    /// <summary>
    /// Dataset fingerprint.
    /// </summary>
    public DatasetFingerprint? Fingerprint { get; set; }

    /// <summary>
    /// Current processing phase.
    /// </summary>
    public string? CurrentPhase { get; set; }

    /// <summary>
    /// Preprocessing steps already applied.
    /// </summary>
    public List<PreprocessingStep>? AppliedSteps { get; set; }

    /// <summary>
    /// Generates a query string for semantic search.
    /// </summary>
    public string ToQueryString()
    {
        var parts = new List<string>();

        if (Fingerprint != null)
            parts.Add(Fingerprint.Describe());

        if (!string.IsNullOrEmpty(CurrentPhase))
            parts.Add($"Phase: {CurrentPhase}");

        if (AppliedSteps?.Count > 0)
            parts.Add($"Steps: {string.Join(", ", AppliedSteps.Select(s => s.Type))}");

        return string.Join(". ", parts);
    }
}
