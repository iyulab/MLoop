using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Contains the results of a completed workflow stage.
/// </summary>
public sealed class StageResult
{
    /// <summary>
    /// The workflow stage that was completed.
    /// </summary>
    public required WorkflowStage Stage { get; init; }

    /// <summary>
    /// Number of records sampled in this stage.
    /// </summary>
    public required int SampleSize { get; init; }

    /// <summary>
    /// Sample ratio used (0.0 to 1.0).
    /// </summary>
    public required double SampleRatio { get; init; }

    /// <summary>
    /// Analysis results from the sample.
    /// </summary>
    public required SampleAnalysis Analysis { get; init; }

    /// <summary>
    /// Preprocessing rules discovered in this stage.
    /// </summary>
    public required List<PreprocessingRule> RulesDiscovered { get; init; }

    /// <summary>
    /// How long this stage took to complete.
    /// </summary>
    public required TimeSpan Duration { get; init; }

    /// <summary>
    /// When this stage was completed.
    /// </summary>
    public DateTime CompletedAt { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// Optional notes or observations from this stage.
    /// </summary>
    public string? Notes { get; init; }
}
