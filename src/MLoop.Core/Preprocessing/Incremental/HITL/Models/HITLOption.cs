namespace MLoop.Core.Preprocessing.Incremental.HITL.Models;

/// <summary>
/// Represents a single option in a HITL question.
/// </summary>
public sealed class HITLOption
{
    /// <summary>
    /// Option key (e.g., "A", "B", "C", "D").
    /// </summary>
    public required string Key { get; init; }

    /// <summary>
    /// Short display label for the option.
    /// </summary>
    public required string Label { get; init; }

    /// <summary>
    /// Detailed description explaining the option.
    /// </summary>
    public required string Description { get; init; }

    /// <summary>
    /// The action that will be taken if this option is selected.
    /// </summary>
    public ActionType Action { get; init; }

    /// <summary>
    /// Indicates if this is the recommended option.
    /// </summary>
    public bool IsRecommended { get; init; }
}
