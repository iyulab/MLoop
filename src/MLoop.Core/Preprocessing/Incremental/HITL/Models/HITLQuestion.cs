using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.HITL.Models;

/// <summary>
/// Represents a question to ask the user during HITL interaction.
/// </summary>
public sealed class HITLQuestion
{
    /// <summary>
    /// Unique identifier for this question.
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// The type of question (MultipleChoice, YesNo, etc.).
    /// </summary>
    public required HITLQuestionType Type { get; init; }

    /// <summary>
    /// Context information about what was discovered in the data.
    /// </summary>
    public required string Context { get; init; }

    /// <summary>
    /// The actual question to ask the user.
    /// </summary>
    public required string Question { get; init; }

    /// <summary>
    /// Available options for the user to choose from.
    /// </summary>
    public required IReadOnlyList<HITLOption> Options { get; init; }

    /// <summary>
    /// The recommended option key (e.g., "B").
    /// </summary>
    public string? RecommendedOption { get; init; }

    /// <summary>
    /// Explanation for why this option is recommended.
    /// </summary>
    public string? RecommendationReason { get; init; }

    /// <summary>
    /// The preprocessing rule that this question is about.
    /// </summary>
    public required PreprocessingRule RelatedRule { get; init; }

    /// <summary>
    /// When this question was created.
    /// </summary>
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
}
