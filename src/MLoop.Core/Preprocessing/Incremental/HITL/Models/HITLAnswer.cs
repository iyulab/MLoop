namespace MLoop.Core.Preprocessing.Incremental.HITL.Models;

/// <summary>
/// Represents a user's answer to a HITL question.
/// </summary>
public sealed class HITLAnswer
{
    /// <summary>
    /// ID of the question this answers.
    /// </summary>
    public required string QuestionId { get; init; }

    /// <summary>
    /// The selected option key (e.g., "A", "B", "Yes", "No").
    /// </summary>
    public required string SelectedOption { get; init; }

    /// <summary>
    /// Custom value provided by user (for TextInput or NumericInput).
    /// </summary>
    public string? CustomValue { get; init; }

    /// <summary>
    /// Optional rationale provided by the user for their decision.
    /// </summary>
    public string? UserRationale { get; init; }

    /// <summary>
    /// When the user answered this question.
    /// </summary>
    public DateTime AnsweredAt { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// How long it took the user to make this decision.
    /// </summary>
    public TimeSpan TimeToDecide { get; init; }
}
