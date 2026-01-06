using MLoop.Core.Preprocessing.Incremental.RuleDiscovery.Models;

namespace MLoop.Core.Preprocessing.Incremental.HITL.Models;

/// <summary>
/// Complete audit log of a HITL decision.
/// </summary>
public sealed class HITLDecisionLog
{
    /// <summary>
    /// Unique identifier for this log entry.
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Session ID for this preprocessing workflow.
    /// </summary>
    public required string SessionId { get; init; }

    /// <summary>
    /// The question that was asked.
    /// </summary>
    public required HITLQuestion Question { get; init; }

    /// <summary>
    /// The user's answer.
    /// </summary>
    public required HITLAnswer Answer { get; init; }

    /// <summary>
    /// The final approved rule after applying the user's decision.
    /// </summary>
    public required PreprocessingRule ApprovedRule { get; init; }

    /// <summary>
    /// ID of the user who made this decision.
    /// </summary>
    public required string UserId { get; init; }

    /// <summary>
    /// When this decision was logged.
    /// </summary>
    public DateTime LoggedAt { get; init; } = DateTime.UtcNow;

    /// <summary>
    /// Optional additional notes about this decision.
    /// </summary>
    public string? Notes { get; init; }
}
