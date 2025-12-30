// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using System.Text.Json.Serialization;

namespace MLoop.AIAgent.Core.Orchestration;

/// <summary>
/// Base class for all orchestration events.
/// Events are streamed to the caller to provide real-time updates.
/// </summary>
[JsonDerivedType(typeof(OrchestrationStartedEvent), "started")]
[JsonDerivedType(typeof(StateChangedEvent), "state_changed")]
[JsonDerivedType(typeof(PhaseStartedEvent), "phase_started")]
[JsonDerivedType(typeof(PhaseCompletedEvent), "phase_completed")]
[JsonDerivedType(typeof(HitlRequestedEvent), "hitl_requested")]
[JsonDerivedType(typeof(HitlResponseReceivedEvent), "hitl_response")]
[JsonDerivedType(typeof(AgentStartedEvent), "agent_started")]
[JsonDerivedType(typeof(AgentCompletedEvent), "agent_completed")]
[JsonDerivedType(typeof(ProgressUpdateEvent), "progress")]
[JsonDerivedType(typeof(OrchestrationCompletedEvent), "completed")]
[JsonDerivedType(typeof(OrchestrationFailedEvent), "failed")]
[JsonDerivedType(typeof(OrchestrationCancelledEvent), "cancelled")]
public abstract record OrchestrationEvent
{
    /// <summary>Unique event ID.</summary>
    public string EventId { get; init; } = Guid.NewGuid().ToString("N")[..8];

    /// <summary>Session ID this event belongs to.</summary>
    public required string SessionId { get; init; }

    /// <summary>Timestamp when the event occurred.</summary>
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>Event type discriminator for serialization.</summary>
    [JsonPropertyName("$type")]
    public abstract string EventType { get; }
}

/// <summary>
/// Event emitted when orchestration starts.
/// </summary>
public sealed record OrchestrationStartedEvent : OrchestrationEvent
{
    public override string EventType => "started";

    /// <summary>Path to the input data file.</summary>
    public required string DataFilePath { get; init; }

    /// <summary>Orchestration options used.</summary>
    public OrchestrationOptions? Options { get; init; }
}

/// <summary>
/// Event emitted when state changes.
/// </summary>
public sealed record StateChangedEvent : OrchestrationEvent
{
    public override string EventType => "state_changed";

    /// <summary>Previous state.</summary>
    public OrchestrationState FromState { get; init; }

    /// <summary>New state.</summary>
    public OrchestrationState ToState { get; init; }

    /// <summary>Reason for the state change.</summary>
    public string? Reason { get; init; }
}

/// <summary>
/// Event emitted when a major phase starts.
/// </summary>
public sealed record PhaseStartedEvent : OrchestrationEvent
{
    public override string EventType => "phase_started";

    /// <summary>Phase number (1-5).</summary>
    public int PhaseNumber { get; init; }

    /// <summary>Phase name.</summary>
    public required string PhaseName { get; init; }

    /// <summary>Description of what this phase does.</summary>
    public string? Description { get; init; }
}

/// <summary>
/// Event emitted when a major phase completes.
/// </summary>
public sealed record PhaseCompletedEvent : OrchestrationEvent
{
    public override string EventType => "phase_completed";

    /// <summary>Phase number (1-5).</summary>
    public int PhaseNumber { get; init; }

    /// <summary>Phase name.</summary>
    public required string PhaseName { get; init; }

    /// <summary>Duration of the phase.</summary>
    public TimeSpan Duration { get; init; }

    /// <summary>Summary of phase results.</summary>
    public Dictionary<string, object>? Summary { get; init; }
}

/// <summary>
/// Event emitted when HITL interaction is required.
/// </summary>
public sealed record HitlRequestedEvent : OrchestrationEvent
{
    public override string EventType => "hitl_requested";

    /// <summary>Checkpoint identifier.</summary>
    public required string CheckpointId { get; init; }

    /// <summary>Checkpoint name for display.</summary>
    public required string CheckpointName { get; init; }

    /// <summary>Question or prompt for the user.</summary>
    public required string Question { get; init; }

    /// <summary>Available options for the user to choose from.</summary>
    public required List<HitlOption> Options { get; init; }

    /// <summary>Context information to help user make decision.</summary>
    public Dictionary<string, object>? Context { get; init; }

    /// <summary>Current confidence level (0-1).</summary>
    public double? Confidence { get; init; }

    /// <summary>Whether auto-approval is possible.</summary>
    public bool CanAutoApprove { get; init; }

    /// <summary>Timeout for response.</summary>
    public TimeSpan? Timeout { get; init; }
}

/// <summary>
/// Event emitted when HITL response is received.
/// </summary>
public sealed record HitlResponseReceivedEvent : OrchestrationEvent
{
    public override string EventType => "hitl_response";

    /// <summary>Checkpoint identifier.</summary>
    public required string CheckpointId { get; init; }

    /// <summary>Selected option ID.</summary>
    public required string SelectedOptionId { get; init; }

    /// <summary>Whether this was an auto-approval.</summary>
    public bool IsAutoApproval { get; init; }

    /// <summary>User's comment if any.</summary>
    public string? UserComment { get; init; }

    /// <summary>Response time.</summary>
    public TimeSpan ResponseTime { get; init; }
}

/// <summary>
/// Event emitted when an agent starts processing.
/// </summary>
public sealed record AgentStartedEvent : OrchestrationEvent
{
    public override string EventType => "agent_started";

    /// <summary>Agent name.</summary>
    public required string AgentName { get; init; }

    /// <summary>Agent type.</summary>
    public required string AgentType { get; init; }

    /// <summary>Task description.</summary>
    public string? TaskDescription { get; init; }
}

/// <summary>
/// Event emitted when an agent completes processing.
/// </summary>
public sealed record AgentCompletedEvent : OrchestrationEvent
{
    public override string EventType => "agent_completed";

    /// <summary>Agent name.</summary>
    public required string AgentName { get; init; }

    /// <summary>Agent type.</summary>
    public required string AgentType { get; init; }

    /// <summary>Whether agent completed successfully.</summary>
    public bool Success { get; init; }

    /// <summary>Duration of agent execution.</summary>
    public TimeSpan Duration { get; init; }

    /// <summary>Summary of agent results.</summary>
    public Dictionary<string, object>? Results { get; init; }

    /// <summary>Error message if failed.</summary>
    public string? Error { get; init; }
}

/// <summary>
/// Event emitted for progress updates.
/// </summary>
public sealed record ProgressUpdateEvent : OrchestrationEvent
{
    public override string EventType => "progress";

    /// <summary>Progress percentage (0-100).</summary>
    public int Percentage { get; init; }

    /// <summary>Current operation description.</summary>
    public required string CurrentOperation { get; init; }

    /// <summary>Optional details.</summary>
    public string? Details { get; init; }

    /// <summary>Items processed so far.</summary>
    public int? ProcessedItems { get; init; }

    /// <summary>Total items to process.</summary>
    public int? TotalItems { get; init; }
}

/// <summary>
/// Event emitted when orchestration completes successfully.
/// </summary>
public sealed record OrchestrationCompletedEvent : OrchestrationEvent
{
    public override string EventType => "completed";

    /// <summary>Total duration of orchestration.</summary>
    public TimeSpan TotalDuration { get; init; }

    /// <summary>Final model metrics.</summary>
    public ModelMetrics? FinalMetrics { get; init; }

    /// <summary>Output artifacts.</summary>
    public Dictionary<string, string>? Artifacts { get; init; }

    /// <summary>Summary of all phases.</summary>
    public Dictionary<string, object>? Summary { get; init; }
}

/// <summary>
/// Event emitted when orchestration fails.
/// </summary>
public sealed record OrchestrationFailedEvent : OrchestrationEvent
{
    public override string EventType => "failed";

    /// <summary>Error message.</summary>
    public required string Error { get; init; }

    /// <summary>Detailed error information.</summary>
    public string? Details { get; init; }

    /// <summary>State when failure occurred.</summary>
    public OrchestrationState FailedAtState { get; init; }

    /// <summary>Whether the session can be resumed.</summary>
    public bool CanResume { get; init; }
}

/// <summary>
/// Event emitted when orchestration is cancelled.
/// </summary>
public sealed record OrchestrationCancelledEvent : OrchestrationEvent
{
    public override string EventType => "cancelled";

    /// <summary>Reason for cancellation.</summary>
    public string? Reason { get; init; }

    /// <summary>State when cancelled.</summary>
    public OrchestrationState CancelledAtState { get; init; }

    /// <summary>Whether the session can be resumed.</summary>
    public bool CanResume { get; init; }
}

/// <summary>
/// HITL option for user selection.
/// </summary>
public record HitlOption
{
    /// <summary>Option identifier.</summary>
    public required string Id { get; init; }

    /// <summary>Display label.</summary>
    public required string Label { get; init; }

    /// <summary>Description of what happens when selected.</summary>
    public string? Description { get; init; }

    /// <summary>Whether this is the default/recommended option.</summary>
    public bool IsDefault { get; init; }

    /// <summary>Keyboard shortcut (e.g., "Y", "N").</summary>
    public string? Shortcut { get; init; }
}

/// <summary>
/// Model metrics summary.
/// </summary>
public record ModelMetrics
{
    /// <summary>Model name/algorithm.</summary>
    public required string ModelName { get; init; }

    /// <summary>Primary metric name.</summary>
    public required string PrimaryMetricName { get; init; }

    /// <summary>Primary metric value.</summary>
    public double PrimaryMetricValue { get; init; }

    /// <summary>All metrics.</summary>
    public Dictionary<string, double>? AllMetrics { get; init; }

    /// <summary>Cross-validation scores if available.</summary>
    public double[]? CrossValidationScores { get; init; }
}

/// <summary>
/// Options for orchestration execution.
/// </summary>
public record OrchestrationOptions
{
    /// <summary>Target column name (if known).</summary>
    public string? TargetColumn { get; init; }

    /// <summary>ML task type (if known).</summary>
    public string? TaskType { get; init; }

    /// <summary>Maximum training time in seconds.</summary>
    public int? MaxTrainingTimeSeconds { get; init; }

    /// <summary>Seed for reproducibility.</summary>
    public int? Seed { get; init; }

    /// <summary>Skip HITL checkpoints (fully automated).</summary>
    public bool SkipHitl { get; init; }

    /// <summary>Auto-approve high-confidence decisions.</summary>
    public bool AutoApproveHighConfidence { get; init; } = true;

    /// <summary>Confidence threshold for auto-approval (0-1).</summary>
    public double AutoApprovalThreshold { get; init; } = 0.85;

    /// <summary>Output directory for artifacts.</summary>
    public string? OutputDirectory { get; init; }

    /// <summary>Whether to deploy after training.</summary>
    public bool AutoDeploy { get; init; }

    /// <summary>Custom ironbees AgenticSettings.</summary>
    public Ironbees.Core.Goals.AgenticSettings? AgenticSettings { get; init; }
}
