// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using System.Text.Json.Serialization;

namespace MLoop.AIAgent.Core.Orchestration;

/// <summary>
/// Represents a persistent orchestration session that can be saved and resumed.
/// </summary>
public class OrchestrationSession
{
    /// <summary>Session schema version for migration.</summary>
    public const int CurrentSchemaVersion = 1;

    /// <summary>Schema version of this session.</summary>
    public int SchemaVersion { get; set; } = CurrentSchemaVersion;

    /// <summary>Unique session identifier.</summary>
    public required string SessionId { get; set; }

    /// <summary>Session creation timestamp.</summary>
    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;

    /// <summary>Last update timestamp.</summary>
    public DateTimeOffset UpdatedAt { get; set; } = DateTimeOffset.UtcNow;

    /// <summary>Session status.</summary>
    public SessionStatus Status { get; set; } = SessionStatus.Active;

    /// <summary>Current orchestration context.</summary>
    public required OrchestrationContext Context { get; set; }

    /// <summary>State history for debugging and audit.</summary>
    public List<StateTransition> StateHistory { get; set; } = [];

    /// <summary>Checkpoints for recovery.</summary>
    public List<SessionCheckpoint> Checkpoints { get; set; } = [];

    /// <summary>Session metadata.</summary>
    public Dictionary<string, string> Metadata { get; set; } = [];

    /// <summary>
    /// Records a state transition.
    /// </summary>
    public void RecordStateTransition(OrchestrationState fromState, OrchestrationState toState, string? reason = null)
    {
        StateHistory.Add(new StateTransition
        {
            FromState = fromState,
            ToState = toState,
            Reason = reason,
            Timestamp = DateTimeOffset.UtcNow
        });
        UpdatedAt = DateTimeOffset.UtcNow;
    }

    /// <summary>
    /// Creates a checkpoint at the current state.
    /// </summary>
    public SessionCheckpoint CreateCheckpoint(string? label = null)
    {
        var checkpoint = new SessionCheckpoint
        {
            CheckpointId = $"cp-{DateTimeOffset.UtcNow:yyyyMMddHHmmss}",
            State = Context.CurrentState,
            Label = label ?? Context.CurrentState.GetDisplayName(),
            Timestamp = DateTimeOffset.UtcNow,
            ContextSnapshot = System.Text.Json.JsonSerializer.Serialize(Context)
        };
        Checkpoints.Add(checkpoint);
        UpdatedAt = DateTimeOffset.UtcNow;
        return checkpoint;
    }

    /// <summary>
    /// Restores context from a checkpoint.
    /// </summary>
    public bool RestoreFromCheckpoint(string checkpointId)
    {
        var checkpoint = Checkpoints.FirstOrDefault(c => c.CheckpointId == checkpointId);
        if (checkpoint?.ContextSnapshot == null) return false;

        var restoredContext = System.Text.Json.JsonSerializer.Deserialize<OrchestrationContext>(checkpoint.ContextSnapshot);
        if (restoredContext == null) return false;

        Context = restoredContext;
        UpdatedAt = DateTimeOffset.UtcNow;
        return true;
    }

    /// <summary>
    /// Gets the duration of the session.
    /// </summary>
    public TimeSpan GetDuration() => UpdatedAt - CreatedAt;

    /// <summary>
    /// Checks if session can be resumed.
    /// </summary>
    public bool CanResume() =>
        Status == SessionStatus.Paused ||
        (Status == SessionStatus.Active && !Context.CurrentState.IsTerminal());

    /// <summary>
    /// Marks session as completed.
    /// </summary>
    public void MarkCompleted()
    {
        Status = SessionStatus.Completed;
        Context.CurrentState = OrchestrationState.Completed;
        Context.CompletedAt = DateTimeOffset.UtcNow;
        UpdatedAt = DateTimeOffset.UtcNow;
    }

    /// <summary>
    /// Marks session as failed.
    /// </summary>
    public void MarkFailed(string error)
    {
        Status = SessionStatus.Failed;
        Context.CurrentState = OrchestrationState.Failed;
        Context.LastError = error;
        Context.CompletedAt = DateTimeOffset.UtcNow;
        UpdatedAt = DateTimeOffset.UtcNow;
    }

    /// <summary>
    /// Marks session as cancelled.
    /// </summary>
    public void MarkCancelled(string? reason = null)
    {
        Status = SessionStatus.Cancelled;
        Context.CurrentState = OrchestrationState.Cancelled;
        Context.CompletedAt = DateTimeOffset.UtcNow;
        UpdatedAt = DateTimeOffset.UtcNow;
        if (reason != null)
        {
            Metadata["cancellation_reason"] = reason;
        }
    }

    /// <summary>
    /// Marks session as paused (waiting for HITL).
    /// </summary>
    public void MarkPaused()
    {
        Status = SessionStatus.Paused;
        UpdatedAt = DateTimeOffset.UtcNow;
    }

    /// <summary>
    /// Resumes a paused session.
    /// </summary>
    public void Resume()
    {
        if (Status == SessionStatus.Paused)
        {
            Status = SessionStatus.Active;
            UpdatedAt = DateTimeOffset.UtcNow;
        }
    }
}

/// <summary>
/// Session status.
/// </summary>
public enum SessionStatus
{
    /// <summary>Session is actively running.</summary>
    Active,

    /// <summary>Session is paused (e.g., waiting for HITL).</summary>
    Paused,

    /// <summary>Session completed successfully.</summary>
    Completed,

    /// <summary>Session failed with error.</summary>
    Failed,

    /// <summary>Session was cancelled.</summary>
    Cancelled
}

/// <summary>
/// Records a state transition for audit trail.
/// </summary>
public record StateTransition
{
    /// <summary>State before transition.</summary>
    public OrchestrationState FromState { get; init; }

    /// <summary>State after transition.</summary>
    public OrchestrationState ToState { get; init; }

    /// <summary>Reason for transition.</summary>
    public string? Reason { get; init; }

    /// <summary>When transition occurred.</summary>
    public DateTimeOffset Timestamp { get; init; }
}

/// <summary>
/// Checkpoint for session recovery.
/// </summary>
public record SessionCheckpoint
{
    /// <summary>Unique checkpoint ID.</summary>
    public required string CheckpointId { get; init; }

    /// <summary>State at checkpoint.</summary>
    public OrchestrationState State { get; init; }

    /// <summary>Human-readable label.</summary>
    public required string Label { get; init; }

    /// <summary>When checkpoint was created.</summary>
    public DateTimeOffset Timestamp { get; init; }

    /// <summary>Serialized context snapshot.</summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ContextSnapshot { get; init; }
}

/// <summary>
/// Summary of a session for listing.
/// </summary>
public record SessionSummary
{
    /// <summary>Session ID.</summary>
    public required string SessionId { get; init; }

    /// <summary>Input data file path.</summary>
    public required string DataFilePath { get; init; }

    /// <summary>Current state.</summary>
    public OrchestrationState State { get; init; }

    /// <summary>Session status.</summary>
    public SessionStatus Status { get; init; }

    /// <summary>When created.</summary>
    public DateTimeOffset CreatedAt { get; init; }

    /// <summary>When last updated.</summary>
    public DateTimeOffset UpdatedAt { get; init; }

    /// <summary>Progress percentage.</summary>
    public int ProgressPercentage { get; init; }

    /// <summary>Whether session can be resumed.</summary>
    public bool CanResume { get; init; }

    /// <summary>Last error if failed.</summary>
    public string? LastError { get; init; }

    /// <summary>
    /// Creates a summary from a session.
    /// </summary>
    public static SessionSummary FromSession(OrchestrationSession session)
    {
        return new SessionSummary
        {
            SessionId = session.SessionId,
            DataFilePath = session.Context.DataFilePath,
            State = session.Context.CurrentState,
            Status = session.Status,
            CreatedAt = session.CreatedAt,
            UpdatedAt = session.UpdatedAt,
            ProgressPercentage = session.Context.CurrentState.GetProgressPercentage(),
            CanResume = session.CanResume(),
            LastError = session.Context.LastError
        };
    }
}
