// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace MLoop.AIAgent.Core.Orchestration;

/// <summary>
/// Manages persistence of orchestration sessions to the file system.
/// Sessions are stored in .mloop/orchestration/ directory.
/// </summary>
public class OrchestrationSessionStore
{
    private readonly string _baseDirectory;
    private readonly ILogger<OrchestrationSessionStore>? _logger;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
    };

    /// <summary>
    /// Directory structure under .mloop/orchestration/
    /// </summary>
    private string SessionsDirectory => Path.Combine(_baseDirectory, "sessions");
    private string CheckpointsDirectory => Path.Combine(_baseDirectory, "checkpoints");
    private string ArtifactsDirectory => Path.Combine(_baseDirectory, "artifacts");
    private string DecisionsDirectory => Path.Combine(_baseDirectory, "decisions");

    /// <summary>
    /// Creates a new session store.
    /// </summary>
    /// <param name="baseDirectory">Base directory for orchestration files. Defaults to .mloop/orchestration in current directory.</param>
    /// <param name="logger">Optional logger.</param>
    public OrchestrationSessionStore(string? baseDirectory = null, ILogger<OrchestrationSessionStore>? logger = null)
    {
        _baseDirectory = baseDirectory ?? Path.Combine(Environment.CurrentDirectory, ".mloop", "orchestration");
        _logger = logger;
        EnsureDirectories();
    }

    /// <summary>
    /// Creates a new session store for a specific project directory.
    /// </summary>
    public static OrchestrationSessionStore ForProject(string projectDirectory, ILogger<OrchestrationSessionStore>? logger = null)
    {
        var baseDir = Path.Combine(projectDirectory, ".mloop", "orchestration");
        return new OrchestrationSessionStore(baseDir, logger);
    }

    private void EnsureDirectories()
    {
        Directory.CreateDirectory(SessionsDirectory);
        Directory.CreateDirectory(CheckpointsDirectory);
        Directory.CreateDirectory(ArtifactsDirectory);
        Directory.CreateDirectory(DecisionsDirectory);
    }

    /// <summary>
    /// Saves a session to disk.
    /// </summary>
    public async Task SaveSessionAsync(OrchestrationSession session, CancellationToken cancellationToken = default)
    {
        session.UpdatedAt = DateTimeOffset.UtcNow;
        var filePath = GetSessionFilePath(session.SessionId);

        var json = JsonSerializer.Serialize(session, JsonOptions);
        await File.WriteAllTextAsync(filePath, json, cancellationToken);

        _logger?.LogDebug("Saved session {SessionId} to {FilePath}", session.SessionId, filePath);
    }

    /// <summary>
    /// Loads a session from disk.
    /// </summary>
    public async Task<OrchestrationSession?> LoadSessionAsync(string sessionId, CancellationToken cancellationToken = default)
    {
        var filePath = GetSessionFilePath(sessionId);
        if (!File.Exists(filePath))
        {
            _logger?.LogDebug("Session file not found: {FilePath}", filePath);
            return null;
        }

        var json = await File.ReadAllTextAsync(filePath, cancellationToken);
        var session = JsonSerializer.Deserialize<OrchestrationSession>(json, JsonOptions);

        _logger?.LogDebug("Loaded session {SessionId} from {FilePath}", sessionId, filePath);
        return session;
    }

    /// <summary>
    /// Deletes a session from disk.
    /// </summary>
    public Task DeleteSessionAsync(string sessionId, CancellationToken cancellationToken = default)
    {
        var filePath = GetSessionFilePath(sessionId);
        if (File.Exists(filePath))
        {
            File.Delete(filePath);
            _logger?.LogDebug("Deleted session {SessionId}", sessionId);
        }

        // Also delete associated checkpoints
        var checkpointDir = Path.Combine(CheckpointsDirectory, sessionId);
        if (Directory.Exists(checkpointDir))
        {
            Directory.Delete(checkpointDir, recursive: true);
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Lists all sessions.
    /// </summary>
    public async Task<List<SessionSummary>> ListSessionsAsync(CancellationToken cancellationToken = default)
    {
        var summaries = new List<SessionSummary>();

        if (!Directory.Exists(SessionsDirectory))
            return summaries;

        foreach (var file in Directory.EnumerateFiles(SessionsDirectory, "*.json"))
        {
            try
            {
                var json = await File.ReadAllTextAsync(file, cancellationToken);
                var session = JsonSerializer.Deserialize<OrchestrationSession>(json, JsonOptions);
                if (session != null)
                {
                    summaries.Add(SessionSummary.FromSession(session));
                }
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "Failed to load session from {FilePath}", file);
            }
        }

        return summaries.OrderByDescending(s => s.UpdatedAt).ToList();
    }

    /// <summary>
    /// Saves a checkpoint for a session.
    /// </summary>
    public async Task SaveCheckpointAsync(string sessionId, SessionCheckpoint checkpoint, CancellationToken cancellationToken = default)
    {
        var checkpointDir = Path.Combine(CheckpointsDirectory, sessionId);
        Directory.CreateDirectory(checkpointDir);

        var filePath = Path.Combine(checkpointDir, $"{checkpoint.CheckpointId}.json");
        var json = JsonSerializer.Serialize(checkpoint, JsonOptions);
        await File.WriteAllTextAsync(filePath, json, cancellationToken);

        _logger?.LogDebug("Saved checkpoint {CheckpointId} for session {SessionId}", checkpoint.CheckpointId, sessionId);
    }

    /// <summary>
    /// Lists checkpoints for a session.
    /// </summary>
    public async Task<List<SessionCheckpoint>> ListCheckpointsAsync(string sessionId, CancellationToken cancellationToken = default)
    {
        var checkpoints = new List<SessionCheckpoint>();
        var checkpointDir = Path.Combine(CheckpointsDirectory, sessionId);

        if (!Directory.Exists(checkpointDir))
            return checkpoints;

        foreach (var file in Directory.EnumerateFiles(checkpointDir, "*.json"))
        {
            try
            {
                var json = await File.ReadAllTextAsync(file, cancellationToken);
                var checkpoint = JsonSerializer.Deserialize<SessionCheckpoint>(json, JsonOptions);
                if (checkpoint != null)
                {
                    checkpoints.Add(checkpoint);
                }
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "Failed to load checkpoint from {FilePath}", file);
            }
        }

        return checkpoints.OrderByDescending(c => c.Timestamp).ToList();
    }

    /// <summary>
    /// Saves a HITL decision.
    /// </summary>
    public async Task SaveDecisionAsync(string sessionId, HitlDecision decision, CancellationToken cancellationToken = default)
    {
        var decisionsFile = Path.Combine(DecisionsDirectory, $"{sessionId}-decisions.json");

        List<HitlDecision> decisions = [];
        if (File.Exists(decisionsFile))
        {
            var existingJson = await File.ReadAllTextAsync(decisionsFile, cancellationToken);
            decisions = JsonSerializer.Deserialize<List<HitlDecision>>(existingJson, JsonOptions) ?? [];
        }

        decisions.Add(decision);
        var json = JsonSerializer.Serialize(decisions, JsonOptions);
        await File.WriteAllTextAsync(decisionsFile, json, cancellationToken);

        _logger?.LogDebug("Saved decision for checkpoint {CheckpointId} in session {SessionId}", decision.CheckpointId, sessionId);
    }

    /// <summary>
    /// Saves an artifact.
    /// </summary>
    public async Task<string> SaveArtifactAsync(string sessionId, string artifactName, object content, CancellationToken cancellationToken = default)
    {
        var artifactDir = Path.Combine(ArtifactsDirectory, sessionId);
        Directory.CreateDirectory(artifactDir);

        var filePath = Path.Combine(artifactDir, $"{artifactName}.json");
        var json = JsonSerializer.Serialize(content, JsonOptions);
        await File.WriteAllTextAsync(filePath, json, cancellationToken);

        _logger?.LogDebug("Saved artifact {ArtifactName} for session {SessionId}", artifactName, sessionId);
        return filePath;
    }

    /// <summary>
    /// Loads an artifact.
    /// </summary>
    public async Task<T?> LoadArtifactAsync<T>(string sessionId, string artifactName, CancellationToken cancellationToken = default)
    {
        var filePath = Path.Combine(ArtifactsDirectory, sessionId, $"{artifactName}.json");
        if (!File.Exists(filePath))
            return default;

        var json = await File.ReadAllTextAsync(filePath, cancellationToken);
        return JsonSerializer.Deserialize<T>(json, JsonOptions);
    }

    /// <summary>
    /// Gets the file path for a session.
    /// </summary>
    private string GetSessionFilePath(string sessionId)
        => Path.Combine(SessionsDirectory, $"{sessionId}.json");

    /// <summary>
    /// Checks if a session exists.
    /// </summary>
    public bool SessionExists(string sessionId)
        => File.Exists(GetSessionFilePath(sessionId));

    /// <summary>
    /// Gets resumable sessions (paused or active non-terminal).
    /// </summary>
    public async Task<List<SessionSummary>> GetResumableSessionsAsync(CancellationToken cancellationToken = default)
    {
        var all = await ListSessionsAsync(cancellationToken);
        return all.Where(s => s.CanResume).ToList();
    }

    /// <summary>
    /// Cleans up old completed sessions.
    /// </summary>
    public async Task CleanupOldSessionsAsync(TimeSpan maxAge, CancellationToken cancellationToken = default)
    {
        var cutoff = DateTimeOffset.UtcNow - maxAge;
        var sessions = await ListSessionsAsync(cancellationToken);

        foreach (var session in sessions.Where(s =>
            s.Status is SessionStatus.Completed or SessionStatus.Cancelled or SessionStatus.Failed &&
            s.UpdatedAt < cutoff))
        {
            await DeleteSessionAsync(session.SessionId, cancellationToken);
            _logger?.LogInformation("Cleaned up old session {SessionId}", session.SessionId);
        }
    }

    /// <summary>
    /// Gets the artifacts directory for a session.
    /// </summary>
    public string GetArtifactsDirectory(string sessionId)
    {
        var dir = Path.Combine(ArtifactsDirectory, sessionId);
        Directory.CreateDirectory(dir);
        return dir;
    }
}
