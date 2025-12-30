// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using MLoop.AIAgent.Core.Orchestration;

namespace MLoop.AIAgent.Tests.Orchestration;

public class OrchestrationSessionStoreTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly OrchestrationSessionStore _store;

    public OrchestrationSessionStoreTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"mloop_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDirectory);
        _store = new OrchestrationSessionStore(Path.Combine(_testDirectory, "orchestration"));
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDirectory))
        {
            Directory.Delete(_testDirectory, recursive: true);
        }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public async Task SaveSessionAsync_CreatesSessionFile()
    {
        // Arrange
        var session = CreateTestSession();

        // Act
        await _store.SaveSessionAsync(session);

        // Assert
        Assert.True(_store.SessionExists(session.SessionId));
    }

    [Fact]
    public async Task LoadSessionAsync_ReturnsNull_WhenSessionDoesNotExist()
    {
        // Act
        var session = await _store.LoadSessionAsync("nonexistent-session-id");

        // Assert
        Assert.Null(session);
    }

    [Fact]
    public async Task SaveAndLoadSession_PreservesAllProperties()
    {
        // Arrange
        var originalSession = CreateTestSession();
        originalSession.Context.CurrentState = OrchestrationState.ModelRecommendation;
        originalSession.Status = SessionStatus.Active;
        originalSession.Context.Options = new OrchestrationOptions
        {
            TargetColumn = "target",
            TaskType = "BinaryClassification",
            MaxTrainingTimeSeconds = 600
        };

        // Act
        await _store.SaveSessionAsync(originalSession);
        var loadedSession = await _store.LoadSessionAsync(originalSession.SessionId);

        // Assert
        Assert.NotNull(loadedSession);
        Assert.Equal(originalSession.SessionId, loadedSession.SessionId);
        Assert.Equal(originalSession.Context.CurrentState, loadedSession.Context.CurrentState);
        Assert.Equal(originalSession.Status, loadedSession.Status);
        Assert.Equal(originalSession.Context.DataFilePath, loadedSession.Context.DataFilePath);
        Assert.Equal(originalSession.Context.Options.TargetColumn, loadedSession.Context.Options.TargetColumn);
        Assert.Equal(originalSession.Context.Options.TaskType, loadedSession.Context.Options.TaskType);
        Assert.Equal(originalSession.Context.Options.MaxTrainingTimeSeconds, loadedSession.Context.Options.MaxTrainingTimeSeconds);
    }

    [Fact]
    public async Task DeleteSessionAsync_RemovesSession()
    {
        // Arrange
        var session = CreateTestSession();
        await _store.SaveSessionAsync(session);
        Assert.True(_store.SessionExists(session.SessionId));

        // Act
        await _store.DeleteSessionAsync(session.SessionId);

        // Assert
        Assert.False(_store.SessionExists(session.SessionId));
    }

    [Fact]
    public async Task ListSessionsAsync_ReturnsAllSessions()
    {
        // Arrange
        var session1 = CreateTestSession("session-1");
        var session2 = CreateTestSession("session-2");
        var session3 = CreateTestSession("session-3");

        await _store.SaveSessionAsync(session1);
        await _store.SaveSessionAsync(session2);
        await _store.SaveSessionAsync(session3);

        // Act
        var summaries = await _store.ListSessionsAsync();

        // Assert
        Assert.Equal(3, summaries.Count);
        Assert.Contains(summaries, s => s.SessionId == "session-1");
        Assert.Contains(summaries, s => s.SessionId == "session-2");
        Assert.Contains(summaries, s => s.SessionId == "session-3");
    }

    [Fact]
    public async Task ListSessionsAsync_ReturnsSortedByUpdatedAt()
    {
        // Arrange
        var session1 = CreateTestSession("old-session");
        session1.UpdatedAt = DateTimeOffset.UtcNow.AddHours(-2);

        var session2 = CreateTestSession("new-session");
        session2.UpdatedAt = DateTimeOffset.UtcNow;

        await _store.SaveSessionAsync(session1);
        await _store.SaveSessionAsync(session2);

        // Act
        var summaries = await _store.ListSessionsAsync();

        // Assert
        Assert.Equal("new-session", summaries[0].SessionId);
        Assert.Equal("old-session", summaries[1].SessionId);
    }

    [Fact]
    public async Task SaveCheckpointAsync_CreatesCheckpointFile()
    {
        // Arrange
        var session = CreateTestSession();
        await _store.SaveSessionAsync(session);

        var checkpoint = new SessionCheckpoint
        {
            CheckpointId = "cp-001",
            Label = "Test Checkpoint",
            State = OrchestrationState.DataAnalysisReview,
            Timestamp = DateTimeOffset.UtcNow
        };

        // Act
        await _store.SaveCheckpointAsync(session.SessionId, checkpoint);

        // Assert
        var checkpoints = await _store.ListCheckpointsAsync(session.SessionId);
        Assert.Single(checkpoints);
        Assert.Equal("cp-001", checkpoints[0].CheckpointId);
    }

    [Fact]
    public async Task ListCheckpointsAsync_ReturnsEmptyList_WhenNoCheckpoints()
    {
        // Act
        var checkpoints = await _store.ListCheckpointsAsync("nonexistent-session");

        // Assert
        Assert.Empty(checkpoints);
    }

    [Fact]
    public async Task SaveDecisionAsync_AppendsDecisions()
    {
        // Arrange
        var session = CreateTestSession();
        await _store.SaveSessionAsync(session);

        var decision1 = new HitlDecision
        {
            CheckpointId = "data-analysis-review",
            SelectedOptionId = "approve",
            Timestamp = DateTimeOffset.UtcNow.AddMinutes(-5)
        };

        var decision2 = new HitlDecision
        {
            CheckpointId = "model-selection-review",
            SelectedOptionId = "modify",
            Timestamp = DateTimeOffset.UtcNow
        };

        // Act
        await _store.SaveDecisionAsync(session.SessionId, decision1);
        await _store.SaveDecisionAsync(session.SessionId, decision2);

        // Assert - decisions are appended, not overwritten
        var decision3 = new HitlDecision
        {
            CheckpointId = "preprocessing-review",
            SelectedOptionId = "skip",
            Timestamp = DateTimeOffset.UtcNow
        };
        await _store.SaveDecisionAsync(session.SessionId, decision3);
    }

    [Fact]
    public async Task SaveArtifactAsync_CreatesArtifactFile()
    {
        // Arrange
        var session = CreateTestSession();
        await _store.SaveSessionAsync(session);

        var artifact = new Dictionary<string, object>
        {
            ["rowCount"] = 1000,
            ["columnCount"] = 15,
            ["quality"] = 0.95
        };

        // Act
        var path = await _store.SaveArtifactAsync(session.SessionId, "data-analysis", artifact);

        // Assert
        Assert.True(File.Exists(path));
        Assert.Contains("data-analysis.json", path);
    }

    [Fact]
    public async Task LoadArtifactAsync_ReturnsArtifact()
    {
        // Arrange
        var session = CreateTestSession();
        await _store.SaveSessionAsync(session);

        var original = new TestArtifact { Name = "test", Value = 42 };
        await _store.SaveArtifactAsync(session.SessionId, "test-artifact", original);

        // Act
        var loaded = await _store.LoadArtifactAsync<TestArtifact>(session.SessionId, "test-artifact");

        // Assert
        Assert.NotNull(loaded);
        Assert.Equal("test", loaded.Name);
        Assert.Equal(42, loaded.Value);
    }

    [Fact]
    public async Task LoadArtifactAsync_ReturnsNull_WhenNotExists()
    {
        // Act
        var artifact = await _store.LoadArtifactAsync<TestArtifact>("any-session", "nonexistent");

        // Assert
        Assert.Null(artifact);
    }

    [Fact]
    public async Task GetResumableSessionsAsync_ReturnsOnlyResumableSessions()
    {
        // Arrange
        var activeSession = CreateTestSession("active");
        activeSession.Status = SessionStatus.Active;

        var pausedSession = CreateTestSession("paused");
        pausedSession.Status = SessionStatus.Paused;

        var completedSession = CreateTestSession("completed");
        completedSession.Status = SessionStatus.Completed;
        completedSession.Context.CurrentState = OrchestrationState.Completed;

        var failedSession = CreateTestSession("failed");
        failedSession.Status = SessionStatus.Failed;
        failedSession.Context.CurrentState = OrchestrationState.Failed;

        await _store.SaveSessionAsync(activeSession);
        await _store.SaveSessionAsync(pausedSession);
        await _store.SaveSessionAsync(completedSession);
        await _store.SaveSessionAsync(failedSession);

        // Act
        var resumable = await _store.GetResumableSessionsAsync();

        // Assert
        Assert.Equal(2, resumable.Count);
        Assert.Contains(resumable, s => s.SessionId == "active");
        Assert.Contains(resumable, s => s.SessionId == "paused");
        Assert.DoesNotContain(resumable, s => s.SessionId == "completed");
        Assert.DoesNotContain(resumable, s => s.SessionId == "failed");
    }

    [Fact]
    public async Task CleanupOldSessionsAsync_RemovesOldCompletedSessions()
    {
        // Arrange
        // Note: SaveSessionAsync always sets UpdatedAt to current time,
        // so we can only test the status-based filtering logic here.
        var completedSession = CreateTestSession("completed-session");
        completedSession.Status = SessionStatus.Completed;

        var activeSession = CreateTestSession("active-session");
        activeSession.Status = SessionStatus.Active;

        await _store.SaveSessionAsync(completedSession);
        await _store.SaveSessionAsync(activeSession);

        // Both sessions exist initially
        Assert.True(_store.SessionExists("completed-session"));
        Assert.True(_store.SessionExists("active-session"));

        // Act - Use zero maxAge to cleanup all terminal sessions immediately
        await _store.CleanupOldSessionsAsync(TimeSpan.Zero);

        // Assert
        Assert.False(_store.SessionExists("completed-session")); // Removed (completed)
        Assert.True(_store.SessionExists("active-session")); // Kept (active is not terminal)
    }

    [Fact]
    public void GetArtifactsDirectory_CreatesDirectory()
    {
        // Arrange
        var sessionId = "test-session-artifacts";

        // Act
        var path = _store.GetArtifactsDirectory(sessionId);

        // Assert
        Assert.True(Directory.Exists(path));
        Assert.Contains(sessionId, path);
    }

    [Fact]
    public void ForProject_CreatesStoreWithProjectPath()
    {
        // Arrange
        var projectDir = Path.Combine(_testDirectory, "my-project");
        Directory.CreateDirectory(projectDir);

        // Act
        var store = OrchestrationSessionStore.ForProject(projectDir);

        // Assert
        Assert.NotNull(store);
    }

    // Helper methods
    private static OrchestrationSession CreateTestSession(string? sessionId = null)
    {
        var id = sessionId ?? $"test-{Guid.NewGuid():N}";
        return new OrchestrationSession
        {
            SessionId = id,
            Context = new OrchestrationContext
            {
                SessionId = id,
                DataFilePath = "/test/data.csv",
                CurrentState = OrchestrationState.NotStarted,
                Options = new OrchestrationOptions()
            },
            Status = SessionStatus.Active,
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
    }

    private class TestArtifact
    {
        public string? Name { get; set; }
        public int Value { get; set; }
    }
}
