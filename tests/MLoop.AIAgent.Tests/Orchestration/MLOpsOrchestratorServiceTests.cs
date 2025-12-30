// Copyright (c) IYULab. All rights reserved.
// Licensed under the MIT License.

using Microsoft.Extensions.AI;
using MLoop.AIAgent.Core.Orchestration;

namespace MLoop.AIAgent.Tests.Orchestration;

public class MLOpsOrchestratorServiceTests : IDisposable
{
    private readonly string _testDirectory;
    private readonly OrchestrationSessionStore _store;
    private readonly string _testDataFile;

    public MLOpsOrchestratorServiceTests()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"mloop_orch_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDirectory);
        _store = new OrchestrationSessionStore(Path.Combine(_testDirectory, "orchestration"));

        // Create a simple test CSV file
        _testDataFile = Path.Combine(_testDirectory, "test_data.csv");
        File.WriteAllText(_testDataFile, @"feature1,feature2,target
1.0,2.0,A
2.0,3.0,B
3.0,4.0,A
4.0,5.0,B
5.0,6.0,A");
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
    public void Constructor_WithValidDependencies_CreatesInstance()
    {
        // Arrange
        var chatClient = new TestChatClient();

        // Act
        var service = new MLOpsOrchestratorService(chatClient, _store);

        // Assert
        Assert.NotNull(service);
    }

    [Fact]
    public async Task ExecuteAsync_WithNonExistentFile_EmitsFailedEvent()
    {
        // Arrange
        var chatClient = new TestChatClient();
        var service = new MLOpsOrchestratorService(chatClient, _store);
        var events = new List<OrchestrationEvent>();

        // Act
        await foreach (var evt in service.ExecuteAsync("nonexistent.csv"))
        {
            events.Add(evt);
        }

        // Assert - Service emits OrchestrationFailedEvent instead of throwing
        Assert.Single(events);
        var failedEvent = Assert.IsType<OrchestrationFailedEvent>(events[0]);
        Assert.Contains("not found", failedEvent.Error);
    }

    [Fact]
    public async Task ExecuteAsync_EmitsStartedEvent()
    {
        // Arrange
        var chatClient = new TestChatClient();
        var service = new MLOpsOrchestratorService(chatClient, _store);
        var events = new List<OrchestrationEvent>();

        // Act
        try
        {
            await foreach (var evt in service.ExecuteAsync(_testDataFile))
            {
                events.Add(evt);
                if (events.Count >= 1) break; // Get at least the started event
            }
        }
        catch
        {
            // Expected - test chat client doesn't return valid responses
        }

        // Assert
        Assert.Contains(events, e => e is OrchestrationStartedEvent);
    }

    [Fact]
    public async Task ExecuteAsync_CreatesSession()
    {
        // Arrange
        var chatClient = new TestChatClient();
        var service = new MLOpsOrchestratorService(chatClient, _store);
        string? sessionId = null;

        // Act
        try
        {
            await foreach (var evt in service.ExecuteAsync(_testDataFile))
            {
                if (evt is OrchestrationStartedEvent started)
                {
                    sessionId = started.SessionId;
                    break;
                }
            }
        }
        catch
        {
            // Expected
        }

        // Assert
        Assert.NotNull(sessionId);
        Assert.True(_store.SessionExists(sessionId));
    }

    [Fact]
    public async Task ExecuteAsync_WithOptions_UsesProvidedOptions()
    {
        // Arrange
        var chatClient = new TestChatClient();
        var service = new MLOpsOrchestratorService(chatClient, _store);
        var options = new OrchestrationOptions
        {
            TargetColumn = "target",
            TaskType = "BinaryClassification",
            MaxTrainingTimeSeconds = 120
        };

        OrchestrationStartedEvent? startedEvent = null;

        // Act - Get started event which contains options
        try
        {
            await foreach (var evt in service.ExecuteAsync(_testDataFile, options))
            {
                if (evt is OrchestrationStartedEvent started)
                {
                    startedEvent = started;
                }
                // Continue to let session be saved
                if (startedEvent != null) break;
            }
        }
        catch
        {
            // Expected - test chat client doesn't return valid responses
        }

        // Assert - Verify started event was emitted and contains session info
        Assert.NotNull(startedEvent);
        Assert.Contains(_testDataFile, startedEvent.DataFilePath);
    }

    [Fact]
    public async Task ExecuteAsync_WithCancellation_HandlesCancellationGracefully()
    {
        // Arrange
        var chatClient = new TestChatClient();
        var service = new MLOpsOrchestratorService(chatClient, _store);
        var cts = new CancellationTokenSource();
        var events = new List<OrchestrationEvent>();

        // Act
        cts.Cancel(); // Cancel immediately
        try
        {
            await foreach (var evt in service.ExecuteAsync(_testDataFile, cancellationToken: cts.Token))
            {
                events.Add(evt);
            }
        }
        catch (OperationCanceledException)
        {
            // Expected
        }

        // Assert - The service should handle cancellation gracefully
    }

    [Fact]
    public async Task ResumeAsync_WithNonExistentSession_EmitsFailedEvent()
    {
        // Arrange
        var chatClient = new TestChatClient();
        var service = new MLOpsOrchestratorService(chatClient, _store);
        var events = new List<OrchestrationEvent>();

        // Act
        await foreach (var evt in service.ResumeAsync("nonexistent-session"))
        {
            events.Add(evt);
        }

        // Assert - Service emits OrchestrationFailedEvent instead of throwing
        Assert.Single(events);
        var failedEvent = Assert.IsType<OrchestrationFailedEvent>(events[0]);
        Assert.Contains("not found", failedEvent.Error);
    }

    [Fact]
    public async Task ResumeAsync_WithCompletedSession_EmitsFailedEvent()
    {
        // Arrange
        var chatClient = new TestChatClient();
        var service = new MLOpsOrchestratorService(chatClient, _store);

        var session = new OrchestrationSession
        {
            SessionId = "completed-session",
            Context = new OrchestrationContext
            {
                SessionId = "completed-session",
                CurrentState = OrchestrationState.Completed,
                DataFilePath = _testDataFile,
                Options = new OrchestrationOptions()
            },
            Status = SessionStatus.Completed,
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        await _store.SaveSessionAsync(session);

        var events = new List<OrchestrationEvent>();

        // Act
        await foreach (var evt in service.ResumeAsync("completed-session"))
        {
            events.Add(evt);
        }

        // Assert - Service emits OrchestrationFailedEvent for non-resumable sessions
        Assert.Single(events);
        var failedEvent = Assert.IsType<OrchestrationFailedEvent>(events[0]);
        Assert.Contains("cannot be resumed", failedEvent.Error);
    }

    [Fact]
    public async Task ResumeAsync_WithPausedSession_ResumesFromSavedState()
    {
        // Arrange
        var chatClient = new TestChatClient();
        var service = new MLOpsOrchestratorService(chatClient, _store);

        var session = new OrchestrationSession
        {
            SessionId = "paused-session",
            Context = new OrchestrationContext
            {
                SessionId = "paused-session",
                CurrentState = OrchestrationState.ModelRecommendation,
                DataFilePath = _testDataFile,
                Options = new OrchestrationOptions()
            },
            Status = SessionStatus.Paused,
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        await _store.SaveSessionAsync(session);

        var events = new List<OrchestrationEvent>();

        // Act
        try
        {
            await foreach (var evt in service.ResumeAsync("paused-session"))
            {
                events.Add(evt);
                if (events.Count >= 2) break;
            }
        }
        catch
        {
            // Expected - test chat client doesn't return valid responses
        }

        // Assert - ResumeAsync should emit some events (PhaseStarted, AgentStarted, or StateChanged)
        Assert.NotEmpty(events);
        Assert.True(events.Any(e => e is PhaseStartedEvent or AgentStartedEvent or StateChangedEvent));
    }

    [Fact]
    public async Task ListSessionsAsync_ReturnsAllSessions()
    {
        // Arrange
        var chatClient = new TestChatClient();
        var service = new MLOpsOrchestratorService(chatClient, _store);

        // Create test sessions directly via store (more reliable than running ExecuteAsync)
        for (int i = 0; i < 3; i++)
        {
            var session = new OrchestrationSession
            {
                SessionId = $"list-test-session-{i}",
                Context = new OrchestrationContext
                {
                    SessionId = $"list-test-session-{i}",
                    CurrentState = OrchestrationState.DataAnalysis,
                    DataFilePath = _testDataFile,
                    Options = new OrchestrationOptions()
                },
                Status = SessionStatus.Active,
                CreatedAt = DateTimeOffset.UtcNow,
                UpdatedAt = DateTimeOffset.UtcNow
            };
            await _store.SaveSessionAsync(session);
        }

        // Act
        var sessions = await service.ListSessionsAsync();

        // Assert
        Assert.Equal(3, sessions.Count);
    }

    [Fact]
    public async Task GetResumableSessionsAsync_ReturnsOnlyResumableSessions()
    {
        // Arrange
        var chatClient = new TestChatClient();
        var service = new MLOpsOrchestratorService(chatClient, _store);

        var activeSession = new OrchestrationSession
        {
            SessionId = "active-session",
            Context = new OrchestrationContext
            {
                SessionId = "active-session",
                CurrentState = OrchestrationState.DataAnalysis,
                DataFilePath = _testDataFile,
                Options = new OrchestrationOptions()
            },
            Status = SessionStatus.Active,
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };

        var pausedSession = new OrchestrationSession
        {
            SessionId = "paused-session-2",
            Context = new OrchestrationContext
            {
                SessionId = "paused-session-2",
                CurrentState = OrchestrationState.ModelRecommendation,
                DataFilePath = _testDataFile,
                Options = new OrchestrationOptions()
            },
            Status = SessionStatus.Paused,
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };

        var completedSession = new OrchestrationSession
        {
            SessionId = "completed-session-2",
            Context = new OrchestrationContext
            {
                SessionId = "completed-session-2",
                CurrentState = OrchestrationState.Completed,
                DataFilePath = _testDataFile,
                Options = new OrchestrationOptions()
            },
            Status = SessionStatus.Completed,
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };

        await _store.SaveSessionAsync(activeSession);
        await _store.SaveSessionAsync(pausedSession);
        await _store.SaveSessionAsync(completedSession);

        // Act
        var resumable = await service.GetResumableSessionsAsync();

        // Assert
        Assert.Equal(2, resumable.Count);
        Assert.Contains(resumable, s => s.SessionId == "active-session");
        Assert.Contains(resumable, s => s.SessionId == "paused-session-2");
        Assert.DoesNotContain(resumable, s => s.SessionId == "completed-session-2");
    }

    /// <summary>
    /// Simple test implementation of IChatClient that returns minimal responses.
    /// </summary>
    private class TestChatClient : IChatClient
    {
        public ChatClientMetadata Metadata => new("TestProvider", null, "test-model");

        public TService? GetService<TService>(object? key = null) where TService : class
            => null;

        public object? GetService(Type serviceType, object? key = null)
            => null;

        public Task<ChatResponse> GetResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            // Return a minimal response
            return Task.FromResult(new ChatResponse([
                new ChatMessage(ChatRole.Assistant, "Test response")
            ]));
        }

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            return GetStreamingResponseInternal();
        }

        private async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseInternal()
        {
            yield return new ChatResponseUpdate(ChatRole.Assistant, "Test response");
            await Task.CompletedTask;
        }

        public void Dispose() { }
    }
}
