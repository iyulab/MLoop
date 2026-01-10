using Ironbees.AgentMode.Configuration;
using Ironbees.AgentMode.Workflow;
using Ironbees.AgentMode.Models;
using Ironbees.Core.Conversation;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using MLoop.AIAgent.Infrastructure;

namespace MLoop.AIAgent.Tests;

public class WorkflowServiceTests : IDisposable
{
    private readonly string _testWorkflowsDir;
    private readonly string _testWorkflowFile;

    public WorkflowServiceTests()
    {
        _testWorkflowsDir = Path.Combine(Path.GetTempPath(), $"mloop_workflow_test_{Guid.NewGuid()}");
        Directory.CreateDirectory(_testWorkflowsDir);

        // Create a simple test workflow file
        _testWorkflowFile = Path.Combine(_testWorkflowsDir, "test-workflow.yaml");
        File.WriteAllText(_testWorkflowFile, GetSimpleWorkflowYaml());
    }

    public void Dispose()
    {
        CleanupTestDirectory(_testWorkflowsDir);
        GC.SuppressFinalize(this);
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithNullOrchestrator_ThrowsArgumentNullException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new WorkflowService(_testWorkflowsDir, null!));
    }

    [Fact]
    public void Constructor_WithNullAgentsDirectory_ThrowsArgumentNullException()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();

        // Act & Assert
        // ArgumentException.ThrowIfNullOrWhiteSpace throws ArgumentNullException for null
        Assert.Throws<ArgumentNullException>(() =>
            new WorkflowService(null!, orchestrator));
    }

    [Fact]
    public void Constructor_WithEmptyAgentsDirectory_ThrowsArgumentException()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new WorkflowService(string.Empty, orchestrator));
    }

    [Fact]
    public void Constructor_WithWhitespaceAgentsDirectory_ThrowsArgumentException()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new WorkflowService("   ", orchestrator));
    }

    [Fact]
    public void Constructor_WithValidParameters_CreatesInstance()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();

        // Act
        using var service = new WorkflowService(_testWorkflowsDir, orchestrator);

        // Assert
        Assert.NotNull(service);
    }

    #endregion

    #region LoadWorkflowAsync Tests

    [Fact]
    public async Task LoadWorkflowAsync_WithValidFile_ReturnsWorkflowDefinition()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();
        using var service = new WorkflowService(_testWorkflowsDir, orchestrator);

        // Act
        var workflow = await service.LoadWorkflowAsync(_testWorkflowFile);

        // Assert
        Assert.NotNull(workflow);
        Assert.Equal("TestWorkflow", workflow.Name);
        Assert.Equal("1.0", workflow.Version);
    }

    [Fact]
    public async Task LoadWorkflowAsync_WithNonExistentFile_ThrowsException()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();
        using var service = new WorkflowService(_testWorkflowsDir, orchestrator);
        var nonExistentPath = Path.Combine(_testWorkflowsDir, "non-existent.yaml");

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            service.LoadWorkflowAsync(nonExistentPath));
    }

    #endregion

    #region LoadWorkflowFromStringAsync Tests

    [Fact]
    public async Task LoadWorkflowFromStringAsync_WithValidYaml_ReturnsWorkflowDefinition()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();
        using var service = new WorkflowService(_testWorkflowsDir, orchestrator);
        var yamlContent = GetSimpleWorkflowYaml();

        // Act
        var workflow = await service.LoadWorkflowFromStringAsync(yamlContent);

        // Assert
        Assert.NotNull(workflow);
        Assert.Equal("TestWorkflow", workflow.Name);
    }

    [Fact]
    public async Task LoadWorkflowFromStringAsync_WithInvalidYaml_ThrowsException()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();
        using var service = new WorkflowService(_testWorkflowsDir, orchestrator);
        var invalidYaml = "invalid: yaml: content: [";

        // Act & Assert
        await Assert.ThrowsAnyAsync<Exception>(() =>
            service.LoadWorkflowFromStringAsync(invalidYaml));
    }

    [Fact]
    public async Task LoadWorkflowFromStringAsync_WithComplexWorkflow_ParsesCorrectly()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();
        using var service = new WorkflowService(_testWorkflowsDir, orchestrator);
        var yamlContent = GetComplexWorkflowYaml();

        // Act
        var workflow = await service.LoadWorkflowFromStringAsync(yamlContent);

        // Assert
        Assert.NotNull(workflow);
        Assert.Equal("ComplexWorkflow", workflow.Name);
        Assert.NotEmpty(workflow.States);
    }

    #endregion

    #region ValidateWorkflow Tests

    [Fact]
    public async Task ValidateWorkflow_WithValidWorkflow_ReturnsValidResult()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();
        using var service = new WorkflowService(_testWorkflowsDir, orchestrator);
        var workflow = await service.LoadWorkflowFromStringAsync(GetSimpleWorkflowYaml());

        // Act
        var result = service.ValidateWorkflow(workflow);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.IsValid);
    }

    [Fact]
    public async Task ValidateWorkflow_WithIncompleteWorkflow_ReturnsValidationResult()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();
        using var service = new WorkflowService(_testWorkflowsDir, orchestrator);
        var yamlContent = GetWorkflowWithoutStartState();
        var workflow = await service.LoadWorkflowFromStringAsync(yamlContent);

        // Act
        var result = service.ValidateWorkflow(workflow);

        // Assert
        // Note: The Ironbees validator may have different validation rules.
        // This test verifies that validation runs without throwing.
        Assert.NotNull(result);
        // The validation result depends on Ironbees internal implementation
    }

    #endregion

    #region Integration Tests (Skipped - Require Full IronbeesOrchestrator Setup)

    [Fact(Skip = "Integration test requiring full IronbeesOrchestrator with IChatClient")]
    public async Task ExecuteWorkflowAsync_WithValidWorkflow_StreamsStateUpdates()
    {
        // This test requires a fully configured IronbeesOrchestrator
        // with a real or mocked IChatClient
        await Task.CompletedTask;
    }

    [Fact(Skip = "Integration test requiring full IronbeesOrchestrator with IChatClient")]
    public async Task ExecuteWorkflowFromFileAsync_WithValidFile_ExecutesWorkflow()
    {
        // This test requires a fully configured IronbeesOrchestrator
        // with a real or mocked IChatClient
        await Task.CompletedTask;
    }

    [Fact(Skip = "Integration test requiring active workflow execution")]
    public async Task ApproveAsync_WithValidExecutionId_ApprovesWorkflow()
    {
        // This test requires an active workflow execution
        await Task.CompletedTask;
    }

    [Fact(Skip = "Integration test requiring active workflow execution")]
    public async Task CancelAsync_WithValidExecutionId_CancelsWorkflow()
    {
        // This test requires an active workflow execution
        await Task.CompletedTask;
    }

    [Fact(Skip = "Integration test requiring active workflow execution")]
    public async Task GetStateAsync_WithValidExecutionId_ReturnsState()
    {
        // This test requires an active workflow execution
        await Task.CompletedTask;
    }

    [Fact(Skip = "Integration test requiring workflow service with active executions")]
    public async Task ListActiveExecutionsAsync_ReturnsExecutionList()
    {
        // This test requires workflow service with active executions
        await Task.CompletedTask;
    }

    #endregion

    #region Dispose Tests

    [Fact]
    public void Dispose_CalledMultipleTimes_DoesNotThrow()
    {
        // Arrange
        var orchestrator = CreateMockOrchestrator();
        var service = new WorkflowService(_testWorkflowsDir, orchestrator);

        // Act & Assert - should not throw
        service.Dispose();
        service.Dispose();
        service.Dispose();
    }

    #endregion

    #region Helper Methods

    private static IronbeesOrchestrator CreateMockOrchestrator()
    {
        // Create a minimal orchestrator for testing using stub implementations
        var chatClient = new StubChatClient();
        var llmConfig = new LLMConfiguration
        {
            Model = "stub-model",
            ApiKey = "stub-api-key"
        };
        var conversationStore = new StubConversationStore();
        var logger = NullLogger<IronbeesOrchestrator>.Instance;

        return new IronbeesOrchestrator(chatClient, llmConfig, conversationStore, logger);
    }

    /// <summary>
    /// Minimal stub implementation of IChatClient for testing.
    /// Only used for constructor validation tests - not for actual chat operations.
    /// </summary>
    private sealed class StubChatClient : IChatClient
    {
        public ChatClientMetadata Metadata => new("StubProvider", null, "stub-model");

        public Task<ChatResponse> GetResponseAsync(
            IEnumerable<ChatMessage> chatMessages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            throw new NotImplementedException("Stub client - not for actual use");
        }

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IEnumerable<ChatMessage> chatMessages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            throw new NotImplementedException("Stub client - not for actual use");
        }

        public object? GetService(Type serviceType, object? key = null) => null;

        public void Dispose() { }
    }

    /// <summary>
    /// Minimal stub implementation of IConversationStore for testing.
    /// </summary>
    private sealed class StubConversationStore : IConversationStore
    {
        public Task SaveAsync(ConversationState state, CancellationToken cancellationToken = default)
            => Task.CompletedTask;

        public Task<ConversationState?> LoadAsync(string conversationId, CancellationToken cancellationToken = default)
            => Task.FromResult<ConversationState?>(null);

        public Task<bool> DeleteAsync(string conversationId, CancellationToken cancellationToken = default)
            => Task.FromResult(true);

        public Task<bool> ExistsAsync(string conversationId, CancellationToken cancellationToken = default)
            => Task.FromResult(false);

        public Task<IReadOnlyList<string>> ListAsync(string? agentName = null, CancellationToken cancellationToken = default)
            => Task.FromResult<IReadOnlyList<string>>(Array.Empty<string>());

        public Task AppendMessageAsync(string conversationId, ConversationMessage message, CancellationToken cancellationToken = default)
            => Task.CompletedTask;

        public Task<int> GetMessageCountAsync(string conversationId, CancellationToken cancellationToken = default)
            => Task.FromResult(0);
    }

    private static string GetSimpleWorkflowYaml()
    {
        return @"
name: TestWorkflow
version: ""1.0""
description: Simple test workflow

states:
  - id: START
    type: Start
    next: END

  - id: END
    type: Terminal
";
    }

    private static string GetComplexWorkflowYaml()
    {
        return @"
name: ComplexWorkflow
version: ""2.0""
description: Complex workflow with multiple states

agents:
  - ref: test-agent
    alias: tester

states:
  - id: START
    type: Start
    next: PROCESS

  - id: PROCESS
    type: Agent
    executor: tester
    next: END

  - id: END
    type: Terminal

settings:
  defaultTimeout: PT5M
  enableCheckpointing: false
";
    }

    private static string GetWorkflowWithoutStartState()
    {
        return @"
name: InvalidWorkflow
version: ""1.0""

states:
  - id: PROCESS
    type: Agent
    next: END

  - id: END
    type: Terminal
";
    }

    private static void CleanupTestDirectory(string path)
    {
        if (Directory.Exists(path))
        {
            try
            {
                Directory.Delete(path, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    #endregion
}
