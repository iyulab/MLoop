using MLoop.AIAgent.Infrastructure;
using Microsoft.Extensions.AI;

namespace MLoop.AIAgent.Tests;

public class ConversationServiceTests : IDisposable
{
    private readonly string _testConversationsDir;
    private readonly ConversationService _service;

    public ConversationServiceTests()
    {
        _testConversationsDir = Path.Combine(Path.GetTempPath(), $"mloop_conv_test_{Guid.NewGuid()}");
        Directory.CreateDirectory(_testConversationsDir);
        _service = new ConversationService(_testConversationsDir);
    }

    public void Dispose()
    {
        _service.Dispose();
        CleanupTestDirectory(_testConversationsDir);
        GC.SuppressFinalize(this);
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidDirectory_CreatesInstance()
    {
        // Arrange & Act
        using var service = new ConversationService(_testConversationsDir);

        // Assert
        Assert.NotNull(service);
    }

    [Fact]
    public void Constructor_WithNullDirectory_ThrowsArgumentException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentException>(() => new ConversationService((string)null!));
    }

    [Fact]
    public void Constructor_WithEmptyDirectory_ThrowsArgumentException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentException>(() => new ConversationService(string.Empty));
    }

    #endregion

    #region Conversation Management Tests

    [Fact]
    public async Task StartOrResumeAsync_NewConversation_InitializesEmptyHistory()
    {
        // Arrange
        var conversationId = $"test_{Guid.NewGuid():N}";

        // Act
        await _service.StartOrResumeAsync(conversationId);

        // Assert
        Assert.Equal(conversationId, _service.CurrentConversationId);
        Assert.Empty(_service.CurrentHistory);
    }

    [Fact]
    public async Task StartOrResumeAsync_WithAgentName_SetsAgentContext()
    {
        // Arrange
        var conversationId = $"test_{Guid.NewGuid():N}";
        var agentName = "data-analyst";

        // Act
        await _service.StartOrResumeAsync(conversationId, agentName);

        // Assert
        Assert.Equal(conversationId, _service.CurrentConversationId);
        Assert.Contains(agentName, _service.GetContextSummary());
    }

    [Fact]
    public async Task StartOrResumeAsync_ExistingConversation_LoadsHistory()
    {
        // Arrange
        var conversationId = $"test_{Guid.NewGuid():N}";

        // First session - create conversation with messages
        await _service.StartOrResumeAsync(conversationId);
        _service.AddUserMessage("Hello");
        _service.AddAgentResponse("Hi there!");
        await _service.SaveCurrentAsync();

        // Create new service to simulate new session
        using var newService = new ConversationService(_testConversationsDir);

        // Act - Resume existing conversation
        await newService.StartOrResumeAsync(conversationId);

        // Assert
        Assert.Equal(2, newService.CurrentHistory.Count);
        Assert.Equal("Hello", newService.CurrentHistory[0].Text);
        Assert.Equal("Hi there!", newService.CurrentHistory[1].Text);
    }

    [Fact]
    public async Task StartOrResumeAsync_NullConversationId_ThrowsArgumentException()
    {
        // Arrange & Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(
            () => _service.StartOrResumeAsync(null!));
    }

    #endregion

    #region Message Management Tests

    [Fact]
    public void AddUserMessage_ValidMessage_AddsToHistory()
    {
        // Arrange
        var message = "Test user message";

        // Act
        _service.AddUserMessage(message);

        // Assert
        Assert.Single(_service.CurrentHistory);
        Assert.Equal(ChatRole.User, _service.CurrentHistory[0].Role);
        Assert.Equal(message, _service.CurrentHistory[0].Text);
    }

    [Fact]
    public void AddUserMessage_NullOrWhitespace_DoesNotAdd()
    {
        // Act
        _service.AddUserMessage(null!);
        _service.AddUserMessage("");
        _service.AddUserMessage("   ");

        // Assert
        Assert.Empty(_service.CurrentHistory);
    }

    [Fact]
    public void AddAgentResponse_ValidResponse_AddsToHistory()
    {
        // Arrange
        var response = "Test agent response";

        // Act
        _service.AddAgentResponse(response);

        // Assert
        Assert.Single(_service.CurrentHistory);
        Assert.Equal(ChatRole.Assistant, _service.CurrentHistory[0].Role);
        Assert.Equal(response, _service.CurrentHistory[0].Text);
    }

    [Fact]
    public void AddAgentResponse_NullOrWhitespace_DoesNotAdd()
    {
        // Act
        _service.AddAgentResponse(null!);
        _service.AddAgentResponse("");
        _service.AddAgentResponse("   ");

        // Assert
        Assert.Empty(_service.CurrentHistory);
    }

    [Fact]
    public void AddSystemMessage_ValidMessage_AddsToHistory()
    {
        // Arrange
        var message = "System instruction";

        // Act
        _service.AddSystemMessage(message);

        // Assert
        Assert.Single(_service.CurrentHistory);
        Assert.Equal(ChatRole.System, _service.CurrentHistory[0].Role);
        Assert.Equal(message, _service.CurrentHistory[0].Text);
    }

    [Fact]
    public void ClearHistory_WithMessages_ClearsAllMessages()
    {
        // Arrange
        _service.AddUserMessage("Message 1");
        _service.AddAgentResponse("Response 1");
        _service.AddUserMessage("Message 2");

        // Act
        _service.ClearHistory();

        // Assert
        Assert.Empty(_service.CurrentHistory);
    }

    #endregion

    #region Persistence Tests

    [Fact]
    public async Task SaveCurrentAsync_WithConversationId_PersistsMessages()
    {
        // Arrange
        var conversationId = $"test_{Guid.NewGuid():N}";
        await _service.StartOrResumeAsync(conversationId);
        _service.AddUserMessage("Test message");
        _service.AddAgentResponse("Test response");

        // Act
        await _service.SaveCurrentAsync();

        // Assert - Verify by loading in new service
        using var verifyService = new ConversationService(_testConversationsDir);
        await verifyService.StartOrResumeAsync(conversationId);
        Assert.Equal(2, verifyService.CurrentHistory.Count);
    }

    [Fact]
    public async Task SaveCurrentAsync_WithoutConversationId_DoesNotThrow()
    {
        // Arrange - No conversation started
        _service.AddUserMessage("Orphan message");

        // Act & Assert - Should not throw
        await _service.SaveCurrentAsync();
    }

    #endregion

    #region Context Management Tests

    [Fact]
    public void SetProjectContext_ValidPath_SetsContext()
    {
        // Arrange
        var projectPath = "/test/project/path";

        // Act
        _service.SetProjectContext(projectPath);

        // Assert
        Assert.Equal(projectPath, _service.CurrentProjectPath);
        Assert.Contains(projectPath, _service.GetContextSummary());
    }

    [Fact]
    public void SetDataFileContext_ValidPath_SetsContext()
    {
        // Arrange
        var dataFile = "/test/data.csv";

        // Act
        _service.SetDataFileContext(dataFile);

        // Assert
        Assert.Equal(dataFile, _service.CurrentDataFile);
        Assert.Contains(dataFile, _service.GetContextSummary());
    }

    [Fact]
    public async Task GetContextSummary_WithFullContext_ReturnsCompleteSummary()
    {
        // Arrange
        var conversationId = $"test_{Guid.NewGuid():N}";
        await _service.StartOrResumeAsync(conversationId, "mlops-manager");
        _service.SetProjectContext("/project/path");
        _service.SetDataFileContext("/data/train.csv");
        _service.AddUserMessage("Analyze my dataset");

        // Act
        var summary = _service.GetContextSummary();

        // Assert
        Assert.Contains(conversationId, summary);
        Assert.Contains("mlops-manager", summary);
        Assert.Contains("/project/path", summary);
        Assert.Contains("/data/train.csv", summary);
        Assert.Contains("Messages: 1", summary);
        Assert.Contains("Analyze my dataset", summary);
    }

    #endregion

    #region Store Operations Tests

    [Fact]
    public async Task ListConversationsAsync_EmptyStore_ReturnsEmptyList()
    {
        // Act
        var conversations = await _service.ListConversationsAsync();

        // Assert
        Assert.Empty(conversations);
    }

    [Fact]
    public async Task ListConversationsAsync_WithConversations_ReturnsList()
    {
        // Arrange
        var conv1 = $"test1_{Guid.NewGuid():N}";
        var conv2 = $"test2_{Guid.NewGuid():N}";

        await _service.StartOrResumeAsync(conv1);
        _service.AddUserMessage("Message 1");
        await _service.SaveCurrentAsync();

        await _service.StartOrResumeAsync(conv2);
        _service.AddUserMessage("Message 2");
        await _service.SaveCurrentAsync();

        // Act
        var conversations = await _service.ListConversationsAsync();

        // Assert
        Assert.Equal(2, conversations.Count);
        Assert.Contains(conv1, conversations);
        Assert.Contains(conv2, conversations);
    }

    [Fact]
    public async Task DeleteConversationAsync_ExistingConversation_ReturnsTrue()
    {
        // Arrange
        var conversationId = $"test_{Guid.NewGuid():N}";
        await _service.StartOrResumeAsync(conversationId);
        _service.AddUserMessage("Test");
        await _service.SaveCurrentAsync();

        // Create new service to test deletion
        using var deleteService = new ConversationService(_testConversationsDir);

        // Act
        var result = await deleteService.DeleteConversationAsync(conversationId);

        // Assert
        Assert.True(result);
        var exists = await deleteService.ConversationExistsAsync(conversationId);
        Assert.False(exists);
    }

    [Fact]
    public async Task DeleteConversationAsync_CurrentConversation_ClearsCurrentState()
    {
        // Arrange
        var conversationId = $"test_{Guid.NewGuid():N}";
        await _service.StartOrResumeAsync(conversationId);
        _service.AddUserMessage("Test");
        await _service.SaveCurrentAsync();

        // Act
        var result = await _service.DeleteConversationAsync(conversationId);

        // Assert
        Assert.True(result);
        Assert.Null(_service.CurrentConversationId);
        Assert.Empty(_service.CurrentHistory);
    }

    [Fact]
    public async Task ConversationExistsAsync_ExistingConversation_ReturnsTrue()
    {
        // Arrange
        var conversationId = $"test_{Guid.NewGuid():N}";
        await _service.StartOrResumeAsync(conversationId);
        _service.AddUserMessage("Test");
        await _service.SaveCurrentAsync();

        // Act
        var exists = await _service.ConversationExistsAsync(conversationId);

        // Assert
        Assert.True(exists);
    }

    [Fact]
    public async Task ConversationExistsAsync_NonExistingConversation_ReturnsFalse()
    {
        // Act
        var exists = await _service.ConversationExistsAsync("nonexistent");

        // Assert
        Assert.False(exists);
    }

    #endregion

    #region GetMessagesForLLM Tests

    [Fact]
    public void GetMessagesForLLM_WithHistory_PrependsSystemPrompt()
    {
        // Arrange
        var systemPrompt = "You are a helpful assistant";
        _service.AddUserMessage("Hello");
        _service.AddAgentResponse("Hi!");

        // Act
        var messages = _service.GetMessagesForLLM(systemPrompt).ToList();

        // Assert
        Assert.Equal(3, messages.Count);
        Assert.Equal(ChatRole.System, messages[0].Role);
        Assert.Equal(systemPrompt, messages[0].Text);
        Assert.Equal(ChatRole.User, messages[1].Role);
        Assert.Equal(ChatRole.Assistant, messages[2].Role);
    }

    #endregion

    #region Helper Methods

    private void CleanupTestDirectory(string path)
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
