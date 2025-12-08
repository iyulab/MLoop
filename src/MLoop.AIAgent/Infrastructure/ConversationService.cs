using Ironbees.Core.Conversation;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace MLoop.AIAgent.Infrastructure;

/// <summary>
/// Conversation service using Ironbees FileSystemConversationStore.
/// Implements Ironbees philosophy: direct List&lt;ChatMessage&gt; management with filesystem persistence.
/// </summary>
public class ConversationService : IDisposable
{
    private readonly IConversationStore _store;
    private readonly ILogger<ConversationService> _logger;
    private readonly List<ChatMessage> _currentHistory;
    private string? _currentConversationId;
    private string? _currentAgentName;
    private string? _currentProjectPath;
    private string? _currentDataFile;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of ConversationService with default file storage.
    /// </summary>
    /// <param name="conversationsDirectory">Base directory for storing conversations.</param>
    /// <param name="logger">Optional logger.</param>
    public ConversationService(
        string conversationsDirectory,
        ILogger<ConversationService>? logger = null)
    {
        if (string.IsNullOrWhiteSpace(conversationsDirectory))
            throw new ArgumentException("Conversations directory cannot be null or empty.", nameof(conversationsDirectory));

        _store = new FileSystemConversationStore(conversationsDirectory);
        _logger = logger ?? NullLogger<ConversationService>.Instance;
        _currentHistory = [];
    }

    /// <summary>
    /// Initializes a new instance of ConversationService with a custom store.
    /// </summary>
    /// <param name="store">Custom conversation store implementation.</param>
    /// <param name="logger">Optional logger.</param>
    public ConversationService(
        IConversationStore store,
        ILogger<ConversationService>? logger = null)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        _logger = logger ?? NullLogger<ConversationService>.Instance;
        _currentHistory = [];
    }

    #region Conversation Management

    /// <summary>
    /// Starts a new conversation or loads an existing one.
    /// </summary>
    /// <param name="conversationId">Unique conversation identifier.</param>
    /// <param name="agentName">Optional agent name for organization.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task StartOrResumeAsync(
        string conversationId,
        string? agentName = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(conversationId))
            throw new ArgumentException("Conversation ID cannot be null or empty.", nameof(conversationId));

        // Save current conversation if exists
        if (!string.IsNullOrEmpty(_currentConversationId) && _currentHistory.Count > 0)
        {
            await SaveCurrentAsync(cancellationToken);
        }

        _currentConversationId = conversationId;
        _currentAgentName = agentName;
        _currentHistory.Clear();

        // Try to load existing conversation
        var existingState = await _store.LoadAsync(conversationId, cancellationToken);
        if (existingState != null)
        {
            foreach (var msg in existingState.Messages)
            {
                _currentHistory.Add(msg.ToChatMessage());
            }
            _logger.LogDebug("Loaded {Count} messages from conversation {Id}", _currentHistory.Count, conversationId);
        }
        else
        {
            _logger.LogDebug("Started new conversation {Id}", conversationId);
        }
    }

    /// <summary>
    /// Adds a user message to the current conversation.
    /// </summary>
    /// <param name="message">User message content.</param>
    public void AddUserMessage(string message)
    {
        if (string.IsNullOrWhiteSpace(message))
            return;

        _currentHistory.Add(new ChatMessage(ChatRole.User, message));
        _logger.LogDebug("Added user message to conversation");
    }

    /// <summary>
    /// Adds an assistant/agent response to the current conversation.
    /// </summary>
    /// <param name="response">Agent response content.</param>
    public void AddAgentResponse(string response)
    {
        if (string.IsNullOrWhiteSpace(response))
            return;

        _currentHistory.Add(new ChatMessage(ChatRole.Assistant, response));
        _logger.LogDebug("Added agent response to conversation");
    }

    /// <summary>
    /// Adds a system message to the current conversation.
    /// </summary>
    /// <param name="systemMessage">System message content.</param>
    public void AddSystemMessage(string systemMessage)
    {
        if (string.IsNullOrWhiteSpace(systemMessage))
            return;

        _currentHistory.Add(new ChatMessage(ChatRole.System, systemMessage));
        _logger.LogDebug("Added system message to conversation");
    }

    /// <summary>
    /// Saves the current conversation to storage.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task SaveCurrentAsync(CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(_currentConversationId))
        {
            _logger.LogWarning("No active conversation to save");
            return;
        }

        var state = new ConversationState
        {
            ConversationId = _currentConversationId,
            AgentName = _currentAgentName,
            Messages = _currentHistory.Select(m => new ConversationMessage
            {
                Role = m.Role.Value,
                Content = m.Text ?? string.Empty,
                Timestamp = DateTimeOffset.UtcNow
            }).ToList(),
            LastUpdatedAt = DateTimeOffset.UtcNow,
            Metadata = BuildMetadata()
        };

        await _store.SaveAsync(state, cancellationToken);
        _logger.LogDebug("Saved conversation {Id} with {Count} messages", _currentConversationId, _currentHistory.Count);
    }

    /// <summary>
    /// Clears the current conversation history (does not delete from storage).
    /// </summary>
    public void ClearHistory()
    {
        _currentHistory.Clear();
        _logger.LogInformation("Conversation history cleared");
    }

    /// <summary>
    /// Gets the current conversation history as a read-only list.
    /// </summary>
    public IReadOnlyList<ChatMessage> CurrentHistory => _currentHistory.AsReadOnly();

    /// <summary>
    /// Gets messages ready for LLM submission (with system prompt prepended).
    /// </summary>
    /// <param name="systemPrompt">System prompt to prepend.</param>
    public IEnumerable<ChatMessage> GetMessagesForLLM(string systemPrompt)
    {
        yield return new ChatMessage(ChatRole.System, systemPrompt);
        foreach (var message in _currentHistory)
        {
            yield return message;
        }
    }

    /// <summary>
    /// Gets the current conversation ID.
    /// </summary>
    public string? CurrentConversationId => _currentConversationId;

    #endregion

    #region Context Management

    /// <summary>
    /// Sets the current project path context.
    /// </summary>
    /// <param name="projectPath">Project path.</param>
    public void SetProjectContext(string projectPath)
    {
        _currentProjectPath = projectPath;
        _logger.LogInformation("Project context set to: {ProjectPath}", projectPath);
    }

    /// <summary>
    /// Sets the current data file context.
    /// </summary>
    /// <param name="dataFile">Data file path.</param>
    public void SetDataFileContext(string dataFile)
    {
        _currentDataFile = dataFile;
        _logger.LogInformation("Data file context set to: {DataFile}", dataFile);
    }

    /// <summary>
    /// Gets the current project path.
    /// </summary>
    public string? CurrentProjectPath => _currentProjectPath;

    /// <summary>
    /// Gets the current data file.
    /// </summary>
    public string? CurrentDataFile => _currentDataFile;

    /// <summary>
    /// Gets a summary of current conversation context.
    /// </summary>
    public string GetContextSummary()
    {
        var summary = new List<string>();

        if (!string.IsNullOrEmpty(_currentConversationId))
        {
            summary.Add($"Conversation ID: {_currentConversationId}");
        }

        if (!string.IsNullOrEmpty(_currentAgentName))
        {
            summary.Add($"Agent: {_currentAgentName}");
        }

        if (!string.IsNullOrEmpty(_currentProjectPath))
        {
            summary.Add($"Project: {_currentProjectPath}");
        }

        if (!string.IsNullOrEmpty(_currentDataFile))
        {
            summary.Add($"Data File: {_currentDataFile}");
        }

        summary.Add($"Messages: {_currentHistory.Count}");

        var lastUserMessage = _currentHistory.LastOrDefault(m => m.Role == ChatRole.User);
        if (lastUserMessage != null)
        {
            var preview = lastUserMessage.Text?.Length > 50
                ? lastUserMessage.Text[..50] + "..."
                : lastUserMessage.Text;
            summary.Add($"Last User: {preview}");
        }

        return string.Join("\n", summary);
    }

    #endregion

    #region Store Operations

    /// <summary>
    /// Lists all conversation IDs, optionally filtered by agent name.
    /// </summary>
    /// <param name="agentName">Optional agent name filter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<IReadOnlyList<string>> ListConversationsAsync(
        string? agentName = null,
        CancellationToken cancellationToken = default)
    {
        return await _store.ListAsync(agentName, cancellationToken);
    }

    /// <summary>
    /// Deletes a conversation from storage.
    /// </summary>
    /// <param name="conversationId">Conversation ID to delete.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<bool> DeleteConversationAsync(
        string conversationId,
        CancellationToken cancellationToken = default)
    {
        var result = await _store.DeleteAsync(conversationId, cancellationToken);
        if (result && conversationId == _currentConversationId)
        {
            _currentConversationId = null;
            _currentHistory.Clear();
        }
        return result;
    }

    /// <summary>
    /// Checks if a conversation exists.
    /// </summary>
    /// <param name="conversationId">Conversation ID to check.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<bool> ConversationExistsAsync(
        string conversationId,
        CancellationToken cancellationToken = default)
    {
        return await _store.ExistsAsync(conversationId, cancellationToken);
    }

    #endregion

    #region Chat Integration

    /// <summary>
    /// Sends a message and gets a response, managing conversation history automatically.
    /// </summary>
    /// <param name="userMessage">User message to send.</param>
    /// <param name="systemPrompt">System prompt for context.</param>
    /// <param name="chatClient">Chat client to use.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Assistant response text.</returns>
    public async Task<string> ChatAsync(
        string userMessage,
        string systemPrompt,
        IChatClient chatClient,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Add user message
        AddUserMessage(userMessage);

        // Build messages for LLM
        var messages = GetMessagesForLLM(systemPrompt).ToList();

        // Get response
        var response = await chatClient.GetResponseAsync(messages, options, cancellationToken);
        var responseText = response.ToString() ?? string.Empty;

        // Add to history
        AddAgentResponse(responseText);

        // Auto-save if conversation ID is set
        if (!string.IsNullOrEmpty(_currentConversationId))
        {
            await SaveCurrentAsync(cancellationToken);
        }

        return responseText;
    }

    /// <summary>
    /// Streams a response, managing conversation history automatically.
    /// </summary>
    /// <param name="userMessage">User message to send.</param>
    /// <param name="systemPrompt">System prompt for context.</param>
    /// <param name="chatClient">Chat client to use.</param>
    /// <param name="options">Optional chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Async enumerable of response chunks.</returns>
    public async IAsyncEnumerable<string> StreamChatAsync(
        string userMessage,
        string systemPrompt,
        IChatClient chatClient,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Add user message
        AddUserMessage(userMessage);

        // Build messages for LLM
        var messages = GetMessagesForLLM(systemPrompt).ToList();

        // Stream response and collect full text
        var fullResponse = new System.Text.StringBuilder();
        await foreach (var update in chatClient.GetStreamingResponseAsync(messages, options, cancellationToken))
        {
            if (update.Text is not null)
            {
                fullResponse.Append(update.Text);
                yield return update.Text;
            }
        }

        // Add to history
        AddAgentResponse(fullResponse.ToString());

        // Auto-save if conversation ID is set
        if (!string.IsNullOrEmpty(_currentConversationId))
        {
            await SaveCurrentAsync(cancellationToken);
        }
    }

    #endregion

    #region Private Methods

    private Dictionary<string, string> BuildMetadata()
    {
        var metadata = new Dictionary<string, string>();

        if (!string.IsNullOrEmpty(_currentProjectPath))
        {
            metadata["projectPath"] = _currentProjectPath;
        }

        if (!string.IsNullOrEmpty(_currentDataFile))
        {
            metadata["dataFile"] = _currentDataFile;
        }

        return metadata;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes the conversation service, saving any pending changes.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes managed resources.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            // Save current conversation before disposing
            if (!string.IsNullOrEmpty(_currentConversationId) && _currentHistory.Count > 0)
            {
                SaveCurrentAsync().GetAwaiter().GetResult();
            }
        }

        _disposed = true;
    }

    #endregion
}
