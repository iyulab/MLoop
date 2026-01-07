using Ironbees.Core.Conversation;
using MemoryIndexer.Interfaces;
using MemoryIndexer.Models;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace MLoop.AIAgent.Infrastructure;

/// <summary>
/// Enhanced conversation service with memory-indexer integration.
/// Combines Ironbees FileSystemConversationStore with VirtualContextManager for semantic memory.
/// </summary>
public class EnhancedConversationService : IDisposable
{
    private readonly IVirtualContextManager _vcm;
    private readonly IRecentlyBuffer _recentlyBuffer;
    private readonly ConversationService _legacyStore;
    private readonly ILogger<EnhancedConversationService> _logger;
    private string? _currentUserId;
    private string? _currentSessionId;
    private bool _memoryEnabled;
    private bool _disposed;

    /// <summary>
    /// Initializes EnhancedConversationService with memory-indexer VCM.
    /// </summary>
    /// <param name="vcm">Virtual Context Manager for semantic memory.</param>
    /// <param name="recentlyBuffer">Recently buffer for conversation staging.</param>
    /// <param name="conversationsDirectory">Directory for legacy conversation storage.</param>
    /// <param name="memoryEnabled">Enable memory-indexer features (default: true).</param>
    /// <param name="logger">Optional logger.</param>
    public EnhancedConversationService(
        IVirtualContextManager vcm,
        IRecentlyBuffer recentlyBuffer,
        string conversationsDirectory,
        bool memoryEnabled = true,
        ILogger<EnhancedConversationService>? logger = null)
    {
        _vcm = vcm ?? throw new ArgumentNullException(nameof(vcm));
        _recentlyBuffer = recentlyBuffer ?? throw new ArgumentNullException(nameof(recentlyBuffer));
        _legacyStore = new ConversationService(conversationsDirectory);
        _memoryEnabled = memoryEnabled;
        _logger = logger ?? NullLogger<EnhancedConversationService>.Instance;

        _logger.LogInformation("EnhancedConversationService initialized (memory: {Enabled})", _memoryEnabled);
    }

    #region Conversation Management

    /// <summary>
    /// Starts or resumes a conversation with memory context.
    /// </summary>
    /// <param name="userId">User identifier for memory indexing.</param>
    /// <param name="conversationId">Conversation identifier.</param>
    /// <param name="agentName">Optional agent name.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task StartOrResumeAsync(
        string userId,
        string conversationId,
        string? agentName = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(userId))
            throw new ArgumentException("User ID cannot be null or empty.", nameof(userId));

        _currentUserId = userId;
        _currentSessionId = conversationId;

        // Legacy conversation store
        await _legacyStore.StartOrResumeAsync(conversationId, agentName, cancellationToken);

        // Initialize VCM session if memory enabled
        if (_memoryEnabled)
        {
            await _vcm.InitializeAsync(userId, conversationId, null, cancellationToken);
            _logger.LogDebug("Initialized VCM session for user {UserId}", userId);
        }

        _logger.LogInformation("Started conversation {ConversationId} for user {UserId}",
            conversationId, userId);
    }

    /// <summary>
    /// Adds user message with automatic memory buffering.
    /// </summary>
    /// <param name="message">User message content.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task AddUserMessageAsync(
        string message,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(message))
            return;

        // Legacy store
        _legacyStore.AddUserMessage(message);

        // Memory-indexer: Add to Recently Buffer
        if (_memoryEnabled && _currentUserId != null && _currentSessionId != null)
        {
            var metadata = new Dictionary<string, string>
            {
                ["timestamp"] = DateTimeOffset.UtcNow.ToString("O")
            };

            await _recentlyBuffer.EnqueueAsync(
                message,
                _currentUserId,
                _currentSessionId,
                "user",
                metadata,
                cancellationToken);

            _logger.LogDebug("Added user message to Recently Buffer");
        }
    }

    /// <summary>
    /// Adds agent response with automatic memory buffering.
    /// </summary>
    /// <param name="response">Agent response content.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task AddAgentResponseAsync(
        string response,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(response))
            return;

        // Legacy store
        _legacyStore.AddAgentResponse(response);

        // Memory-indexer: Add to Recently Buffer
        if (_memoryEnabled && _currentUserId != null && _currentSessionId != null)
        {
            var metadata = new Dictionary<string, string>
            {
                ["timestamp"] = DateTimeOffset.UtcNow.ToString("O")
            };

            await _recentlyBuffer.EnqueueAsync(
                response,
                _currentUserId,
                _currentSessionId,
                "assistant",
                metadata,
                cancellationToken);

            _logger.LogDebug("Added agent response to Recently Buffer");
        }
    }

    /// <summary>
    /// Saves current conversation state.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task SaveCurrentAsync(CancellationToken cancellationToken = default)
    {
        await _legacyStore.SaveCurrentAsync(cancellationToken);
        _logger.LogDebug("Saved conversation state");
    }

    /// <summary>
    /// Gets current conversation history.
    /// </summary>
    public IReadOnlyList<ChatMessage> CurrentHistory => _legacyStore.CurrentHistory;

    #endregion

    #region Memory-Enhanced Features

    /// <summary>
    /// Retrieves relevant memories for the given query using semantic search.
    /// </summary>
    /// <param name="query">Search query.</param>
    /// <param name="limit">Maximum number of memories to retrieve.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of relevant memories.</returns>
    public async Task<IReadOnlyList<MemoryUnit>> RecallMemoriesAsync(
        string query,
        int limit = 5,
        CancellationToken cancellationToken = default)
    {
        if (!_memoryEnabled || !IsVcmInitialized())
        {
            return Array.Empty<MemoryUnit>();
        }

        var memories = await _vcm.PageInAsync(query, limit, cancellationToken);

        _logger.LogDebug("Retrieved {Count} memories for query: {Query}", memories.Count, query);
        return memories;
    }

    /// <summary>
    /// Gets working memory items (active context).
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Working memory items.</returns>
    public async Task<ContextUsageStatistics> GetWorkingMemoryAsync(
        CancellationToken cancellationToken = default)
    {
        if (!_memoryEnabled || !IsVcmInitialized())
        {
            return new ContextUsageStatistics();
        }

        var stats = await _vcm.GetContextUsageAsync(cancellationToken);
        _logger.LogDebug("Working memory: {Count} items, {Tokens} tokens",
            stats.WorkingMemoryCount, stats.WorkingMemoryTokens);
        return stats;
    }

    /// <summary>
    /// Builds enhanced prompt with memory context.
    /// </summary>
    /// <param name="userQuery">Current user query.</param>
    /// <param name="systemPrompt">Base system prompt.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Enhanced system prompt with memory context.</returns>
    public async Task<string> BuildMemoryEnhancedPromptAsync(
        string userQuery,
        string systemPrompt,
        CancellationToken cancellationToken = default)
    {
        if (!_memoryEnabled || !IsVcmInitialized())
        {
            return systemPrompt;
        }

        // Retrieve relevant memories
        var memories = await RecallMemoriesAsync(userQuery, limit: 5, cancellationToken);

        if (memories.Count == 0)
        {
            return systemPrompt;
        }

        var memoryContext = new System.Text.StringBuilder();
        memoryContext.AppendLine();
        memoryContext.AppendLine("## Memory Context");
        memoryContext.AppendLine();

        // Group memories by tier
        var workingMemories = memories.Where(m => m.Tier == MemoryTier.Working).ToList();
        var sessionMemories = memories.Where(m => m.Tier == MemoryTier.Session).ToList();
        var userMemories = memories.Where(m => m.Tier == MemoryTier.User).ToList();

        // User profile facts (Tier 3)
        if (userMemories.Count > 0)
        {
            memoryContext.AppendLine("### User Profile:");
            foreach (var fact in userMemories)
            {
                memoryContext.AppendLine($"- {fact.Content}");
            }
            memoryContext.AppendLine();
        }

        // Session context (Tier 2)
        if (sessionMemories.Count > 0)
        {
            memoryContext.AppendLine("### Session Context:");
            foreach (var memory in sessionMemories)
            {
                memoryContext.AppendLine($"- {memory.Content}");
            }
            memoryContext.AppendLine();
        }

        // Working memory (Tier 1)
        if (workingMemories.Count > 0)
        {
            memoryContext.AppendLine("### Recent Context:");
            foreach (var memory in workingMemories)
            {
                memoryContext.AppendLine($"- {memory.Content}");
            }
            memoryContext.AppendLine();
        }

        return systemPrompt + memoryContext.ToString();
    }

    /// <summary>
    /// Chat with automatic memory enhancement.
    /// </summary>
    /// <param name="userMessage">User message.</param>
    /// <param name="systemPrompt">Base system prompt.</param>
    /// <param name="chatClient">Chat client.</param>
    /// <param name="options">Chat options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Agent response.</returns>
    public async Task<string> ChatWithMemoryAsync(
        string userMessage,
        string systemPrompt,
        IChatClient chatClient,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Add user message
        await AddUserMessageAsync(userMessage, cancellationToken);

        // Build enhanced prompt with memories
        var enhancedPrompt = await BuildMemoryEnhancedPromptAsync(
            userMessage,
            systemPrompt,
            cancellationToken);

        // Get messages for LLM
        var messages = _legacyStore.GetMessagesForLLM(enhancedPrompt).ToList();

        // Get response
        var response = await chatClient.GetResponseAsync(messages, options, cancellationToken);
        var responseText = response.ToString() ?? string.Empty;

        // Add agent response
        await AddAgentResponseAsync(responseText, cancellationToken);

        // Auto-save
        await SaveCurrentAsync(cancellationToken);

        return responseText;
    }

    #endregion

    #region Context Management

    /// <summary>
    /// Sets project context for memory tagging.
    /// </summary>
    public void SetProjectContext(string projectPath)
    {
        _legacyStore.SetProjectContext(projectPath);
    }

    /// <summary>
    /// Sets data file context for memory tagging.
    /// </summary>
    public void SetDataFileContext(string dataFile)
    {
        _legacyStore.SetDataFileContext(dataFile);
    }

    /// <summary>
    /// Gets context summary including memory stats.
    /// </summary>
    public async Task<string> GetContextSummaryAsync(CancellationToken cancellationToken = default)
    {
        var legacySummary = _legacyStore.GetContextSummary();

        if (!_memoryEnabled || !IsVcmInitialized())
        {
            return legacySummary;
        }

        var stats = await GetWorkingMemoryAsync(cancellationToken);

        return $"{legacySummary}\n" +
               $"Working Memory: {stats.WorkingMemoryCount} items ({stats.WorkingMemoryTokens} tokens)\n" +
               $"Saturation: {stats.SaturationPercentage:F1}% ({stats.SaturationLevel})";
    }

    #endregion

    #region Configuration

    /// <summary>
    /// Enables or disables memory features at runtime.
    /// </summary>
    public void SetMemoryEnabled(bool enabled)
    {
        _memoryEnabled = enabled;
        _logger.LogInformation("Memory features {Status}", enabled ? "enabled" : "disabled");
    }

    /// <summary>
    /// Gets current memory enabled status.
    /// </summary>
    public bool IsMemoryEnabled => _memoryEnabled;

    /// <summary>
    /// Checks if VCM is initialized for the current session.
    /// </summary>
    private bool IsVcmInitialized()
    {
        return _vcm.State.IsInitialized;
    }

    #endregion

    #region IDisposable

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            _legacyStore?.Dispose();
        }

        _disposed = true;
    }

    #endregion
}
