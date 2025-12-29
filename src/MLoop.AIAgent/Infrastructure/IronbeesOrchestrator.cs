using System.Runtime.CompilerServices;
using System.Text.Json;
using Ironbees.AgentMode.Configuration;
using Ironbees.AgentMode.Providers;
using Ironbees.Core.Conversation;
using Ironbees.Core.Middleware;
using Ironbees.Core.Streaming;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;
using MLoop.AIAgent.Agents;

namespace MLoop.AIAgent.Infrastructure;

/// <summary>
/// Orchestrates AI agents for MLoop using Ironbees Agent Mode LLM infrastructure.
/// Integrates full middleware stack: Rate Limiting, Resilience, Token Tracking.
/// </summary>
public class IronbeesOrchestrator
{
    private readonly IChatClient _chatClient;
    private readonly LLMConfiguration _llmConfig;
    private readonly ILogger<IronbeesOrchestrator> _logger;
    private readonly IConversationStore _conversationStore;
    private readonly Dictionary<string, AgentConfiguration> _agents = new();
    private IList<AITool>? _tools;
    private bool _isInitialized;

    /// <summary>
    /// Gets the conversation store for managing conversation history.
    /// </summary>
    public IConversationStore ConversationStore => _conversationStore;

    /// <summary>
    /// Gets the underlying chat client with full middleware pipeline.
    /// </summary>
    public IChatClient ChatClient => _chatClient;

    /// <summary>
    /// Gets the registered tools for function calling.
    /// </summary>
    public IList<AITool>? Tools => _tools;

    public IronbeesOrchestrator(
        IChatClient chatClient,
        LLMConfiguration llmConfig,
        IConversationStore conversationStore,
        ILogger<IronbeesOrchestrator> logger)
    {
        _chatClient = chatClient ?? throw new ArgumentNullException(nameof(chatClient));
        _llmConfig = llmConfig ?? throw new ArgumentNullException(nameof(llmConfig));
        _conversationStore = conversationStore ?? throw new ArgumentNullException(nameof(conversationStore));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Agent configuration loaded from YAML.
    /// </summary>
    private record AgentConfiguration(
        string Name,
        string Description,
        string SystemPrompt,
        float Temperature = 0.0f,
        int MaxTokens = 4096,
        bool ConversationEnabled = true,
        int MaxHistoryTurns = 10,
        int ContextWindowTokens = 8192);

    /// <summary>
    /// Factory method to create IronbeesOrchestrator with environment configuration.
    /// Uses Ironbees Agent Mode multi-provider LLM infrastructure with full middleware stack.
    ///
    /// Environment variables (priority order):
    /// 1. GPUSTACK_ENDPOINT and GPUSTACK_API_KEY (preferred for local GPUStack)
    /// 2. ANTHROPIC_API_KEY (recommended for Claude models)
    /// 3. AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY (fallback for Azure OpenAI)
    /// 4. OPENAI_API_KEY (fallback for OpenAI)
    /// </summary>
    public static IronbeesOrchestrator CreateFromEnvironment(
        ILoggerFactory? loggerFactory = null,
        string? agentsDirectory = null,
        string? conversationsDirectory = null,
        bool useProductionSettings = true)
    {
        // Use provided logger factory or create a default one
        loggerFactory ??= LoggerFactory.Create(builder =>
        {
            builder.AddConsole();
            builder.SetMinimumLevel(LogLevel.Warning);
        });

        var logger = loggerFactory.CreateLogger<IronbeesOrchestrator>();

        // Detect LLM configuration from environment
        var config = DetectLLMConfiguration(logger);

        if (config == null)
        {
            throw new InvalidOperationException(
                "No LLM provider credentials found. Please set one of the following in .env file:\n" +
                "  - GPUSTACK_ENDPOINT + GPUSTACK_API_KEY + GPUSTACK_MODEL (local)\n" +
                "  - ANTHROPIC_API_KEY + ANTHROPIC_MODEL (Claude models)\n" +
                "  - AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY + AZURE_OPENAI_MODEL (Azure)\n" +
                "  - OPENAI_API_KEY + OPENAI_MODEL (OpenAI)");
        }

        // Create base IChatClient using Agent Mode factory pattern
        var registry = LLMProviderFactoryRegistry.CreateDefault();
        var factory = registry.GetFactory(config.Provider);
        var baseChatClient = factory.CreateChatClient(config);

        // Build middleware pipeline (Ironbees middleware stack integration)
        var chatClient = BuildMiddlewarePipeline(
            baseChatClient,
            loggerFactory,
            useProductionSettings);

        // Create conversation store
        conversationsDirectory ??= Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".mloop", "conversations");
        var conversationStore = new FileSystemConversationStore(
            conversationsDirectory,
            loggerFactory.CreateLogger<FileSystemConversationStore>());

        // Create orchestrator instance
        return new IronbeesOrchestrator(chatClient, config, conversationStore, logger);
    }

    /// <summary>
    /// Detects LLM configuration from environment variables.
    /// </summary>
    private static LLMConfiguration? DetectLLMConfiguration(ILogger logger)
    {
        // Priority 1: GPUStack (local OpenAI-compatible endpoint)
        var gpuStackEndpoint = Environment.GetEnvironmentVariable("GPUSTACK_ENDPOINT");
        var gpuStackKey = Environment.GetEnvironmentVariable("GPUSTACK_API_KEY");
        var gpuStackModel = Environment.GetEnvironmentVariable("GPUSTACK_MODEL");

        if (!string.IsNullOrEmpty(gpuStackEndpoint) && !string.IsNullOrEmpty(gpuStackKey))
        {
            var model = gpuStackModel ?? "default";
            logger.LogInformation("Using GPUStack endpoint: {Endpoint}, model: {Model}", gpuStackEndpoint, model);
            return new LLMConfiguration
            {
                Provider = LLMProvider.OpenAICompatible,
                Model = model,
                Endpoint = gpuStackEndpoint,
                ApiKey = gpuStackKey
            };
        }

        // Priority 2: Anthropic Claude
        var anthropicKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY");
        var anthropicModel = Environment.GetEnvironmentVariable("ANTHROPIC_MODEL");

        if (!string.IsNullOrEmpty(anthropicKey))
        {
            var model = anthropicModel ?? "claude-sonnet-4-20250514";
            logger.LogInformation("Using Anthropic Claude API, model: {Model}", model);
            return new LLMConfiguration
            {
                Provider = LLMProvider.Anthropic,
                Model = model,
                ApiKey = anthropicKey
            };
        }

        // Priority 3: Azure OpenAI
        var azureEndpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
        var azureKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY");
        var azureModel = Environment.GetEnvironmentVariable("AZURE_OPENAI_MODEL");

        if (!string.IsNullOrEmpty(azureEndpoint) && !string.IsNullOrEmpty(azureKey))
        {
            var model = azureModel ?? "gpt-4o";
            logger.LogInformation("Using Azure OpenAI endpoint: {Endpoint}, model: {Model}", azureEndpoint, model);
            return new LLMConfiguration
            {
                Provider = LLMProvider.AzureOpenAI,
                Model = model,
                Endpoint = azureEndpoint,
                ApiKey = azureKey
            };
        }

        // Priority 4: OpenAI
        var openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        var openAIModel = Environment.GetEnvironmentVariable("OPENAI_MODEL");

        if (!string.IsNullOrEmpty(openAIKey))
        {
            var model = openAIModel ?? "gpt-4o-mini";
            logger.LogInformation("Using OpenAI API, model: {Model}", model);
            return new LLMConfiguration
            {
                Provider = LLMProvider.OpenAI,
                Model = model,
                ApiKey = openAIKey
            };
        }

        return null;
    }

    /// <summary>
    /// Builds the full Ironbees middleware pipeline with function invocation support.
    /// Order: Function Invocation -> Rate Limiting -> Resilience -> Caching -> Token Tracking -> Base Client
    /// </summary>
    private static IChatClient BuildMiddlewarePipeline(
        IChatClient baseChatClient,
        ILoggerFactory loggerFactory,
        bool useProductionSettings)
    {
        // Select options based on environment
        var rateLimitOptions = useProductionSettings
            ? RateLimitOptions.Production
            : RateLimitOptions.Development;

        var resilienceOptions = useProductionSettings
            ? ResilienceOptions.Production
            : ResilienceOptions.Development;

        var cachingOptions = useProductionSettings
            ? CachingOptions.Production
            : CachingOptions.Development;

        // Check if caching is disabled via environment variable
        var cachingEnabled = Environment.GetEnvironmentVariable("MLOOP_CACHING_ENABLED");
        if (cachingEnabled?.Equals("false", StringComparison.OrdinalIgnoreCase) == true)
        {
            cachingOptions = CachingOptions.Disabled;
        }

        // Create token usage store (filesystem-based for observability)
        var tokenStoreDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".mloop", "token-usage");
        var tokenStore = new FileSystemTokenUsageStore(tokenStoreDirectory);

        // Create memory cache for response caching
        var memoryCache = new MemoryCache(new MemoryCacheOptions
        {
            SizeLimit = 100 // Max 100 cached responses
        });

        // Build pipeline: Function Invocation -> Rate Limiting -> Resilience -> Caching -> Token Tracking -> Base
        // Note: Middleware wraps from inner to outer, so we add in reverse order
        var withTokenTracking = new TokenTrackingMiddleware(
            baseChatClient,
            tokenStore,
            new TokenTrackingOptions { LogUsage = true },
            loggerFactory.CreateLogger<TokenTrackingMiddleware>());

        var withCaching = new CachingMiddleware(
            withTokenTracking,
            memoryCache,
            cachingOptions,
            loggerFactory.CreateLogger<CachingMiddleware>());

        var withResilience = new ResilienceMiddleware(
            withCaching,
            resilienceOptions,
            loggerFactory.CreateLogger<ResilienceMiddleware>());

        var withRateLimiting = new RateLimitingMiddleware(
            withResilience,
            rateLimitOptions,
            loggerFactory.CreateLogger<RateLimitingMiddleware>());

        // Add function invocation support using ChatClientBuilder
        // This enables automatic tool calling when tools are provided in ChatOptions
        var withFunctionInvocation = new ChatClientBuilder(withRateLimiting)
            .UseFunctionInvocation()
            .Build();

        return withFunctionInvocation;
    }

    /// <summary>
    /// Initialize and load all agents from filesystem.
    /// Auto-installs built-in agents if they don't exist.
    /// Checks for updates and warns about user modifications.
    /// </summary>
    public async Task InitializeAsync(string? agentsDirectory = null)
    {
        if (_isInitialized)
        {
            _logger.LogWarning("Orchestrator already initialized");
            return;
        }

        try
        {
            // Determine agents directory
            agentsDirectory ??= Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".mloop", "agents");

            // Check if agents directory exists and has agents
            if (!Directory.Exists(agentsDirectory) || !Directory.GetDirectories(agentsDirectory).Any())
            {
                _logger.LogInformation("No agents directory found: {Directory}", agentsDirectory);
                _logger.LogInformation("Auto-installing built-in agents...");

                // Auto-install built-in agents
                using var loggerFactory = LoggerFactory.Create(builder =>
                {
                    builder.AddConsole();
                    builder.SetMinimumLevel(LogLevel.Information);
                });
                var agentManager = new AgentManager(loggerFactory.CreateLogger<AgentManager>());
                var result = await agentManager.InstallBuiltInAgentsAsync(force: false);

                if (result.Success)
                {
                    _logger.LogInformation("Built-in agents installed successfully");
                }
                else
                {
                    _logger.LogWarning("Failed to auto-install agents: {Message}", result.Message);
                }
            }

            // Check for updates to built-in agents
            await CheckForAgentUpdatesAsync();

            // Load agents from filesystem
            _logger.LogInformation("Loading agents from {Directory}...", agentsDirectory);

            if (!Directory.Exists(agentsDirectory))
            {
                _logger.LogWarning("Agents directory still doesn't exist after auto-install: {Directory}", agentsDirectory);
                _isInitialized = true;
                return;
            }

            var agentDirs = Directory.GetDirectories(agentsDirectory);

            foreach (var agentDir in agentDirs)
            {
                try
                {
                    var agent = await LoadAgentFromDirectoryAsync(agentDir);
                    _agents[agent.Name] = agent;
                    _logger.LogInformation("Loaded agent: {Name}", agent.Name);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to load agent from {Directory}", agentDir);
                }
            }

            if (_agents.Count == 0)
            {
                _logger.LogWarning("No agents loaded");
            }

            _isInitialized = true;
            _logger.LogInformation("Initialization complete. Total agents: {Count}", _agents.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize agents");
            throw;
        }
    }

    /// <summary>
    /// Check for updates to built-in agents and warn if user has modified them.
    /// </summary>
    private async Task CheckForAgentUpdatesAsync()
    {
        try
        {
            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Warning);
            });

            var agentManager = new AgentManager(loggerFactory.CreateLogger<AgentManager>());
            var statuses = await agentManager.CheckAgentStatusAsync();

            var modifiedWithUpdates = statuses
                .Where(s => s.UserModified && s.HasUpdate)
                .ToList();

            if (modifiedWithUpdates.Any())
            {
                _logger.LogWarning("Updates available for modified agents:");
                foreach (var agent in modifiedWithUpdates)
                {
                    _logger.LogWarning("  - {AgentName} - Run 'mloop agents install --agent {AgentName} --force' to update",
                        agent.Name, agent.Name);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to check for agent updates");
        }
    }

    /// <summary>
    /// Load agent from directory containing agent.yaml and system-prompt.md.
    /// </summary>
    private async Task<AgentConfiguration> LoadAgentFromDirectoryAsync(string agentDir)
    {
        var agentName = Path.GetFileName(agentDir);
        var yamlPath = Path.Combine(agentDir, "agent.yaml");
        var promptPath = Path.Combine(agentDir, "system-prompt.md");

        if (!File.Exists(yamlPath))
        {
            throw new FileNotFoundException($"agent.yaml not found in {agentDir}");
        }

        if (!File.Exists(promptPath))
        {
            throw new FileNotFoundException($"system-prompt.md not found in {agentDir}");
        }

        // Load YAML configuration
        var yaml = await File.ReadAllTextAsync(yamlPath);
        var deserializer = new YamlDotNet.Serialization.DeserializerBuilder().Build();
        var dict = deserializer.Deserialize<Dictionary<string, object>>(yaml);

        var name = dict.GetValueOrDefault("name")?.ToString() ?? agentName;
        var description = dict.GetValueOrDefault("description")?.ToString() ?? "";

        // Load system prompt from markdown file
        var systemPrompt = await File.ReadAllTextAsync(promptPath);

        float temperature = 0.0f;
        if (dict.TryGetValue("temperature", out var tempObj) && tempObj != null)
        {
            temperature = Convert.ToSingle(tempObj);
        }

        int maxTokens = 4096;
        if (dict.TryGetValue("max_tokens", out var tokensObj) && tokensObj != null)
        {
            maxTokens = Convert.ToInt32(tokensObj);
        }

        // Conversation history settings
        bool conversationEnabled = true;
        if (dict.TryGetValue("conversation_enabled", out var convObj) && convObj != null)
        {
            conversationEnabled = Convert.ToBoolean(convObj);
        }

        int maxHistoryTurns = 10;
        if (dict.TryGetValue("max_history_turns", out var turnsObj) && turnsObj != null)
        {
            maxHistoryTurns = Convert.ToInt32(turnsObj);
        }

        int contextWindowTokens = 8192;
        if (dict.TryGetValue("context_window_tokens", out var windowObj) && windowObj != null)
        {
            contextWindowTokens = Convert.ToInt32(windowObj);
        }

        return new AgentConfiguration(
            name, description, systemPrompt, temperature, maxTokens,
            conversationEnabled, maxHistoryTurns, contextWindowTokens);
    }

    /// <summary>
    /// Process user request with specific agent and optional conversation history.
    /// </summary>
    public async Task<string> ProcessAsync(
        string request,
        string agentName,
        string? conversationId = null)
    {
        EnsureInitialized();

        if (!_agents.TryGetValue(agentName, out var agent))
        {
            throw new InvalidOperationException($"Agent '{agentName}' not found");
        }

        _logger.LogInformation("Processing request with agent '{AgentName}'", agentName);

        try
        {
            var messages = new List<ChatMessage>
            {
                new(ChatRole.System, agent.SystemPrompt)
            };

            // Load conversation history if enabled for this agent and conversationId is specified
            if (agent.ConversationEnabled && !string.IsNullOrEmpty(conversationId))
            {
                var existingState = await _conversationStore.LoadAsync(conversationId);
                if (existingState != null && existingState.Messages.Count > 0)
                {
                    // Limit to MaxHistoryTurns (each turn = 2 messages: user + assistant)
                    var maxMessages = agent.MaxHistoryTurns * 2;
                    var historyMessages = existingState.Messages
                        .TakeLast(maxMessages)
                        .Select(m => m.ToChatMessage());
                    messages.AddRange(historyMessages);
                }
            }

            // Add current user message
            messages.Add(new ChatMessage(ChatRole.User, request));

            var options = new ChatOptions
            {
                Temperature = agent.Temperature,
                MaxOutputTokens = agent.MaxTokens,
                Tools = _tools,  // Include registered tools for function calling
                AdditionalProperties = new AdditionalPropertiesDictionary
                {
                    ["AgentName"] = agentName,
                    ["SessionId"] = conversationId ?? Guid.NewGuid().ToString()
                }
            };

            if (HasTools)
            {
                _logger.LogDebug("Processing request with {ToolCount} tools available", _tools!.Count);
            }

            var response = await _chatClient.GetResponseAsync(messages, options);
            var responseText = response.ToString() ?? string.Empty;

            // Save conversation history if enabled for this agent and conversationId specified
            if (agent.ConversationEnabled && !string.IsNullOrEmpty(conversationId))
            {
                await _conversationStore.AppendMessageAsync(
                    conversationId,
                    new ConversationMessage { Role = "user", Content = request });
                await _conversationStore.AppendMessageAsync(
                    conversationId,
                    new ConversationMessage { Role = "assistant", Content = responseText });
            }

            return responseText;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to process request with agent '{AgentName}'", agentName);
            throw;
        }
    }

    /// <summary>
    /// Process user request with automatic agent selection.
    /// </summary>
    public async Task<string> ProcessAsync(string request, string? conversationId = null)
    {
        EnsureInitialized();

        if (_agents.Count == 0)
        {
            throw new InvalidOperationException("No agents loaded");
        }

        // Simple auto-selection: use first agent for now
        var selectedAgent = _agents.Values.First();

        _logger.LogInformation("Auto-selected agent '{AgentName}' for request", selectedAgent.Name);

        return await ProcessAsync(request, selectedAgent.Name, conversationId);
    }

    /// <summary>
    /// Stream response from agent with specific agent.
    /// </summary>
    public async IAsyncEnumerable<string> StreamAsync(
        string request,
        string agentName,
        string? conversationId = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        if (!_agents.TryGetValue(agentName, out var agent))
        {
            throw new InvalidOperationException($"Agent '{agentName}' not found");
        }

        _logger.LogInformation("Streaming request with agent '{AgentName}'", agentName);

        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, agent.SystemPrompt)
        };

        // Load conversation history if enabled for this agent and conversationId is specified
        if (agent.ConversationEnabled && !string.IsNullOrEmpty(conversationId))
        {
            var existingState = await _conversationStore.LoadAsync(conversationId, cancellationToken);
            if (existingState != null && existingState.Messages.Count > 0)
            {
                // Limit to MaxHistoryTurns (each turn = 2 messages: user + assistant)
                var maxMessages = agent.MaxHistoryTurns * 2;
                var historyMessages = existingState.Messages
                    .TakeLast(maxMessages)
                    .Select(m => m.ToChatMessage());
                messages.AddRange(historyMessages);
            }
        }

        messages.Add(new ChatMessage(ChatRole.User, request));

        var options = new ChatOptions
        {
            Temperature = agent.Temperature,
            MaxOutputTokens = agent.MaxTokens,
            Tools = _tools,  // Include registered tools for function calling
            AdditionalProperties = new AdditionalPropertiesDictionary
            {
                ["AgentName"] = agentName,
                ["SessionId"] = conversationId ?? Guid.NewGuid().ToString()
            }
        };

        var fullResponse = new System.Text.StringBuilder();

        await foreach (var update in _chatClient.GetStreamingResponseAsync(messages, options, cancellationToken))
        {
            if (update.Text is not null)
            {
                fullResponse.Append(update.Text);
                yield return update.Text;
            }
        }

        // Save conversation history if enabled for this agent and conversationId specified
        if (agent.ConversationEnabled && !string.IsNullOrEmpty(conversationId))
        {
            await _conversationStore.AppendMessageAsync(
                conversationId,
                new ConversationMessage { Role = "user", Content = request },
                cancellationToken);
            await _conversationStore.AppendMessageAsync(
                conversationId,
                new ConversationMessage { Role = "assistant", Content = fullResponse.ToString() },
                cancellationToken);
        }
    }

    /// <summary>
    /// Stream response from agent with automatic selection.
    /// </summary>
    public async IAsyncEnumerable<string> StreamWithAutoSelectionAsync(
        string request,
        string? conversationId = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        if (_agents.Count == 0)
        {
            _logger.LogWarning("No agents loaded");
            yield return "No agents loaded";
            yield break;
        }

        var selectedAgent = _agents.Values.First();

        _logger.LogInformation("Auto-selected agent '{AgentName}' for streaming", selectedAgent.Name);

        await foreach (var chunk in StreamAsync(request, selectedAgent.Name, conversationId, cancellationToken))
        {
            yield return chunk;
        }
    }

    #region StreamChunk Enhanced Streaming

    /// <summary>
    /// Stream response from agent with typed StreamChunk events.
    /// Provides rich streaming with text, usage, tool calls, and completion events.
    /// </summary>
    public async IAsyncEnumerable<StreamChunk> StreamWithEventsAsync(
        string request,
        string agentName,
        string? conversationId = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        if (!_agents.TryGetValue(agentName, out var agent))
        {
            yield return new ErrorChunk($"Agent '{agentName}' not found", IsFatal: true);
            yield break;
        }

        _logger.LogInformation("Streaming request with events using agent '{AgentName}'", agentName);

        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, agent.SystemPrompt)
        };

        // Load conversation history if enabled for this agent and conversationId is specified
        if (agent.ConversationEnabled && !string.IsNullOrEmpty(conversationId))
        {
            var existingState = await _conversationStore.LoadAsync(conversationId, cancellationToken);
            if (existingState != null && existingState.Messages.Count > 0)
            {
                // Limit to MaxHistoryTurns (each turn = 2 messages: user + assistant)
                var maxMessages = agent.MaxHistoryTurns * 2;
                var historyMessages = existingState.Messages
                    .TakeLast(maxMessages)
                    .Select(m => m.ToChatMessage());
                messages.AddRange(historyMessages);
            }
        }

        messages.Add(new ChatMessage(ChatRole.User, request));

        var options = new ChatOptions
        {
            Temperature = agent.Temperature,
            MaxOutputTokens = agent.MaxTokens,
            Tools = _tools,  // Include registered tools for function calling
            AdditionalProperties = new AdditionalPropertiesDictionary
            {
                ["AgentName"] = agentName,
                ["SessionId"] = conversationId ?? Guid.NewGuid().ToString()
            }
        };

        var fullResponse = new System.Text.StringBuilder();

        await foreach (var update in _chatClient.GetStreamingResponseAsync(messages, options, cancellationToken))
        {
            // Text content
            if (update.Text is not null)
            {
                fullResponse.Append(update.Text);
                yield return new TextChunk(update.Text, IsComplete: false);
            }

            // Tool call processing
            if (update.Contents is { Count: > 0 })
            {
                foreach (var content in update.Contents)
                {
                    if (content is FunctionCallContent functionCall)
                    {
                        yield return new ToolCallStartChunk(
                            functionCall.Name,
                            functionCall.CallId,
                            functionCall.Arguments as IReadOnlyDictionary<string, object>);
                    }
                    else if (content is FunctionResultContent functionResult)
                    {
                        yield return new ToolCallCompleteChunk(
                            functionResult.CallId ?? "unknown",
                            functionResult.CallId,
                            Success: true,
                            Result: functionResult.Result);
                    }
                }
            }

            // Note: ChatResponseUpdate doesn't provide Usage during streaming
            // Usage information is only available on final ChatResponse
        }

        // Emit final text chunk marked as complete
        if (fullResponse.Length > 0)
        {
            yield return new TextChunk(string.Empty, IsComplete: true);
        }

        // Note: Usage information is not available during streaming with MS.Extensions.AI
        // UsageChunk would need to be obtained from a non-streaming call if required

        // Save conversation history if enabled for this agent and conversationId specified
        if (agent.ConversationEnabled && !string.IsNullOrEmpty(conversationId))
        {
            await _conversationStore.AppendMessageAsync(
                conversationId,
                new ConversationMessage { Role = "user", Content = request },
                cancellationToken);
            await _conversationStore.AppendMessageAsync(
                conversationId,
                new ConversationMessage { Role = "assistant", Content = fullResponse.ToString() },
                cancellationToken);
        }

        // Emit completion chunk
        yield return new CompletionChunk(Success: true, FinishReason: "stop");
    }

    /// <summary>
    /// Stream response from agent with automatic selection and typed StreamChunk events.
    /// </summary>
    public async IAsyncEnumerable<StreamChunk> StreamWithAutoSelectionEventsAsync(
        string request,
        string? conversationId = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        EnsureInitialized();

        if (_agents.Count == 0)
        {
            _logger.LogWarning("No agents loaded");
            yield return new ErrorChunk("No agents loaded", IsFatal: true);
            yield break;
        }

        var selectedAgent = _agents.Values.First();

        _logger.LogInformation("Auto-selected agent '{AgentName}' for event streaming", selectedAgent.Name);

        // Emit metadata about agent selection
        yield return new MetadataChunk("selected_agent", selectedAgent.Name);

        await foreach (var chunk in StreamWithEventsAsync(request, selectedAgent.Name, conversationId, cancellationToken))
        {
            yield return chunk;
        }
    }

    #endregion

    /// <summary>
    /// Start a new conversation and return its ID.
    /// </summary>
    public async Task<string> StartConversationAsync(string agentName)
    {
        EnsureInitialized();

        if (!_agents.ContainsKey(agentName))
        {
            throw new InvalidOperationException($"Agent '{agentName}' not found");
        }

        var conversationId = Guid.NewGuid().ToString("N");

        await _conversationStore.SaveAsync(new ConversationState
        {
            ConversationId = conversationId,
            AgentName = agentName,
            Messages = [],
            CreatedAt = DateTimeOffset.UtcNow,
            LastUpdatedAt = DateTimeOffset.UtcNow
        });

        _logger.LogInformation("Started conversation {ConversationId} with agent {AgentName}",
            conversationId, agentName);

        return conversationId;
    }

    /// <summary>
    /// Get conversation history.
    /// </summary>
    public async Task<ConversationState?> GetConversationAsync(string conversationId)
    {
        return await _conversationStore.LoadAsync(conversationId);
    }

    /// <summary>
    /// List all conversations for an agent.
    /// </summary>
    public async Task<IReadOnlyList<string>> ListConversationsAsync(string? agentName = null)
    {
        return await _conversationStore.ListAsync(agentName);
    }

    /// <summary>
    /// Get list of available agent names.
    /// </summary>
    public IEnumerable<string> GetAvailableAgents()
    {
        EnsureInitialized();
        return _agents.Keys;
    }

    /// <summary>
    /// Register tools for function calling capability.
    /// Tools will be included in ChatOptions when processing requests.
    /// </summary>
    /// <param name="tools">List of AITools created via AIFunctionFactory.</param>
    public void RegisterTools(IList<AITool> tools)
    {
        _tools = tools ?? throw new ArgumentNullException(nameof(tools));
        _logger.LogInformation("Registered {Count} tools for function calling", tools.Count);

        foreach (var tool in tools)
        {
            if (tool is AIFunction function)
            {
                var desc = function.Description ?? "";
                _logger.LogDebug("  - {ToolName}: {Description}",
                    function.Name,
                    desc.Length > 50 ? desc[..50] + "..." : desc);
            }
        }
    }

    /// <summary>
    /// Check if tools are registered for function calling.
    /// </summary>
    public bool HasTools => _tools != null && _tools.Count > 0;

    private void EnsureInitialized()
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException(
                "Orchestrator not initialized. Call InitializeAsync() first.");
        }
    }
}
