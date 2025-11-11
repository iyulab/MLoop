using System.Runtime.CompilerServices;
using Ironbees.AgentMode.Configuration;
using Ironbees.AgentMode.Providers;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;

namespace MLoop.AIAgent.Infrastructure;

/// <summary>
/// Orchestrates AI agents for MLoop using Ironbees Agent Mode LLM infrastructure
/// </summary>
public class IronbeesOrchestrator
{
    private readonly IChatClient _chatClient;
    private readonly LLMConfiguration _llmConfig;
    private readonly ILogger<IronbeesOrchestrator> _logger;
    private readonly Dictionary<string, AgentConfiguration> _agents = new();
    private bool _isInitialized;

    public IronbeesOrchestrator(
        IChatClient chatClient,
        LLMConfiguration llmConfig,
        ILogger<IronbeesOrchestrator> logger)
    {
        _chatClient = chatClient ?? throw new ArgumentNullException(nameof(chatClient));
        _llmConfig = llmConfig ?? throw new ArgumentNullException(nameof(llmConfig));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Agent configuration loaded from YAML
    /// </summary>
    private record AgentConfiguration(
        string Name,
        string Description,
        string SystemPrompt,
        float Temperature = 0.0f,
        int MaxTokens = 4096);

    /// <summary>
    /// Factory method to create IronbeesOrchestrator with environment configuration.
    /// Uses Ironbees Agent Mode multi-provider LLM infrastructure.
    ///
    /// Environment variables (priority order):
    /// 1. GPUSTACK_ENDPOINT and GPUSTACK_API_KEY (preferred for local GPUStack)
    /// 2. ANTHROPIC_API_KEY (recommended for Claude models)
    /// 3. AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY (fallback for Azure OpenAI)
    /// 4. OPENAI_API_KEY (fallback for OpenAI)
    /// </summary>
    public static IronbeesOrchestrator CreateFromEnvironment(
        ILoggerFactory? loggerFactory = null,
        string? agentsDirectory = null)
    {
        // Use provided logger factory or create a default one
        loggerFactory ??= LoggerFactory.Create(builder =>
        {
            builder.AddConsole();
            builder.SetMinimumLevel(LogLevel.Warning);
        });

        var logger = loggerFactory.CreateLogger<IronbeesOrchestrator>();

        // Detect LLM configuration from environment
        LLMConfiguration? config = null;
        string? model = null;

        // Priority 1: GPUStack (local OpenAI-compatible endpoint)
        var gpuStackEndpoint = Environment.GetEnvironmentVariable("GPUSTACK_ENDPOINT");
        var gpuStackKey = Environment.GetEnvironmentVariable("GPUSTACK_API_KEY");
        var gpuStackModel = Environment.GetEnvironmentVariable("GPUSTACK_MODEL");

        if (!string.IsNullOrEmpty(gpuStackEndpoint) && !string.IsNullOrEmpty(gpuStackKey))
        {
            model = gpuStackModel ?? "default";
            logger.LogInformation("Using GPUStack endpoint: {Endpoint}, model: {Model}", gpuStackEndpoint, model);
            config = new LLMConfiguration
            {
                Provider = LLMProvider.OpenAICompatible,
                Model = model,
                Endpoint = gpuStackEndpoint,
                ApiKey = gpuStackKey
            };
        }

        // Priority 2: Anthropic Claude
        if (config == null)
        {
            var anthropicKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY");
            var anthropicModel = Environment.GetEnvironmentVariable("ANTHROPIC_MODEL");

            if (!string.IsNullOrEmpty(anthropicKey))
            {
                model = anthropicModel ?? "claude-3-5-sonnet-20241022";
                logger.LogInformation("Using Anthropic Claude API, model: {Model}", model);
                config = new LLMConfiguration
                {
                    Provider = LLMProvider.Anthropic,
                    Model = model,
                    ApiKey = anthropicKey
                };
            }
        }

        // Priority 3: Azure OpenAI
        if (config == null)
        {
            var azureEndpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
            var azureKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY");
            var azureModel = Environment.GetEnvironmentVariable("AZURE_OPENAI_MODEL");

            if (!string.IsNullOrEmpty(azureEndpoint) && !string.IsNullOrEmpty(azureKey))
            {
                model = azureModel ?? "gpt-4o";
                logger.LogInformation("Using Azure OpenAI endpoint: {Endpoint}, model: {Model}",
                    azureEndpoint, model);
                config = new LLMConfiguration
                {
                    Provider = LLMProvider.AzureOpenAI,
                    Model = model,
                    Endpoint = azureEndpoint,
                    ApiKey = azureKey
                };
            }
        }

        // Priority 4: OpenAI
        if (config == null)
        {
            var openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            var openAIModel = Environment.GetEnvironmentVariable("OPENAI_MODEL");

            if (!string.IsNullOrEmpty(openAIKey))
            {
                model = openAIModel ?? "gpt-4o-mini";
                logger.LogInformation("Using OpenAI API, model: {Model}", model);
                config = new LLMConfiguration
                {
                    Provider = LLMProvider.OpenAI,
                    Model = model,
                    ApiKey = openAIKey
                };
            }
        }

        // No credentials found
        if (config == null)
        {
            throw new InvalidOperationException(
                "No LLM provider credentials found. Please set one of the following in .env file:\n" +
                "  - GPUSTACK_ENDPOINT + GPUSTACK_API_KEY + GPUSTACK_MODEL (local)\n" +
                "  - ANTHROPIC_API_KEY + ANTHROPIC_MODEL (Claude models)\n" +
                "  - AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY + AZURE_OPENAI_MODEL (Azure)\n" +
                "  - OPENAI_API_KEY + OPENAI_MODEL (OpenAI)");
        }

        // Create IChatClient using Agent Mode factory pattern
        var registry = LLMProviderFactoryRegistry.CreateDefault();
        var factory = registry.GetFactory(config.Provider);
        var chatClient = factory.CreateChatClient(config);

        // Create orchestrator instance
        return new IronbeesOrchestrator(chatClient, config, logger);
    }


    /// <summary>
    /// Initialize and load all agents from filesystem
    /// </summary>
    public async Task InitializeAsync(string? agentsDirectory = null)
    {
        if (_isInitialized)
        {
            _logger.LogWarning("Orchestrator already initialized");
            return;
        }

        agentsDirectory ??= Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".mloop", "agents");

        _logger.LogInformation("Loading agents from {Directory}...", agentsDirectory);

        try
        {
            if (!Directory.Exists(agentsDirectory))
            {
                _logger.LogWarning("Agents directory not found: {Directory}", agentsDirectory);
                _isInitialized = true;
                return;
            }

            var yamlFiles = Directory.GetFiles(agentsDirectory, "*.yaml", SearchOption.AllDirectories)
                .Concat(Directory.GetFiles(agentsDirectory, "*.yml", SearchOption.AllDirectories));

            foreach (var yamlFile in yamlFiles)
            {
                try
                {
                    var agent = await LoadAgentFromYamlAsync(yamlFile);
                    _agents[agent.Name] = agent;
                    _logger.LogInformation("Loaded agent: {Name}", agent.Name);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to load agent from {File}", yamlFile);
                }
            }

            _isInitialized = true;
            _logger.LogInformation("Loaded {Count} agents successfully", _agents.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load agents");
            throw;
        }
    }

    private async Task<AgentConfiguration> LoadAgentFromYamlAsync(string yamlFilePath)
    {
        var yaml = await File.ReadAllTextAsync(yamlFilePath);
        var deserializer = new YamlDotNet.Serialization.DeserializerBuilder().Build();
        var dict = deserializer.Deserialize<Dictionary<string, object>>(yaml);

        var name = dict.GetValueOrDefault("name")?.ToString()
            ?? Path.GetFileNameWithoutExtension(yamlFilePath);
        var description = dict.GetValueOrDefault("description")?.ToString() ?? "";
        var systemPrompt = dict.GetValueOrDefault("system_prompt")?.ToString()
            ?? dict.GetValueOrDefault("instructions")?.ToString() ?? "";

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

        return new AgentConfiguration(name, description, systemPrompt, temperature, maxTokens);
    }

    /// <summary>
    /// Process user request with specific agent
    /// </summary>
    public async Task<string> ProcessAsync(string request, string agentName)
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
                new(ChatRole.System, agent.SystemPrompt),
                new(ChatRole.User, request)
            };

            var options = new ChatOptions
            {
                Temperature = agent.Temperature,
                MaxOutputTokens = agent.MaxTokens
            };

            var response = await _chatClient.GetResponseAsync(messages, options);
            return response.ToString() ?? string.Empty;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to process request with agent '{AgentName}'", agentName);
            throw;
        }
    }

    /// <summary>
    /// Process user request with automatic agent selection
    /// </summary>
    public async Task<string> ProcessAsync(string request)
    {
        EnsureInitialized();

        if (_agents.Count == 0)
        {
            throw new InvalidOperationException("No agents loaded");
        }

        // Simple auto-selection: use first agent for now
        // TODO: Implement intelligent agent selection based on request analysis
        var selectedAgent = _agents.Values.First();

        _logger.LogInformation("Auto-selected agent '{AgentName}' for request", selectedAgent.Name);

        return await ProcessAsync(request, selectedAgent.Name);
    }

    /// <summary>
    /// Stream response from agent with specific agent
    /// </summary>
    public async IAsyncEnumerable<string> StreamAsync(string request, string agentName)
    {
        EnsureInitialized();

        if (!_agents.TryGetValue(agentName, out var agent))
        {
            throw new InvalidOperationException($"Agent '{agentName}' not found");
        }

        _logger.LogInformation("Streaming request with agent '{AgentName}'", agentName);

        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, agent.SystemPrompt),
            new(ChatRole.User, request)
        };

        var options = new ChatOptions
        {
            Temperature = agent.Temperature,
            MaxOutputTokens = agent.MaxTokens
        };

        await foreach (var update in _chatClient.GetStreamingResponseAsync(messages, options))
        {
            if (update.Text is not null)
            {
                yield return update.Text;
            }
        }
    }

    /// <summary>
    /// Stream response from agent with automatic selection
    /// </summary>
    public async IAsyncEnumerable<string> StreamWithAutoSelectionAsync(string request)
    {
        EnsureInitialized();

        if (_agents.Count == 0)
        {
            _logger.LogWarning("No agents loaded");
            yield return "⚠️ No agents loaded";
            yield break;
        }

        // Simple auto-selection: use first agent for now
        var selectedAgent = _agents.Values.First();

        _logger.LogInformation("Auto-selected agent '{AgentName}' for streaming", selectedAgent.Name);

        await foreach (var chunk in StreamAsync(request, selectedAgent.Name))
        {
            yield return chunk;
        }
    }

    /// <summary>
    /// Get list of available agent names
    /// </summary>
    public IEnumerable<string> GetAvailableAgents()
    {
        EnsureInitialized();
        return _agents.Keys;
    }

    private void EnsureInitialized()
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException(
                "Orchestrator not initialized. Call InitializeAsync() first.");
        }
    }
}
