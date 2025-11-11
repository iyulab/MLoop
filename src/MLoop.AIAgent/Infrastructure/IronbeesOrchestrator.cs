using Ironbees.Core;
using Ironbees.AgentFramework;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;

namespace MLoop.AIAgent.Infrastructure;

/// <summary>
/// Orchestrates Ironbees agents for MLoop AI operations
/// </summary>
public class IronbeesOrchestrator
{
    private readonly IAgentOrchestrator _orchestrator;
    private readonly ILogger<IronbeesOrchestrator> _logger;
    private bool _isInitialized;

    public IronbeesOrchestrator(
        IAgentOrchestrator orchestrator,
        ILogger<IronbeesOrchestrator> logger)
    {
        _orchestrator = orchestrator ?? throw new ArgumentNullException(nameof(orchestrator));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Factory method to create IronbeesOrchestrator with environment configuration.
    /// Supports multiple LLM providers in priority order.
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

        // Priority 1: GPUStack (local OpenAI-compatible endpoint)
        var gpuStackEndpoint = Environment.GetEnvironmentVariable("GPUSTACK_ENDPOINT");
        var gpuStackKey = Environment.GetEnvironmentVariable("GPUSTACK_API_KEY");

        if (!string.IsNullOrEmpty(gpuStackEndpoint) && !string.IsNullOrEmpty(gpuStackKey))
        {
            logger.LogInformation("Using GPUStack endpoint: {Endpoint}", gpuStackEndpoint);
            return CreateWithOpenAICompatible(gpuStackEndpoint, gpuStackKey, agentsDirectory, loggerFactory);
        }

        // Priority 2: Anthropic Claude
        var anthropicKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY");
        if (!string.IsNullOrEmpty(anthropicKey))
        {
            logger.LogInformation("Using Anthropic Claude API");
            return CreateWithAnthropic(anthropicKey, agentsDirectory, loggerFactory);
        }

        // Priority 3: Azure OpenAI
        var azureEndpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
        var azureKey = Environment.GetEnvironmentVariable("AZURE_OPENAI_KEY");

        if (!string.IsNullOrEmpty(azureEndpoint) && !string.IsNullOrEmpty(azureKey))
        {
            logger.LogInformation("Using Azure OpenAI endpoint: {Endpoint}", azureEndpoint);
            return CreateWithOpenAICompatible(azureEndpoint, azureKey, agentsDirectory, loggerFactory);
        }

        // Priority 4: OpenAI
        var openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (!string.IsNullOrEmpty(openAIKey))
        {
            logger.LogInformation("Using OpenAI API");
            return CreateWithOpenAICompatible("https://api.openai.com/v1", openAIKey, agentsDirectory, loggerFactory);
        }

        // No credentials found
        throw new InvalidOperationException(
            "No LLM provider credentials found. Please set one of the following in .env file:\n" +
            "  - GPUSTACK_ENDPOINT + GPUSTACK_API_KEY (local)\n" +
            "  - ANTHROPIC_API_KEY (Claude models)\n" +
            "  - AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY (Azure)\n" +
            "  - OPENAI_API_KEY (OpenAI)");
    }

    private static IronbeesOrchestrator CreateWithOpenAICompatible(
        string endpoint,
        string apiKey,
        string? agentsDirectory,
        ILoggerFactory loggerFactory)
    {
        // Set up dependency injection
        var services = new ServiceCollection();

        // Add logging services - must be added before Ironbees
        services.AddLogging(builder =>
        {
            builder.AddConsole();
            builder.SetMinimumLevel(LogLevel.Warning);
        });

        // Add Ironbees services with OpenAI-compatible endpoint
        services.AddIronbees(options =>
        {
            options.AzureOpenAIEndpoint = endpoint;
            options.AzureOpenAIKey = apiKey;
            options.AgentsDirectory = agentsDirectory;
            options.MinimumConfidenceThreshold = 0.3;
            options.UseMicrosoftAgentFramework = false;
        });

        // Add IronbeesOrchestrator
        services.AddSingleton<IronbeesOrchestrator>();

        // Build service provider
        var serviceProvider = services.BuildServiceProvider();

        // Return orchestrator instance
        return serviceProvider.GetRequiredService<IronbeesOrchestrator>();
    }

    private static IronbeesOrchestrator CreateWithAnthropic(
        string apiKey,
        string? agentsDirectory,
        ILoggerFactory loggerFactory)
    {
        // Set up dependency injection
        var services = new ServiceCollection();

        // Add logging services - must be added before Ironbees
        services.AddLogging(builder =>
        {
            builder.AddConsole();
            builder.SetMinimumLevel(LogLevel.Warning);
        });

        // Add Ironbees services with Anthropic configuration
        // Note: Ironbees may need to support Anthropic natively
        // For now, we configure it as OpenAI-compatible if possible
        services.AddIronbees(options =>
        {
            // Anthropic uses different API structure than OpenAI
            // This is a placeholder - actual implementation depends on Ironbees support
            options.AzureOpenAIEndpoint = "https://api.anthropic.com/v1";
            options.AzureOpenAIKey = apiKey;
            options.AgentsDirectory = agentsDirectory;
            options.MinimumConfidenceThreshold = 0.3;
            options.UseMicrosoftAgentFramework = false;
        });

        // Add IronbeesOrchestrator
        services.AddSingleton<IronbeesOrchestrator>();

        // Build service provider
        var serviceProvider = services.BuildServiceProvider();

        // Return orchestrator instance
        return serviceProvider.GetRequiredService<IronbeesOrchestrator>();
    }

    /// <summary>
    /// Initialize and load all agents from filesystem
    /// </summary>
    public async Task InitializeAsync()
    {
        if (_isInitialized)
        {
            _logger.LogWarning("Orchestrator already initialized");
            return;
        }

        _logger.LogInformation("Loading Ironbees agents...");
        
        try
        {
            await _orchestrator.LoadAgentsAsync();
            _isInitialized = true;
            _logger.LogInformation("Agents loaded successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load agents");
            throw;
        }
    }

    /// <summary>
    /// Process user request with specific agent
    /// </summary>
    public async Task<string> ProcessAsync(string request, string agentName)
    {
        EnsureInitialized();
        
        _logger.LogInformation("Processing request with agent '{AgentName}'", agentName);
        
        try
        {
            var response = await _orchestrator.ProcessAsync(request, agentName);
            return response;
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
        
        _logger.LogInformation("Processing request with auto agent selection");
        
        try
        {
            var response = await _orchestrator.ProcessAsync(request);
            return response;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to process request with auto selection");
            throw;
        }
    }

    /// <summary>
    /// Stream response from agent with specific agent
    /// </summary>
    public async IAsyncEnumerable<string> StreamAsync(string request, string agentName)
    {
        EnsureInitialized();
        
        _logger.LogInformation("Streaming request with agent '{AgentName}'", agentName);
        
        await foreach (var chunk in _orchestrator.StreamAsync(request, agentName))
        {
            yield return chunk;
        }
    }

    /// <summary>
    /// Stream response from agent with automatic selection
    /// </summary>
    public async IAsyncEnumerable<string> StreamWithAutoSelectionAsync(string request)
    {
        EnsureInitialized();
        
        _logger.LogInformation("Streaming request with auto selection");
        
        // First, select the best agent
        var selection = await _orchestrator.SelectAgentAsync(request);
        
        if (selection.SelectedAgent == null)
        {
            _logger.LogWarning("No suitable agent found for request");
            yield return "⚠️ No suitable agent found for this request.";
            yield break;
        }
        
        _logger.LogInformation("Auto-selected agent: {AgentName} (confidence: {Confidence})", 
            selection.SelectedAgent.Name, selection.ConfidenceScore);
        
        // Stream using the selected agent
        await foreach (var chunk in _orchestrator.StreamAsync(request, selection.SelectedAgent.Name))
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
        
        // This would require extending Ironbees IAgentOrchestrator interface
        // For now, return known agents
        return new[] 
        { 
            "data-analyst", 
            "preprocessing-expert", 
            "model-architect", 
            "mlops-manager" 
        };
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
