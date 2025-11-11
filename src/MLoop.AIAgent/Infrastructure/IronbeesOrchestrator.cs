using Ironbees.Core;
using Microsoft.Extensions.Logging;

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
