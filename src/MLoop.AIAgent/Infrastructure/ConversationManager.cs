using Microsoft.Extensions.Logging;

namespace MLoop.AIAgent.Infrastructure;

/// <summary>
/// Manages conversation context and history for AI agent interactions
/// </summary>
public class ConversationManager
{
    private readonly ILogger<ConversationManager> _logger;
    private readonly List<ConversationTurn> _history;
    private string? _currentProjectPath;
    private string? _currentDataFile;

    public ConversationManager(ILogger<ConversationManager> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _history = new List<ConversationTurn>();
    }

    /// <summary>
    /// Add a user message to conversation history
    /// </summary>
    public void AddUserMessage(string message)
    {
        _history.Add(new ConversationTurn
        {
            Timestamp = DateTime.UtcNow,
            Speaker = "user",
            Message = message
        });
        
        _logger.LogDebug("Added user message to history");
    }

    /// <summary>
    /// Add an agent response to conversation history
    /// </summary>
    public void AddAgentResponse(string agentName, string response)
    {
        _history.Add(new ConversationTurn
        {
            Timestamp = DateTime.UtcNow,
            Speaker = agentName,
            Message = response
        });
        
        _logger.LogDebug("Added agent '{AgentName}' response to history", agentName);
    }

    /// <summary>
    /// Get full conversation history
    /// </summary>
    public IReadOnlyList<ConversationTurn> GetHistory()
    {
        return _history.AsReadOnly();
    }

    /// <summary>
    /// Clear conversation history
    /// </summary>
    public void ClearHistory()
    {
        _history.Clear();
        _logger.LogInformation("Conversation history cleared");
    }

    /// <summary>
    /// Set current project path context
    /// </summary>
    public void SetProjectContext(string projectPath)
    {
        _currentProjectPath = projectPath;
        _logger.LogInformation("Project context set to: {ProjectPath}", projectPath);
    }

    /// <summary>
    /// Set current data file context
    /// </summary>
    public void SetDataFileContext(string dataFile)
    {
        _currentDataFile = dataFile;
        _logger.LogInformation("Data file context set to: {DataFile}", dataFile);
    }

    /// <summary>
    /// Get current project path
    /// </summary>
    public string? CurrentProjectPath => _currentProjectPath;

    /// <summary>
    /// Get current data file
    /// </summary>
    public string? CurrentDataFile => _currentDataFile;

    /// <summary>
    /// Get conversation context summary for agent
    /// </summary>
    public string GetContextSummary()
    {
        var summary = new List<string>();

        if (!string.IsNullOrEmpty(_currentProjectPath))
        {
            summary.Add($"Current Project: {_currentProjectPath}");
        }

        if (!string.IsNullOrEmpty(_currentDataFile))
        {
            summary.Add($"Current Data File: {_currentDataFile}");
        }

        if (_history.Count > 0)
        {
            summary.Add($"Conversation Turns: {_history.Count}");
            summary.Add($"Last User Message: {_history.LastOrDefault(t => t.Speaker == "user")?.Message}");
        }

        return string.Join("\n", summary);
    }
}

/// <summary>
/// Represents a single turn in the conversation
/// </summary>
public class ConversationTurn
{
    public DateTime Timestamp { get; set; }
    public string Speaker { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
}
