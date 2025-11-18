using System.Text.Json.Serialization;

namespace MLoop.AIAgent.Infrastructure;

/// <summary>
/// Metadata for tracking agent versions and user modifications
/// Stored as .mloop-metadata.json in each agent directory
/// </summary>
public class AgentMetadata
{
    /// <summary>
    /// Version of the agent template (e.g., "1.0.0")
    /// </summary>
    [JsonPropertyName("version")]
    public string Version { get; set; } = "1.0.0";

    /// <summary>
    /// SHA256 checksum of system-prompt.md at installation time
    /// </summary>
    [JsonPropertyName("system_prompt_checksum")]
    public string SystemPromptChecksum { get; set; } = string.Empty;

    /// <summary>
    /// SHA256 checksum of agent.yaml at installation time
    /// </summary>
    [JsonPropertyName("agent_yaml_checksum")]
    public string AgentYamlChecksum { get; set; } = string.Empty;

    /// <summary>
    /// Whether the user has modified this agent
    /// If true, auto-update will skip this agent and warn instead
    /// </summary>
    [JsonPropertyName("user_modified")]
    public bool UserModified { get; set; } = false;

    /// <summary>
    /// Timestamp when this agent was installed
    /// </summary>
    [JsonPropertyName("installed_at")]
    public DateTime InstalledAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Timestamp of last update (if any)
    /// </summary>
    [JsonPropertyName("last_updated_at")]
    public DateTime? LastUpdatedAt { get; set; }

    /// <summary>
    /// Source of this agent: "builtin" or "custom"
    /// </summary>
    [JsonPropertyName("source")]
    public string Source { get; set; } = "builtin";
}
