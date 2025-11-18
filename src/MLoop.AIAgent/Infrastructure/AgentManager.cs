using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace MLoop.AIAgent.Infrastructure;

/// <summary>
/// Manages installation, updates, and validation of AI agents
/// Handles both built-in system agents and user-custom agents
/// </summary>
public class AgentManager
{
    private readonly ILogger<AgentManager> _logger;
    private readonly string _userAgentsDirectory;
    private readonly string _builtInTemplatesDirectory;
    private const string MetadataFileName = ".mloop-metadata.json";

    public AgentManager(ILogger<AgentManager> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        _userAgentsDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".mloop", "agents");

        // Built-in templates are in AgentTemplates directory relative to application base directory
        var assemblyDir = AppContext.BaseDirectory
            ?? throw new InvalidOperationException("Could not determine assembly directory");
        _builtInTemplatesDirectory = Path.Combine(assemblyDir, "AgentTemplates");
    }

    /// <summary>
    /// Install or update built-in system agents
    /// </summary>
    /// <param name="force">If true, overwrites user modifications</param>
    public async Task<InstallationResult> InstallBuiltInAgentsAsync(bool force = false)
    {
        _logger.LogInformation("Installing built-in agents to {Directory}...", _userAgentsDirectory);

        if (!Directory.Exists(_builtInTemplatesDirectory))
        {
            _logger.LogError("Built-in templates directory not found: {Directory}", _builtInTemplatesDirectory);
            return new InstallationResult
            {
                Success = false,
                Message = $"Built-in templates directory not found: {_builtInTemplatesDirectory}"
            };
        }

        // Ensure user agents directory exists
        Directory.CreateDirectory(_userAgentsDirectory);

        var result = new InstallationResult { Success = true };
        var agentDirs = Directory.GetDirectories(_builtInTemplatesDirectory);

        foreach (var templateDir in agentDirs)
        {
            var agentName = Path.GetFileName(templateDir);

            try
            {
                var agentResult = await InstallAgentAsync(agentName, force);
                result.InstalledAgents.AddRange(agentResult.InstalledAgents);
                result.SkippedAgents.AddRange(agentResult.SkippedAgents);
                result.UpdatedAgents.AddRange(agentResult.UpdatedAgents);
                result.Warnings.AddRange(agentResult.Warnings);

                _logger.LogInformation("Installed agent '{AgentName}': {Status}",
                    agentName, agentResult.Message);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to install agent '{AgentName}'", agentName);
                result.Warnings.Add($"Failed to install {agentName}: {ex.Message}");
            }
        }

        result.Message = $"Installed {result.InstalledAgents.Count} agents, " +
                        $"updated {result.UpdatedAgents.Count}, " +
                        $"skipped {result.SkippedAgents.Count}";

        _logger.LogInformation("Installation complete: {Message}", result.Message);
        return result;
    }

    /// <summary>
    /// Install or update a specific agent
    /// </summary>
    public async Task<InstallationResult> InstallAgentAsync(string agentName, bool force = false)
    {
        var templateDir = Path.Combine(_builtInTemplatesDirectory, agentName);
        var targetDir = Path.Combine(_userAgentsDirectory, agentName);

        if (!Directory.Exists(templateDir))
        {
            return new InstallationResult
            {
                Success = false,
                Message = $"Agent template '{agentName}' not found"
            };
        }

        // Check if agent already exists
        var metadataPath = Path.Combine(targetDir, MetadataFileName);
        var isUpdate = File.Exists(metadataPath);
        AgentMetadata? existingMetadata = null;

        if (isUpdate)
        {
            existingMetadata = await LoadMetadataAsync(metadataPath);

            // Check for user modifications
            if (!force && existingMetadata.UserModified)
            {
                return new InstallationResult
                {
                    Success = true,
                    SkippedAgents = { agentName },
                    Warnings = { $"Agent '{agentName}' has user modifications. Use --force to overwrite." },
                    Message = $"Skipped '{agentName}' due to user modifications"
                };
            }

            // Check if update is needed by comparing checksums
            var currentChecksums = await CalculateChecksumsAsync(targetDir);
            if (!force && currentChecksums.SystemPromptChecksum == existingMetadata.SystemPromptChecksum &&
                currentChecksums.AgentYamlChecksum == existingMetadata.AgentYamlChecksum)
            {
                return new InstallationResult
                {
                    Success = true,
                    SkippedAgents = { agentName },
                    Message = $"Agent '{agentName}' is already up to date"
                };
            }
        }

        // Create target directory
        Directory.CreateDirectory(targetDir);

        // Copy files
        var yamlFile = Path.Combine(templateDir, "agent.yaml");
        var promptFile = Path.Combine(templateDir, "system-prompt.md");

        if (!File.Exists(yamlFile) || !File.Exists(promptFile))
        {
            return new InstallationResult
            {
                Success = false,
                Message = $"Agent template '{agentName}' is incomplete (missing agent.yaml or system-prompt.md)"
            };
        }

        File.Copy(yamlFile, Path.Combine(targetDir, "agent.yaml"), true);
        File.Copy(promptFile, Path.Combine(targetDir, "system-prompt.md"), true);

        // Calculate checksums
        var checksums = await CalculateChecksumsAsync(targetDir);

        // Create metadata
        var metadata = new AgentMetadata
        {
            Version = "1.0.0",
            SystemPromptChecksum = checksums.SystemPromptChecksum,
            AgentYamlChecksum = checksums.AgentYamlChecksum,
            UserModified = false,
            InstalledAt = existingMetadata?.InstalledAt ?? DateTime.UtcNow,
            LastUpdatedAt = isUpdate ? DateTime.UtcNow : null,
            Source = "builtin"
        };

        // Save metadata
        await SaveMetadataAsync(metadataPath, metadata);

        var result = new InstallationResult
        {
            Success = true,
            Message = isUpdate ? $"Updated agent '{agentName}'" : $"Installed agent '{agentName}'"
        };

        if (isUpdate)
        {
            result.UpdatedAgents.Add(agentName);
        }
        else
        {
            result.InstalledAgents.Add(agentName);
        }

        return result;
    }

    /// <summary>
    /// Check for user modifications in installed agents
    /// </summary>
    public async Task<List<AgentStatus>> CheckAgentStatusAsync()
    {
        var statuses = new List<AgentStatus>();

        if (!Directory.Exists(_userAgentsDirectory))
        {
            _logger.LogInformation("No agents directory found: {Directory}", _userAgentsDirectory);
            return statuses;
        }

        var agentDirs = Directory.GetDirectories(_userAgentsDirectory);

        foreach (var agentDir in agentDirs)
        {
            var agentName = Path.GetFileName(agentDir);
            var metadataPath = Path.Combine(agentDir, MetadataFileName);

            try
            {
                var status = await CheckSingleAgentStatusAsync(agentName);
                statuses.Add(status);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to check status for agent '{AgentName}'", agentName);
                statuses.Add(new AgentStatus
                {
                    Name = agentName,
                    IsInstalled = true,
                    HasError = true,
                    ErrorMessage = ex.Message
                });
            }
        }

        return statuses;
    }

    /// <summary>
    /// Check status of a specific agent
    /// </summary>
    public async Task<AgentStatus> CheckSingleAgentStatusAsync(string agentName)
    {
        var agentDir = Path.Combine(_userAgentsDirectory, agentName);
        var metadataPath = Path.Combine(agentDir, MetadataFileName);
        var templateDir = Path.Combine(_builtInTemplatesDirectory, agentName);

        var status = new AgentStatus
        {
            Name = agentName,
            IsInstalled = Directory.Exists(agentDir)
        };

        if (!status.IsInstalled)
        {
            status.HasUpdate = Directory.Exists(templateDir);
            return status;
        }

        // Load metadata
        if (!File.Exists(metadataPath))
        {
            status.HasError = true;
            status.ErrorMessage = "Missing metadata file";
            return status;
        }

        var metadata = await LoadMetadataAsync(metadataPath);
        status.Version = metadata.Version;
        status.IsBuiltIn = metadata.Source == "builtin";
        status.UserModified = metadata.UserModified;

        // Check for modifications by comparing checksums
        var currentChecksums = await CalculateChecksumsAsync(agentDir);

        if (currentChecksums.SystemPromptChecksum != metadata.SystemPromptChecksum ||
            currentChecksums.AgentYamlChecksum != metadata.AgentYamlChecksum)
        {
            status.UserModified = true;

            // Update metadata if not already marked
            if (!metadata.UserModified)
            {
                metadata.UserModified = true;
                await SaveMetadataAsync(metadataPath, metadata);
                _logger.LogInformation("Marked agent '{AgentName}' as user-modified", agentName);
            }
        }

        // Check for updates (only for built-in agents)
        if (status.IsBuiltIn && Directory.Exists(templateDir))
        {
            var templateChecksums = await CalculateChecksumsAsync(templateDir);
            status.HasUpdate = templateChecksums.SystemPromptChecksum != metadata.SystemPromptChecksum ||
                              templateChecksums.AgentYamlChecksum != metadata.AgentYamlChecksum;
        }

        return status;
    }

    /// <summary>
    /// Validate agent directory structure and files
    /// </summary>
    public async Task<ValidationResult> ValidateAgentAsync(string agentName)
    {
        var agentDir = Path.Combine(_userAgentsDirectory, agentName);
        var result = new ValidationResult { AgentName = agentName, IsValid = true };

        // Check directory exists
        if (!Directory.Exists(agentDir))
        {
            result.IsValid = false;
            result.Errors.Add($"Agent directory not found: {agentDir}");
            return result;
        }

        // Check required files
        var yamlPath = Path.Combine(agentDir, "agent.yaml");
        var promptPath = Path.Combine(agentDir, "system-prompt.md");
        var metadataPath = Path.Combine(agentDir, MetadataFileName);

        if (!File.Exists(yamlPath))
        {
            result.IsValid = false;
            result.Errors.Add("Missing agent.yaml file");
        }

        if (!File.Exists(promptPath))
        {
            result.IsValid = false;
            result.Errors.Add("Missing system-prompt.md file");
        }

        if (!File.Exists(metadataPath))
        {
            result.Warnings.Add("Missing metadata file (will be regenerated)");
        }

        // Validate YAML structure
        if (File.Exists(yamlPath))
        {
            try
            {
                var yaml = await File.ReadAllTextAsync(yamlPath);
                var deserializer = new YamlDotNet.Serialization.DeserializerBuilder().Build();
                var dict = deserializer.Deserialize<Dictionary<string, object>>(yaml);

                if (!dict.ContainsKey("name"))
                    result.Warnings.Add("agent.yaml missing 'name' field");
                if (!dict.ContainsKey("description"))
                    result.Warnings.Add("agent.yaml missing 'description' field");
                if (!dict.ContainsKey("system_prompt"))
                    result.Warnings.Add("agent.yaml missing 'system_prompt' field");
            }
            catch (Exception ex)
            {
                result.IsValid = false;
                result.Errors.Add($"Invalid YAML format: {ex.Message}");
            }
        }

        // Validate system prompt is not empty
        if (File.Exists(promptPath))
        {
            var content = await File.ReadAllTextAsync(promptPath);
            if (string.IsNullOrWhiteSpace(content))
            {
                result.Warnings.Add("system-prompt.md is empty");
            }
        }

        return result;
    }

    /// <summary>
    /// Calculate SHA256 checksums for agent files
    /// </summary>
    private async Task<(string SystemPromptChecksum, string AgentYamlChecksum)> CalculateChecksumsAsync(string agentDir)
    {
        var promptPath = Path.Combine(agentDir, "system-prompt.md");
        var yamlPath = Path.Combine(agentDir, "agent.yaml");

        var promptChecksum = await CalculateFileChecksumAsync(promptPath);
        var yamlChecksum = await CalculateFileChecksumAsync(yamlPath);

        return (promptChecksum, yamlChecksum);
    }

    /// <summary>
    /// Calculate SHA256 checksum for a file
    /// </summary>
    private async Task<string> CalculateFileChecksumAsync(string filePath)
    {
        if (!File.Exists(filePath))
            return string.Empty;

        using var stream = File.OpenRead(filePath);
        var hash = await SHA256.HashDataAsync(stream);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    /// <summary>
    /// Load agent metadata from JSON file
    /// </summary>
    private async Task<AgentMetadata> LoadMetadataAsync(string metadataPath)
    {
        var json = await File.ReadAllTextAsync(metadataPath);
        return JsonSerializer.Deserialize<AgentMetadata>(json)
            ?? new AgentMetadata();
    }

    /// <summary>
    /// Save agent metadata to JSON file
    /// </summary>
    private async Task SaveMetadataAsync(string metadataPath, AgentMetadata metadata)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true
        };
        var json = JsonSerializer.Serialize(metadata, options);
        await File.WriteAllTextAsync(metadataPath, json);
    }
}

/// <summary>
/// Result of agent installation operation
/// </summary>
public class InstallationResult
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public List<string> InstalledAgents { get; set; } = new();
    public List<string> UpdatedAgents { get; set; } = new();
    public List<string> SkippedAgents { get; set; } = new();
    public List<string> Warnings { get; set; } = new();
}

/// <summary>
/// Status of an installed agent
/// </summary>
public class AgentStatus
{
    public string Name { get; set; } = string.Empty;
    public bool IsInstalled { get; set; }
    public bool IsBuiltIn { get; set; }
    public string Version { get; set; } = string.Empty;
    public bool UserModified { get; set; }
    public bool HasUpdate { get; set; }
    public bool HasError { get; set; }
    public string? ErrorMessage { get; set; }
}

/// <summary>
/// Result of agent validation
/// </summary>
public class ValidationResult
{
    public string AgentName { get; set; } = string.Empty;
    public bool IsValid { get; set; }
    public List<string> Errors { get; set; } = new();
    public List<string> Warnings { get; set; } = new();
}
