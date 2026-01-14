using MLoop.CLI.Infrastructure.FileSystem;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// Loads MLoop configuration from JSON and YAML files
/// </summary>
public class ConfigLoader
{
    private const string ConfigJsonFileName = "config.json";
    private const string MLoopYamlFileName = "mloop.yaml";

    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;

    public ConfigLoader(
        IFileSystemManager fileSystem,
        IProjectDiscovery projectDiscovery)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
        _projectDiscovery = projectDiscovery ?? throw new ArgumentNullException(nameof(projectDiscovery));
    }

    /// <summary>
    /// Loads .mloop/config.json (project metadata)
    /// </summary>
    public async Task<MLoopConfig> LoadProjectConfigAsync(
        CancellationToken cancellationToken = default)
    {
        var projectRoot = _projectDiscovery.FindRoot();
        var mloopDir = _projectDiscovery.GetMLoopDirectory(projectRoot);
        var configPath = _fileSystem.CombinePath(mloopDir, ConfigJsonFileName);

        if (!_fileSystem.FileExists(configPath))
        {
            return new MLoopConfig();
        }

        return await _fileSystem.ReadJsonAsync<MLoopConfig>(configPath, cancellationToken);
    }

    /// <summary>
    /// Loads mloop.yaml (user configuration) with backward compatibility for old format
    /// </summary>
    public async Task<MLoopConfig> LoadUserConfigAsync(
        CancellationToken cancellationToken = default)
    {
        var projectRoot = _projectDiscovery.FindRoot();
        var yamlPath = _fileSystem.CombinePath(projectRoot, MLoopYamlFileName);

        if (!_fileSystem.FileExists(yamlPath))
        {
            return new MLoopConfig();
        }

        var yamlContent = await _fileSystem.ReadTextAsync(yamlPath, cancellationToken);

        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build();

        // Try loading as new multi-model format first
        var config = deserializer.Deserialize<MLoopConfig>(yamlContent);

        // Check if it's valid new format (has project and models)
        if (config != null && !string.IsNullOrEmpty(config.Project) && config.Models.Count > 0)
        {
            return config;
        }

        // Try loading as old single-model format and convert
        var oldConfig = TryLoadOldFormat(yamlContent, deserializer);
        if (oldConfig != null)
        {
            Console.WriteLine("[Warning] Detected old mloop.yaml format. Consider migrating to new multi-model format.");
            return oldConfig;
        }

        return config ?? new MLoopConfig();
    }

    /// <summary>
    /// Old mloop.yaml format (v1) for backward compatibility
    /// </summary>
    private class MLoopConfigV1
    {
        public string? ProjectName { get; set; }
        public string? Task { get; set; }
        public string? LabelColumn { get; set; }
        public int? TimeLimitSeconds { get; set; }
        public string? Metric { get; set; }
        public double? TestSplit { get; set; }
    }

    /// <summary>
    /// Tries to load old format and convert to new format
    /// </summary>
    private MLoopConfig? TryLoadOldFormat(string yamlContent, IDeserializer deserializer)
    {
        try
        {
            var oldConfig = deserializer.Deserialize<MLoopConfigV1>(yamlContent);

            if (oldConfig == null)
                return null;

            // Check if it has old format fields
            bool hasOldFields = !string.IsNullOrEmpty(oldConfig.ProjectName) ||
                               !string.IsNullOrEmpty(oldConfig.Task) ||
                               !string.IsNullOrEmpty(oldConfig.LabelColumn);

            if (!hasOldFields)
                return null;

            // Convert to new format
            var newConfig = new MLoopConfig
            {
                Project = oldConfig.ProjectName ?? "unnamed-project",
                Models = new Dictionary<string, ModelDefinition>()
            };

            // Only add default model if we have task and label
            if (!string.IsNullOrEmpty(oldConfig.Task) && !string.IsNullOrEmpty(oldConfig.LabelColumn))
            {
                newConfig.Models[ConfigDefaults.DefaultModelName] = new ModelDefinition
                {
                    Task = oldConfig.Task,
                    Label = oldConfig.LabelColumn,
                    Training = new TrainingSettings
                    {
                        TimeLimitSeconds = oldConfig.TimeLimitSeconds ?? ConfigDefaults.DefaultTimeLimitSeconds,
                        Metric = oldConfig.Metric ?? ConfigDefaults.DefaultMetric,
                        TestSplit = oldConfig.TestSplit ?? ConfigDefaults.DefaultTestSplit
                    }
                };
            }

            return newConfig;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Saves project configuration to .mloop/config.json
    /// </summary>
    public async Task SaveProjectConfigAsync(
        MLoopConfig config,
        CancellationToken cancellationToken = default)
    {
        var projectRoot = _projectDiscovery.FindRoot();
        var mloopDir = _projectDiscovery.GetMLoopDirectory(projectRoot);
        var configPath = _fileSystem.CombinePath(mloopDir, ConfigJsonFileName);

        await _fileSystem.WriteJsonAsync(configPath, config, cancellationToken);
    }

    /// <summary>
    /// Saves user configuration to mloop.yaml
    /// </summary>
    public async Task SaveUserConfigAsync(
        MLoopConfig config,
        CancellationToken cancellationToken = default)
    {
        var projectRoot = _projectDiscovery.FindRoot();
        var yamlPath = _fileSystem.CombinePath(projectRoot, MLoopYamlFileName);

        var serializer = new SerializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build();

        var yamlContent = serializer.Serialize(config);
        await _fileSystem.WriteTextAsync(yamlPath, yamlContent, cancellationToken);
    }
}
