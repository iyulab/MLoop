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
    /// Loads mloop.yaml (user configuration)
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
            .Build();

        return deserializer.Deserialize<MLoopConfig>(yamlContent) ?? new MLoopConfig();
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
