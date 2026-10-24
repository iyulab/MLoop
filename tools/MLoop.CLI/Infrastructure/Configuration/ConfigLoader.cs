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

    /// <summary>What a command says when an in-place edit fell back to writing <c>mloop.yaml</c> whole.</summary>
    public const string RewrittenWarning =
        "mloop.yaml was rewritten from its settings, so its comments and key order were not kept — its layout is not one the in-place edit reads.";

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

        var yamlContent = Serializer().Serialize(config);
        await _fileSystem.WriteTextAsync(yamlPath, yamlContent, cancellationToken);
    }

    /// <summary>
    /// Writes <paramref name="changes"/> (YAML key → value: <c>task</c>, <c>label</c>, <c>group_column</c>)
    /// into <c>mloop.yaml</c>'s definition of <paramref name="modelName"/> — or, when the file does not
    /// define it, <paramref name="config"/>'s definition of it as a new entry — on the file's own lines,
    /// so its comments and key order stay as written (<see cref="MLoopYamlEdit"/>). A file not in the
    /// shape those edits know, or one whose edit does not read back as intended, is written whole from
    /// <paramref name="config"/> as <see cref="SaveUserConfigAsync"/> does.
    /// </summary>
    /// <returns><see langword="true"/> when the file was edited in place; <see langword="false"/> when it was rewritten.</returns>
    public async Task<bool> SaveModelAsync(
        MLoopConfig config,
        string modelName,
        IReadOnlyList<(string Key, string Value)> changes,
        CancellationToken cancellationToken = default)
    {
        var yamlPath = _fileSystem.CombinePath(_projectDiscovery.FindRoot(), MLoopYamlFileName);
        var text = _fileSystem.FileExists(yamlPath) ? await _fileSystem.ReadTextAsync(yamlPath, cancellationToken) : null;
        if (text is null)
        {
            await SaveUserConfigAsync(config, cancellationToken);
            return true;   // no file, so nothing of the user's to keep
        }

        var edited = Defines(text, modelName)
                ? MLoopYamlEdit.SetModelScalars(text, modelName, changes)
                : MLoopYamlEdit.AddModel(text, Serializer().Serialize(new Dictionary<string, ModelDefinition> { [modelName] = config.Models[modelName] }));
        if (edited is not null && ReadsAs(edited, modelName, config.Models[modelName]))
        {
            await _fileSystem.WriteTextAsync(yamlPath, edited, cancellationToken);
            return true;
        }

        await SaveUserConfigAsync(config, cancellationToken);
        return false;
    }

    /// <summary>
    /// Writes <paramref name="config"/>'s value of one key of <paramref name="modelName"/> — <c>prep</c> or
    /// <c>columns</c>, as <paramref name="value"/> reads it — into <c>mloop.yaml</c> on that key's lines
    /// (<see cref="MLoopYamlEdit.SetModelBlock"/>); a <see langword="null"/> value removes the key. The rest of
    /// the file, its comments included, stays as written. A file not in that shape, or an edit that does not
    /// read back as the intended value, is written whole as <see cref="SaveUserConfigAsync"/> does.
    /// </summary>
    /// <returns><see langword="true"/> when the file was edited in place; <see langword="false"/> when it was rewritten.</returns>
    public async Task<bool> SaveModelKeyAsync(
        MLoopConfig config,
        string modelName,
        string key,
        Func<ModelDefinition, object?> value,
        CancellationToken cancellationToken = default)
    {
        var yamlPath = _fileSystem.CombinePath(_projectDiscovery.FindRoot(), MLoopYamlFileName);
        var text = _fileSystem.FileExists(yamlPath) ? await _fileSystem.ReadTextAsync(yamlPath, cancellationToken) : null;
        var intended = value(config.Models[modelName]);
        if (text is not null)
        {
            var block = intended is null ? null : Serializer().Serialize(new Dictionary<string, object> { [key] = intended });
            var edited = MLoopYamlEdit.SetModelBlock(text, modelName, key, block);
            if (edited is not null && ReadsAs(edited, modelName, value, intended))
            {
                await _fileSystem.WriteTextAsync(yamlPath, edited, cancellationToken);
                return true;
            }
        }

        await SaveUserConfigAsync(config, cancellationToken);
        return text is null;   // no file, so nothing of the user's to keep
    }

    private static ISerializer Serializer() => new SerializerBuilder()
        .WithNamingConvention(UnderscoredNamingConvention.Instance)
        .ConfigureDefaultValuesHandling(DefaultValuesHandling.OmitNull)
        .Build();

    private static IDeserializer Deserializer() => new DeserializerBuilder()
        .WithNamingConvention(UnderscoredNamingConvention.Instance)
        .IgnoreUnmatchedProperties()
        .Build();

    private static bool Defines(string yaml, string modelName)
    {
        try { return Deserializer().Deserialize<MLoopConfig>(yaml)?.Models.ContainsKey(modelName) == true; }
        catch (YamlDotNet.Core.YamlException) { return false; }
    }

    // The edited text parses, and the model's task, label and group column are the intended ones.
    private static bool ReadsAs(string yaml, string modelName, ModelDefinition intended)
    {
        try
        {
            var read = Deserializer().Deserialize<MLoopConfig>(yaml);
            return read?.Models.TryGetValue(modelName, out var model) == true
                   && model.Task == intended.Task && model.Label == intended.Label && model.GroupColumn == intended.GroupColumn;
        }
        catch (YamlDotNet.Core.YamlException)
        {
            return false;
        }
    }

    // The edited text parses, and the key reads as the intended value — compared as YAML, which is how
    // the value is stored (the list and map types have no value equality of their own).
    private static bool ReadsAs(string yaml, string modelName, Func<ModelDefinition, object?> value, object? intended)
    {
        try
        {
            var read = Deserializer().Deserialize<MLoopConfig>(yaml);
            return read?.Models.TryGetValue(modelName, out var model) == true
                   && Serializer().Serialize(value(model) ?? "") == Serializer().Serialize(intended ?? "");
        }
        catch (YamlDotNet.Core.YamlException)
        {
            return false;
        }
    }
}
