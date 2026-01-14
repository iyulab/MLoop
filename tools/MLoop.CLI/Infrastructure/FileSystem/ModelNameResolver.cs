using System.Text.RegularExpressions;
using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Resolves and manages model names within an MLoop project
/// </summary>
public partial class ModelNameResolver : IModelNameResolver
{
    private const string ModelsDirectory = "models";
    private const string ModelsIndexFileName = "models.json";
    private const string StagingDirectory = "staging";
    private const string ProductionDirectory = "production";
    private const string RegistryFileName = "registry.json";

    // Reserved names that cannot be used as model names
    private static readonly HashSet<string> ReservedNames = new(StringComparer.OrdinalIgnoreCase)
    {
        "staging", "production", "temp", "cache", "index", "registry"
    };

    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly ConfigLoader _configLoader;
    private readonly string _projectRoot;
    private readonly string _modelsPath;
    private readonly string _indexPath;

    public ModelNameResolver(
        IFileSystemManager fileSystem,
        IProjectDiscovery projectDiscovery,
        ConfigLoader configLoader)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
        _projectDiscovery = projectDiscovery ?? throw new ArgumentNullException(nameof(projectDiscovery));
        _configLoader = configLoader ?? throw new ArgumentNullException(nameof(configLoader));

        _projectRoot = _projectDiscovery.FindRoot();
        _modelsPath = _fileSystem.CombinePath(_projectRoot, ModelsDirectory);
        _indexPath = _fileSystem.CombinePath(
            _projectDiscovery.GetMLoopDirectory(_projectRoot),
            ModelsIndexFileName);
    }

    /// <inheritdoc />
    public string Resolve(string? name)
    {
        return string.IsNullOrWhiteSpace(name)
            ? ConfigDefaults.DefaultModelName
            : name.Trim().ToLowerInvariant();
    }

    /// <inheritdoc />
    public bool Exists(string name)
    {
        var modelPath = GetModelPath(name);
        return _fileSystem.DirectoryExists(modelPath);
    }

    /// <inheritdoc />
    public string GetModelPath(string name)
    {
        var resolvedName = Resolve(name);
        return _fileSystem.CombinePath(_modelsPath, resolvedName);
    }

    /// <inheritdoc />
    public async Task<IEnumerable<ModelSummary>> ListAsync(CancellationToken cancellationToken = default)
    {
        var summaries = new List<ModelSummary>();

        // Load config for model definitions
        var config = await _configLoader.LoadUserConfigAsync(cancellationToken);

        // Load model index for metadata
        var index = await LoadOrCreateIndexAsync(cancellationToken);

        // Scan models directory for existing models
        if (!_fileSystem.DirectoryExists(_modelsPath))
        {
            return summaries;
        }

        var modelDirs = Directory.GetDirectories(_modelsPath);

        foreach (var modelDir in modelDirs)
        {
            var modelName = Path.GetFileName(modelDir);

            // Get metadata from index or config
            var metadata = index.Models.GetValueOrDefault(modelName);
            var definition = config.Models.GetValueOrDefault(modelName);

            // Count experiments in staging
            var stagingPath = _fileSystem.CombinePath(modelDir, StagingDirectory);
            var experimentCount = 0;
            if (_fileSystem.DirectoryExists(stagingPath))
            {
                experimentCount = Directory.GetDirectories(stagingPath).Length;
            }

            // Check for production model
            string? productionExperiment = null;
            var registryPath = _fileSystem.CombinePath(modelDir, RegistryFileName);
            if (_fileSystem.FileExists(registryPath))
            {
                try
                {
                    var registry = await _fileSystem.ReadJsonAsync<Dictionary<string, Dictionary<string, object?>>>(
                        registryPath, cancellationToken);
                    if (registry.TryGetValue("production", out var prodEntry) &&
                        prodEntry.TryGetValue("experimentId", out var expId))
                    {
                        productionExperiment = expId?.ToString();
                    }
                }
                catch
                {
                    // Ignore registry read errors
                }
            }

            summaries.Add(new ModelSummary
            {
                Name = modelName,
                Task = definition?.Task ?? metadata?.Task ?? "unknown",
                Label = definition?.Label ?? metadata?.Label ?? "unknown",
                CreatedAt = metadata?.CreatedAt ?? Directory.GetCreationTimeUtc(modelDir),
                ExperimentCount = experimentCount,
                ProductionExperiment = productionExperiment,
                Description = definition?.Description ?? metadata?.Description
            });
        }

        return summaries.OrderBy(m => m.Name == ConfigDefaults.DefaultModelName ? 0 : 1)
                        .ThenBy(m => m.Name);
    }

    /// <inheritdoc />
    public async Task CreateAsync(string name, ModelDefinition definition, CancellationToken cancellationToken = default)
    {
        var resolvedName = Resolve(name);

        if (!IsValidName(resolvedName))
        {
            throw new ArgumentException($"Invalid model name: '{resolvedName}'. " +
                "Model names must be lowercase alphanumeric with hyphens, 2-50 characters.");
        }

        if (Exists(resolvedName))
        {
            throw new InvalidOperationException($"Model '{resolvedName}' already exists.");
        }

        // Create directory structure
        await EnsureModelDirectoryAsync(resolvedName, cancellationToken);

        // Update index
        var index = await LoadOrCreateIndexAsync(cancellationToken);
        index.Models[resolvedName] = new ModelMetadata
        {
            CreatedAt = DateTime.UtcNow,
            Task = definition.Task,
            Label = definition.Label,
            Description = definition.Description
        };
        await SaveIndexAsync(index, cancellationToken);
    }

    /// <inheritdoc />
    public async Task RemoveAsync(string name, CancellationToken cancellationToken = default)
    {
        var resolvedName = Resolve(name);

        if (resolvedName == ConfigDefaults.DefaultModelName)
        {
            throw new InvalidOperationException("Cannot remove the default model.");
        }

        if (!Exists(resolvedName))
        {
            throw new FileNotFoundException($"Model '{resolvedName}' not found.");
        }

        // Remove directory
        var modelPath = GetModelPath(resolvedName);
        Directory.Delete(modelPath, recursive: true);

        // Update index
        var index = await LoadOrCreateIndexAsync(cancellationToken);
        index.Models.Remove(resolvedName);
        await SaveIndexAsync(index, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<ModelDefinition?> GetDefinitionAsync(string name, CancellationToken cancellationToken = default)
    {
        var resolvedName = Resolve(name);
        var config = await _configLoader.LoadUserConfigAsync(cancellationToken);

        return config.Models.GetValueOrDefault(resolvedName);
    }

    /// <inheritdoc />
    public async Task EnsureModelDirectoryAsync(string name, CancellationToken cancellationToken = default)
    {
        var resolvedName = Resolve(name);
        var modelPath = GetModelPath(resolvedName);

        await _fileSystem.CreateDirectoryAsync(modelPath, cancellationToken);
        await _fileSystem.CreateDirectoryAsync(
            _fileSystem.CombinePath(modelPath, StagingDirectory), cancellationToken);
        await _fileSystem.CreateDirectoryAsync(
            _fileSystem.CombinePath(modelPath, ProductionDirectory), cancellationToken);
    }

    /// <inheritdoc />
    public bool IsValidName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return false;

        if (name.Length < 2 || name.Length > 50)
            return false;

        if (ReservedNames.Contains(name))
            return false;

        // Must be lowercase alphanumeric with hyphens, no leading/trailing hyphens
        return ModelNamePattern().IsMatch(name);
    }

    private async Task<ModelIndex> LoadOrCreateIndexAsync(CancellationToken cancellationToken)
    {
        if (!_fileSystem.FileExists(_indexPath))
        {
            return new ModelIndex();
        }

        try
        {
            return await _fileSystem.ReadJsonAsync<ModelIndex>(_indexPath, cancellationToken);
        }
        catch
        {
            return new ModelIndex();
        }
    }

    private async Task SaveIndexAsync(ModelIndex index, CancellationToken cancellationToken)
    {
        await _fileSystem.WriteJsonAsync(_indexPath, index, cancellationToken);
    }

    [GeneratedRegex(@"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")]
    private static partial Regex ModelNamePattern();
}
