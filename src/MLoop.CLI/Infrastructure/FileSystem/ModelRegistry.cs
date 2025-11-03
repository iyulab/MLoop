namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Model registry implementation with staging and production support
/// </summary>
public class ModelRegistry : IModelRegistry
{
    private const string ModelsDirectory = "models";
    private const string RegistryFileName = "registry.json";
    private const string ModelFileName = "model.zip";
    private const string MetadataFileName = "metadata.json";

    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;
    private readonly string _projectRoot;
    private readonly string _modelsPath;
    private readonly string _registryPath;

    public ModelRegistry(
        IFileSystemManager fileSystem,
        IProjectDiscovery projectDiscovery,
        IExperimentStore experimentStore)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
        _projectDiscovery = projectDiscovery ?? throw new ArgumentNullException(nameof(projectDiscovery));
        _experimentStore = experimentStore ?? throw new ArgumentNullException(nameof(experimentStore));

        _projectRoot = _projectDiscovery.FindRoot();
        _modelsPath = _fileSystem.CombinePath(_projectRoot, ModelsDirectory);
        _registryPath = _fileSystem.CombinePath(
            _projectDiscovery.GetMLoopDirectory(_projectRoot),
            RegistryFileName);
    }

    public async Task PromoteAsync(
        string experimentId,
        ModelStage stage,
        CancellationToken cancellationToken = default)
    {
        // Verify experiment exists
        if (!_experimentStore.ExperimentExists(experimentId))
        {
            throw new FileNotFoundException($"Experiment not found: {experimentId}");
        }

        var experimentPath = _experimentStore.GetExperimentPath(experimentId);
        var sourceModelPath = _fileSystem.CombinePath(experimentPath, ModelFileName);

        if (!_fileSystem.FileExists(sourceModelPath))
        {
            throw new FileNotFoundException(
                $"Model file not found for experiment {experimentId}. " +
                $"Expected: {sourceModelPath}");
        }

        // Create target directory
        var targetPath = GetModelPath(stage);
        await _fileSystem.CreateDirectoryAsync(targetPath, cancellationToken);

        // Copy model
        var targetModelPath = _fileSystem.CombinePath(targetPath, ModelFileName);
        await _fileSystem.CopyFileAsync(sourceModelPath, targetModelPath, overwrite: true, cancellationToken);

        // Load experiment metadata
        var experiment = await _experimentStore.LoadAsync(experimentId, cancellationToken);

        // Save metadata
        var metadataPath = _fileSystem.CombinePath(targetPath, MetadataFileName);
        await _fileSystem.WriteJsonAsync(metadataPath, new
        {
            Stage = stage.ToString().ToLowerInvariant(),
            experiment.ExperimentId,
            PromotedAt = DateTime.UtcNow,
            experiment.Metrics,
            experiment.Task,
            BestTrainer = experiment.Result?.BestTrainer
        }, cancellationToken);

        // Update registry
        await UpdateRegistryAsync(stage, experimentId, experiment.Metrics, cancellationToken);
    }

    public async Task<ModelInfo?> GetAsync(
        ModelStage stage,
        CancellationToken cancellationToken = default)
    {
        var registry = await LoadOrCreateRegistryAsync(cancellationToken);

        var stageName = stage.ToString().ToLowerInvariant();
        if (!registry.ContainsKey(stageName))
        {
            return null;
        }

        var entry = registry[stageName];
        return new ModelInfo
        {
            Stage = stage,
            ExperimentId = entry["experimentId"]?.ToString() ?? string.Empty,
            PromotedAt = DateTime.Parse(entry["promotedAt"]?.ToString() ?? DateTime.UtcNow.ToString()),
            Metrics = entry.ContainsKey("metrics")
                ? ((Dictionary<string, object>)entry["metrics"]).ToDictionary(
                    kvp => kvp.Key,
                    kvp => Convert.ToDouble(kvp.Value))
                : null,
            ModelPath = GetModelPath(stage)
        };
    }

    public async Task<IEnumerable<ModelInfo>> ListAsync(CancellationToken cancellationToken = default)
    {
        var models = new List<ModelInfo>();

        foreach (ModelStage stage in Enum.GetValues<ModelStage>())
        {
            var model = await GetAsync(stage, cancellationToken);
            if (model != null)
            {
                models.Add(model);
            }
        }

        return models;
    }

    public string GetModelPath(ModelStage stage)
    {
        var stageName = stage.ToString().ToLowerInvariant();
        return _fileSystem.CombinePath(_modelsPath, stageName);
    }

    private async Task<Dictionary<string, Dictionary<string, object>>> LoadOrCreateRegistryAsync(
        CancellationToken cancellationToken)
    {
        if (!_fileSystem.FileExists(_registryPath))
        {
            var newRegistry = new Dictionary<string, Dictionary<string, object>>();
            await SaveRegistryAsync(newRegistry, cancellationToken);
            return newRegistry;
        }

        return await _fileSystem.ReadJsonAsync<Dictionary<string, Dictionary<string, object>>>(
            _registryPath,
            cancellationToken);
    }

    private async Task SaveRegistryAsync(
        Dictionary<string, Dictionary<string, object>> registry,
        CancellationToken cancellationToken)
    {
        await _fileSystem.WriteJsonAsync(_registryPath, registry, cancellationToken);
    }

    private async Task UpdateRegistryAsync(
        ModelStage stage,
        string experimentId,
        Dictionary<string, double>? metrics,
        CancellationToken cancellationToken)
    {
        var registry = await LoadOrCreateRegistryAsync(cancellationToken);

        var stageName = stage.ToString().ToLowerInvariant();
        registry[stageName] = new Dictionary<string, object>
        {
            ["experimentId"] = experimentId,
            ["promotedAt"] = DateTime.UtcNow,
        };

        if (metrics != null)
        {
            registry[stageName]["metrics"] = metrics;
        }

        await SaveRegistryAsync(registry, cancellationToken);
    }
}
