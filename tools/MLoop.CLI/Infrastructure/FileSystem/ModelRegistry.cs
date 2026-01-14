using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Model registry implementation for multi-model production deployment.
/// Each model has its own production slot at models/{modelName}/production/.
/// Registry stored per model at models/{modelName}/registry.json.
/// </summary>
public class ModelRegistry : IModelRegistry
{
    private const string ModelsDirectory = "models";
    private const string ProductionDirectory = "production";
    private const string RegistryFileName = "registry.json";
    private const string ModelFileName = "model.zip";
    private const string MetadataFileName = "metadata.json";

    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;
    private readonly string _projectRoot;
    private readonly string _modelsPath;

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
    }

    /// <inheritdoc />
    public async Task PromoteAsync(
        string modelName,
        string experimentId,
        CancellationToken cancellationToken = default)
    {
        var resolvedName = ResolveModelName(modelName);

        // Verify experiment exists
        if (!_experimentStore.ExperimentExists(resolvedName, experimentId))
        {
            throw new FileNotFoundException($"Experiment not found: {resolvedName}/{experimentId}");
        }

        var experimentPath = _experimentStore.GetExperimentPath(resolvedName, experimentId);
        var sourceModelPath = _fileSystem.CombinePath(experimentPath, ModelFileName);

        if (!_fileSystem.FileExists(sourceModelPath))
        {
            throw new FileNotFoundException(
                $"Model file not found for experiment {resolvedName}/{experimentId}. " +
                $"Expected: {sourceModelPath}");
        }

        // Create target directory
        var targetPath = GetProductionPath(resolvedName);
        await _fileSystem.CreateDirectoryAsync(targetPath, cancellationToken);

        // Copy model
        var targetModelPath = _fileSystem.CombinePath(targetPath, ModelFileName);
        await _fileSystem.CopyFileAsync(sourceModelPath, targetModelPath, overwrite: true, cancellationToken);

        // Load experiment metadata
        var experiment = await _experimentStore.LoadAsync(resolvedName, experimentId, cancellationToken);

        // Save metadata
        var metadataPath = _fileSystem.CombinePath(targetPath, MetadataFileName);
        await _fileSystem.WriteJsonAsync(metadataPath, new
        {
            ModelName = resolvedName,
            experiment.ExperimentId,
            PromotedAt = DateTime.UtcNow,
            experiment.Metrics,
            experiment.Task,
            BestTrainer = experiment.Result?.BestTrainer,
            LabelColumn = experiment.Config.LabelColumn
        }, cancellationToken);

        // Update registry
        await UpdateRegistryAsync(resolvedName, experimentId, experiment, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<ModelInfo?> GetProductionAsync(
        string modelName,
        CancellationToken cancellationToken = default)
    {
        var resolvedName = ResolveModelName(modelName);
        var registry = await LoadRegistryAsync(resolvedName, cancellationToken);

        if (registry == null || !registry.TryGetValue("production", out var entry))
        {
            return null;
        }

        return ParseModelInfo(resolvedName, entry);
    }

    /// <inheritdoc />
    public async Task<IEnumerable<ModelInfo>> ListAsync(
        string? modelName = null,
        CancellationToken cancellationToken = default)
    {
        var models = new List<ModelInfo>();

        if (!string.IsNullOrEmpty(modelName))
        {
            // List production for specific model
            var model = await GetProductionAsync(modelName, cancellationToken);
            if (model != null)
            {
                models.Add(model);
            }
        }
        else
        {
            // List production across all models
            if (!_fileSystem.DirectoryExists(_modelsPath))
            {
                return models;
            }

            var modelDirs = Directory.GetDirectories(_modelsPath);
            foreach (var modelDir in modelDirs)
            {
                var name = Path.GetFileName(modelDir);
                try
                {
                    var model = await GetProductionAsync(name, cancellationToken);
                    if (model != null)
                    {
                        models.Add(model);
                    }
                }
                catch
                {
                    // Skip models with invalid registry
                }
            }
        }

        return models.OrderBy(m => m.ModelName == ConfigDefaults.DefaultModelName ? 0 : 1)
                     .ThenBy(m => m.ModelName);
    }

    /// <inheritdoc />
    public string GetProductionPath(string modelName)
    {
        var resolvedName = ResolveModelName(modelName);
        var modelPath = _fileSystem.CombinePath(_modelsPath, resolvedName);
        return _fileSystem.CombinePath(modelPath, ProductionDirectory);
    }

    /// <inheritdoc />
    public bool HasProduction(string modelName)
    {
        var productionPath = GetProductionPath(modelName);
        var modelFilePath = _fileSystem.CombinePath(productionPath, ModelFileName);
        return _fileSystem.FileExists(modelFilePath);
    }

    /// <inheritdoc />
    public string GetProductionModelFile(string modelName)
    {
        var productionPath = GetProductionPath(modelName);
        return _fileSystem.CombinePath(productionPath, ModelFileName);
    }

    /// <inheritdoc />
    public async Task<bool> ShouldPromoteAsync(
        string modelName,
        string experimentId,
        string primaryMetric,
        CancellationToken cancellationToken = default)
    {
        var resolvedName = ResolveModelName(modelName);

        // Load new experiment metrics
        var experiment = await _experimentStore.LoadAsync(resolvedName, experimentId, cancellationToken);

        if (experiment.Metrics == null || !experiment.Metrics.ContainsKey(primaryMetric))
        {
            return false; // Cannot promote without metrics
        }

        var newMetricValue = experiment.Metrics[primaryMetric];

        // Get current production model
        var currentProduction = await GetProductionAsync(resolvedName, cancellationToken);

        if (currentProduction == null)
        {
            return true; // No production model yet, promote first model
        }

        if (currentProduction.Metrics == null || !currentProduction.Metrics.ContainsKey(primaryMetric))
        {
            return true; // Current production has no metrics, promote new model
        }

        var currentMetricValue = currentProduction.Metrics[primaryMetric];

        // Promote if new model is better
        // For most metrics (accuracy, r_squared, auc, f1), higher is better
        // For error metrics (mae, rmse, mse), lower is better
        var isErrorMetric = IsErrorMetric(primaryMetric);

        return isErrorMetric
            ? newMetricValue < currentMetricValue  // Lower error is better
            : newMetricValue > currentMetricValue; // Higher score is better
    }

    /// <inheritdoc />
    public async Task<bool> AutoPromoteAsync(
        string modelName,
        string experimentId,
        string primaryMetric,
        CancellationToken cancellationToken = default)
    {
        if (await ShouldPromoteAsync(modelName, experimentId, primaryMetric, cancellationToken))
        {
            await PromoteAsync(modelName, experimentId, cancellationToken);
            return true;
        }

        return false;
    }

    private string GetRegistryPath(string modelName)
    {
        var modelPath = _fileSystem.CombinePath(_modelsPath, modelName);
        return _fileSystem.CombinePath(modelPath, RegistryFileName);
    }

    private async Task<Dictionary<string, Dictionary<string, object?>>?> LoadRegistryAsync(
        string modelName,
        CancellationToken cancellationToken)
    {
        var registryPath = GetRegistryPath(modelName);

        if (!_fileSystem.FileExists(registryPath))
        {
            return null;
        }

        try
        {
            return await _fileSystem.ReadJsonAsync<Dictionary<string, Dictionary<string, object?>>>(
                registryPath,
                cancellationToken);
        }
        catch
        {
            return null;
        }
    }

    private async Task SaveRegistryAsync(
        string modelName,
        Dictionary<string, Dictionary<string, object?>> registry,
        CancellationToken cancellationToken)
    {
        // Ensure model directory exists
        var modelPath = _fileSystem.CombinePath(_modelsPath, modelName);
        await _fileSystem.CreateDirectoryAsync(modelPath, cancellationToken);

        var registryPath = GetRegistryPath(modelName);
        await _fileSystem.WriteJsonAsync(registryPath, registry, cancellationToken);
    }

    private async Task UpdateRegistryAsync(
        string modelName,
        string experimentId,
        ExperimentData experiment,
        CancellationToken cancellationToken)
    {
        var registry = await LoadRegistryAsync(modelName, cancellationToken)
            ?? new Dictionary<string, Dictionary<string, object?>>();

        registry["production"] = new Dictionary<string, object?>
        {
            ["experimentId"] = experimentId,
            ["promotedAt"] = DateTime.UtcNow,
            ["task"] = experiment.Task,
            ["bestTrainer"] = experiment.Result?.BestTrainer,
            ["labelColumn"] = experiment.Config.LabelColumn,
            ["metrics"] = experiment.Metrics
        };

        await SaveRegistryAsync(modelName, registry, cancellationToken);
    }

    private ModelInfo? ParseModelInfo(string modelName, Dictionary<string, object?> entry)
    {
        if (!entry.TryGetValue("experimentId", out var expIdObj) || expIdObj == null)
        {
            return null;
        }

        // Parse metrics from JSON (handles JsonElement from System.Text.Json)
        Dictionary<string, double>? metrics = null;
        if (entry.TryGetValue("metrics", out var metricsObj) && metricsObj != null)
        {
            metrics = ParseMetrics(metricsObj);
        }

        // Parse promoted date
        DateTime promotedAt = DateTime.UtcNow;
        if (entry.TryGetValue("promotedAt", out var promotedAtObj) && promotedAtObj != null)
        {
            if (promotedAtObj is DateTime dt)
            {
                promotedAt = dt;
            }
            else if (DateTime.TryParse(promotedAtObj.ToString(), out var parsed))
            {
                promotedAt = parsed;
            }
        }

        return new ModelInfo
        {
            ModelName = modelName,
            ExperimentId = expIdObj.ToString() ?? string.Empty,
            PromotedAt = promotedAt,
            Metrics = metrics,
            ModelPath = GetProductionPath(modelName),
            Task = entry.TryGetValue("task", out var taskObj) ? taskObj?.ToString() : null,
            BestTrainer = entry.TryGetValue("bestTrainer", out var trainerObj) ? trainerObj?.ToString() : null
        };
    }

    private static Dictionary<string, double>? ParseMetrics(object metricsObj)
    {
        if (metricsObj is System.Text.Json.JsonElement jsonElement)
        {
            if (jsonElement.ValueKind == System.Text.Json.JsonValueKind.Object)
            {
                var metrics = new Dictionary<string, double>();
                foreach (var prop in jsonElement.EnumerateObject())
                {
                    if (prop.Value.ValueKind == System.Text.Json.JsonValueKind.Number)
                    {
                        metrics[prop.Name] = prop.Value.GetDouble();
                    }
                }
                return metrics.Count > 0 ? metrics : null;
            }
        }
        else if (metricsObj is Dictionary<string, object> dict)
        {
            var metrics = new Dictionary<string, double>();
            foreach (var kvp in dict)
            {
                if (double.TryParse(kvp.Value?.ToString(), out var value))
                {
                    metrics[kvp.Key] = value;
                }
            }
            return metrics.Count > 0 ? metrics : null;
        }
        else if (metricsObj is Dictionary<string, double> directMetrics)
        {
            return directMetrics.Count > 0 ? directMetrics : null;
        }

        return null;
    }

    private static bool IsErrorMetric(string metricName)
    {
        var lower = metricName.ToLowerInvariant();
        return lower.Contains("error") ||
               lower.Contains("mae") ||
               lower.Contains("mse") ||
               lower.Contains("rmse") ||
               lower.Contains("loss");
    }

    private static string ResolveModelName(string? name)
    {
        return string.IsNullOrWhiteSpace(name)
            ? ConfigDefaults.DefaultModelName
            : name.Trim().ToLowerInvariant();
    }
}
