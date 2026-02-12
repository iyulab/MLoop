using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Experiment storage with atomic ID generation and thread-safe operations.
/// MLOps convention: stores experiments in models/{modelName}/staging/{expId}/
/// </summary>
public class ExperimentStore : IExperimentStore
{
    private const string ModelsDirectory = "models";
    private const string StagingDirectory = "staging";
    private const string IndexFileName = "experiment-index.json";
    private const string MetadataFileName = "metadata.json";
    private const string MetricsFileName = "metrics.json";
    private const string ConfigFileName = "config.json";

    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly string _projectRoot;
    private readonly string _modelsPath;

    public ExperimentStore(
        IFileSystemManager fileSystem,
        IProjectDiscovery projectDiscovery)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
        _projectDiscovery = projectDiscovery ?? throw new ArgumentNullException(nameof(projectDiscovery));

        _projectRoot = _projectDiscovery.FindRoot();
        _modelsPath = _fileSystem.CombinePath(_projectRoot, ModelsDirectory);
    }

    /// <summary>
    /// Constructor with explicit project root (for testing).
    /// </summary>
    internal ExperimentStore(
        IFileSystemManager fileSystem,
        IProjectDiscovery projectDiscovery,
        string projectRoot)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
        _projectDiscovery = projectDiscovery ?? throw new ArgumentNullException(nameof(projectDiscovery));

        _projectRoot = projectRoot ?? throw new ArgumentNullException(nameof(projectRoot));
        _modelsPath = _fileSystem.CombinePath(_projectRoot, ModelsDirectory);
    }

    /// <inheritdoc />
    public async Task<string> GenerateIdAsync(string modelName, CancellationToken cancellationToken = default)
    {
        var resolvedName = ResolveModelName(modelName);

        // Atomic ID generation with retry logic for concurrent access
        const int maxRetries = 5;
        const int retryDelayMs = 100;

        for (int retry = 0; retry < maxRetries; retry++)
        {
            try
            {
                var index = await LoadOrCreateIndexAsync(resolvedName, cancellationToken);
                var experimentId = $"exp-{index.NextId:D3}";

                // Update index
                index.NextId++;
                await SaveIndexAsync(resolvedName, index, cancellationToken);

                return experimentId;
            }
            catch (IOException) when (retry < maxRetries - 1)
            {
                // Another process may have locked the file, retry
                await Task.Delay(retryDelayMs, cancellationToken);
            }
        }

        throw new InvalidOperationException(
            $"Failed to generate experiment ID for model '{resolvedName}' after multiple retries. " +
            "This may indicate high concurrent access.");
    }

    /// <inheritdoc />
    public async Task SaveAsync(string modelName, ExperimentData experiment, CancellationToken cancellationToken = default)
    {
        var resolvedName = ResolveModelName(modelName);
        var experimentPath = GetExperimentPath(resolvedName, experiment.ExperimentId);

        await _fileSystem.CreateDirectoryAsync(experimentPath, cancellationToken);

        // Save metadata
        var metadataPath = _fileSystem.CombinePath(experimentPath, MetadataFileName);
        await _fileSystem.WriteJsonAsync(metadataPath, new
        {
            experiment.ModelName,
            experiment.ExperimentId,
            experiment.Timestamp,
            experiment.Status,
            experiment.Task,
            Data = new
            {
                experiment.Config.DataFile,
                experiment.Config.LabelColumn
            },
            experiment.Config,
            experiment.Result,
            Versions = new
            {
                MlNet = "5.0.0",
                MLoop = "0.2.0"
            }
        }, cancellationToken);

        // Save metrics if available
        if (experiment.Metrics != null)
        {
            var metricsPath = _fileSystem.CombinePath(experimentPath, MetricsFileName);
            await _fileSystem.WriteJsonAsync(metricsPath, experiment.Metrics, cancellationToken);
        }

        // Save config
        var configPath = _fileSystem.CombinePath(experimentPath, ConfigFileName);
        await _fileSystem.WriteJsonAsync(configPath, experiment.Config, cancellationToken);

        // Update experiment index
        await UpdateIndexWithExperimentAsync(resolvedName, experiment, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<ExperimentData> LoadAsync(string modelName, string experimentId, CancellationToken cancellationToken = default)
    {
        var resolvedName = ResolveModelName(modelName);
        var experimentPath = GetExperimentPath(resolvedName, experimentId);

        if (!_fileSystem.DirectoryExists(experimentPath))
        {
            throw new FileNotFoundException($"Experiment not found: {resolvedName}/{experimentId}");
        }

        var metadataPath = _fileSystem.CombinePath(experimentPath, MetadataFileName);
        var metadata = await _fileSystem.ReadJsonAsync<Dictionary<string, object>>(metadataPath, cancellationToken);

        // Load metrics if exists
        Dictionary<string, double>? metrics = null;
        var metricsPath = _fileSystem.CombinePath(experimentPath, MetricsFileName);
        if (_fileSystem.FileExists(metricsPath))
        {
            metrics = await _fileSystem.ReadJsonAsync<Dictionary<string, double>>(metricsPath, cancellationToken);
        }

        // Load config
        var configPath = _fileSystem.CombinePath(experimentPath, ConfigFileName);
        var config = await _fileSystem.ReadJsonAsync<ExperimentConfig>(configPath, cancellationToken);

        return new ExperimentData
        {
            ModelName = resolvedName,
            ExperimentId = experimentId,
            Timestamp = DateTime.Parse(metadata["timestamp"].ToString()!),
            Status = metadata["status"].ToString()!,
            Task = metadata["task"].ToString()!,
            Config = config,
            Metrics = metrics
        };
    }

    /// <inheritdoc />
    public async Task<IEnumerable<ExperimentSummary>> ListAsync(string? modelName = null, CancellationToken cancellationToken = default)
    {
        var summaries = new List<ExperimentSummary>();

        if (!string.IsNullOrEmpty(modelName))
        {
            // List experiments for a specific model
            var resolvedName = ResolveModelName(modelName);
            var index = await LoadOrCreateIndexAsync(resolvedName, cancellationToken);
            summaries.AddRange(index.Experiments);
        }
        else
        {
            // List experiments across all models
            if (!_fileSystem.DirectoryExists(_modelsPath))
            {
                return summaries;
            }

            var modelDirs = Directory.GetDirectories(_modelsPath);
            foreach (var modelDir in modelDirs)
            {
                var name = Path.GetFileName(modelDir);
                try
                {
                    var index = await LoadOrCreateIndexAsync(name, cancellationToken);
                    summaries.AddRange(index.Experiments);
                }
                catch
                {
                    // Skip models with corrupted indexes
                }
            }
        }

        return summaries.OrderByDescending(e => e.Timestamp);
    }

    /// <inheritdoc />
    public string GetExperimentPath(string modelName, string experimentId)
    {
        var resolvedName = ResolveModelName(modelName);
        var modelPath = _fileSystem.CombinePath(_modelsPath, resolvedName);
        var stagingPath = _fileSystem.CombinePath(modelPath, StagingDirectory);
        return _fileSystem.CombinePath(stagingPath, experimentId);
    }

    /// <inheritdoc />
    public bool ExperimentExists(string modelName, string experimentId)
    {
        return _fileSystem.DirectoryExists(GetExperimentPath(modelName, experimentId));
    }

    private string GetIndexPath(string modelName)
    {
        var modelPath = _fileSystem.CombinePath(_modelsPath, modelName);
        return _fileSystem.CombinePath(modelPath, IndexFileName);
    }

    private async Task<ExperimentIndex> LoadOrCreateIndexAsync(string modelName, CancellationToken cancellationToken)
    {
        var indexPath = GetIndexPath(modelName);

        if (!_fileSystem.FileExists(indexPath))
        {
            // Ensure model directory structure exists
            var modelPath = _fileSystem.CombinePath(_modelsPath, modelName);
            await _fileSystem.CreateDirectoryAsync(modelPath, cancellationToken);
            await _fileSystem.CreateDirectoryAsync(
                _fileSystem.CombinePath(modelPath, StagingDirectory), cancellationToken);

            var newIndex = new ExperimentIndex
            {
                NextId = 1,
                Experiments = new List<ExperimentSummary>()
            };
            await SaveIndexAsync(modelName, newIndex, cancellationToken);
            return newIndex;
        }

        return await _fileSystem.ReadJsonAsync<ExperimentIndex>(indexPath, cancellationToken);
    }

    private async Task SaveIndexAsync(string modelName, ExperimentIndex index, CancellationToken cancellationToken)
    {
        var indexPath = GetIndexPath(modelName);
        await _fileSystem.WriteJsonAsync(indexPath, index, cancellationToken);
    }

    private async Task UpdateIndexWithExperimentAsync(
        string modelName,
        ExperimentData experiment,
        CancellationToken cancellationToken)
    {
        var index = await LoadOrCreateIndexAsync(modelName, cancellationToken);

        // Remove old entry if exists
        var experiments = index.Experiments
            .Where(e => e.ExperimentId != experiment.ExperimentId)
            .ToList();

        // Add new entry
        experiments.Add(new ExperimentSummary
        {
            ModelName = modelName,
            ExperimentId = experiment.ExperimentId,
            Timestamp = experiment.Timestamp,
            Status = experiment.Status,
            BestMetric = experiment.Metrics?.Values.FirstOrDefault(),
            LabelColumn = experiment.Config.LabelColumn,
            BestTrainer = experiment.Result?.BestTrainer,
            MetricName = experiment.Config.Metric,
            TrainingTimeSeconds = experiment.Result?.TrainingTimeSeconds
        });

        index.Experiments = experiments;
        await SaveIndexAsync(modelName, index, cancellationToken);
    }

    private static string ResolveModelName(string? name)
    {
        return string.IsNullOrWhiteSpace(name)
            ? ConfigDefaults.DefaultModelName
            : name.Trim().ToLowerInvariant();
    }
}
