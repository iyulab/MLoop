namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Experiment storage with atomic ID generation and thread-safe operations
/// MLOps convention: stores experiments in models/staging/
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
    private readonly string _experimentsPath;
    private readonly string _indexPath;

    public ExperimentStore(
        IFileSystemManager fileSystem,
        IProjectDiscovery projectDiscovery)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
        _projectDiscovery = projectDiscovery ?? throw new ArgumentNullException(nameof(projectDiscovery));

        _projectRoot = _projectDiscovery.FindRoot();

        // MLOps convention: models/staging/
        var modelsPath = _fileSystem.CombinePath(_projectRoot, ModelsDirectory);
        _experimentsPath = _fileSystem.CombinePath(modelsPath, StagingDirectory);

        _indexPath = _fileSystem.CombinePath(
            _projectDiscovery.GetMLoopDirectory(_projectRoot),
            IndexFileName);
    }

    public async Task<string> GenerateIdAsync(CancellationToken cancellationToken = default)
    {
        // Atomic ID generation with retry logic for concurrent access
        const int maxRetries = 5;
        const int retryDelayMs = 100;

        for (int retry = 0; retry < maxRetries; retry++)
        {
            try
            {
                var index = await LoadOrCreateIndexAsync(cancellationToken);
                var experimentId = $"exp-{index.NextId:D3}";

                // Update index
                index.NextId++;
                await SaveIndexAsync(index, cancellationToken);

                return experimentId;
            }
            catch (IOException) when (retry < maxRetries - 1)
            {
                // Another process may have locked the file, retry
                await Task.Delay(retryDelayMs, cancellationToken);
            }
        }

        throw new InvalidOperationException(
            "Failed to generate experiment ID after multiple retries. " +
            "This may indicate high concurrent access.");
    }

    public async Task SaveAsync(ExperimentData experiment, CancellationToken cancellationToken = default)
    {
        var experimentPath = GetExperimentPath(experiment.ExperimentId);
        await _fileSystem.CreateDirectoryAsync(experimentPath, cancellationToken);

        // Save metadata
        var metadataPath = _fileSystem.CombinePath(experimentPath, MetadataFileName);
        await _fileSystem.WriteJsonAsync(metadataPath, new
        {
            experiment.ExperimentId,
            experiment.Timestamp,
            experiment.Status,
            experiment.Task,
            Data = new
            {
                experiment.Config.DataFile,
                LabelColumn = experiment.Config.LabelColumn
            },
            experiment.Config,
            experiment.Result,
            Versions = new
            {
                MlNet = "4.0.0",
                MLoop = "0.1.0-alpha"
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
        await UpdateIndexWithExperimentAsync(experiment, cancellationToken);
    }

    public async Task<ExperimentData> LoadAsync(string experimentId, CancellationToken cancellationToken = default)
    {
        var experimentPath = GetExperimentPath(experimentId);
        if (!_fileSystem.DirectoryExists(experimentPath))
        {
            throw new FileNotFoundException($"Experiment not found: {experimentId}");
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
            ExperimentId = experimentId,
            Timestamp = DateTime.Parse(metadata["timestamp"].ToString()!),
            Status = metadata["status"].ToString()!,
            Task = metadata["task"].ToString()!,
            Config = config,
            Metrics = metrics
        };
    }

    public async Task<IEnumerable<ExperimentSummary>> ListAsync(CancellationToken cancellationToken = default)
    {
        var index = await LoadOrCreateIndexAsync(cancellationToken);
        return index.Experiments;
    }

    public string GetExperimentPath(string experimentId)
    {
        return _fileSystem.CombinePath(_experimentsPath, experimentId);
    }

    public bool ExperimentExists(string experimentId)
    {
        return _fileSystem.DirectoryExists(GetExperimentPath(experimentId));
    }

    private async Task<ExperimentIndex> LoadOrCreateIndexAsync(CancellationToken cancellationToken)
    {
        if (!_fileSystem.FileExists(_indexPath))
        {
            var newIndex = new ExperimentIndex
            {
                NextId = 1,
                Experiments = new List<ExperimentSummary>()
            };
            await SaveIndexAsync(newIndex, cancellationToken);
            return newIndex;
        }

        return await _fileSystem.ReadJsonAsync<ExperimentIndex>(_indexPath, cancellationToken);
    }

    private async Task SaveIndexAsync(ExperimentIndex index, CancellationToken cancellationToken)
    {
        await _fileSystem.WriteJsonAsync(_indexPath, index, cancellationToken);
    }

    private async Task UpdateIndexWithExperimentAsync(ExperimentData experiment, CancellationToken cancellationToken)
    {
        var index = await LoadOrCreateIndexAsync(cancellationToken);

        // Remove old entry if exists
        var experiments = index.Experiments.Where(e => e.ExperimentId != experiment.ExperimentId).ToList();

        // Add new entry
        experiments.Add(new ExperimentSummary
        {
            ExperimentId = experiment.ExperimentId,
            Timestamp = experiment.Timestamp,
            Status = experiment.Status,
            BestMetric = experiment.Metrics?.Values.FirstOrDefault()
        });

        index.Experiments = experiments;
        await SaveIndexAsync(index, cancellationToken);
    }

    private class ExperimentIndex
    {
        public int NextId { get; set; }
        public IEnumerable<ExperimentSummary> Experiments { get; set; } = Array.Empty<ExperimentSummary>();
    }
}
