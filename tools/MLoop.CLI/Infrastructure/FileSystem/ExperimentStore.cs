using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.Core.Evaluation;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Experiment storage with atomic ID generation and thread-safe operations.
/// MLOps convention: stores experiments in models/{modelName}/staging/{expId}/
/// </summary>
public class ExperimentStore : IExperimentStore
{
    // Layout names delegate to the single ExperimentLayout authority in MLoop.Core so this
    // writer and the MLoop.Ops readers cannot drift apart (F-33). Local aliases keep call sites
    // readable; the values have exactly one definition.
    private const string ModelsDirectory = ExperimentLayout.ModelsDirectory;
    private const string StagingDirectory = ExperimentLayout.StagingDirectory;
    private const string IndexFileName = ExperimentLayout.IndexFileName;
    private const string MetadataFileName = ExperimentLayout.MetadataFileName;
    private const string MetricsFileName = ExperimentLayout.MetricsFileName;
    private const string ConfigFileName = ExperimentLayout.ConfigFileName;
    private const string TrialsFileName = ExperimentLayout.TrialsFileName;
    private const string LeaderboardFileName = ExperimentLayout.LeaderboardFileName;

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
                // Record the actual tool version (assembly informational version, +hash stripped)
                // for reproducibility — was hardcoded "0.2.0", which every experiment falsely claimed.
                MLoop = Update.UpdateChecker.GetCurrentVersion()
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

        // Save the search's trial history — what else was tried and how it scored. Nothing is
        // written when the task fit a single pipeline: an empty leaderboard would claim a search
        // happened and found nothing.
        if (experiment.Trials.Count > 0)
            await SaveTrialsAsync(experimentPath, experiment, cancellationToken);

        // Update experiment index
        await UpdateIndexWithExperimentAsync(resolvedName, experiment, cancellationToken);
    }

    /// <summary>
    /// Writes the trial history two ways: <c>trials.ndjson</c> in completion order (streamable, one
    /// object per line) and <c>leaderboard.json</c> ranked by the metric the search optimized.
    /// </summary>
    /// <remarks>
    /// The ranking direction comes from <see cref="MetricDirection"/> rather than a local
    /// lower-is-better test, because that knowledge had already drifted across four sites once and
    /// ranked the worst clustering model first (F-27). When the metric is one this authority does not
    /// recognize, the leaderboard records the trials unranked and says so — sorting by a guessed
    /// direction would silently present the worst trial as the best.
    /// </remarks>
    private async Task SaveTrialsAsync(
        string experimentPath, ExperimentData experiment, CancellationToken cancellationToken)
    {
        var lines = experiment.Trials.Select(t => JsonSerializer.Serialize(t, TrialJsonOptions));
        var trialsPath = _fileSystem.CombinePath(experimentPath, TrialsFileName);
        await _fileSystem.WriteTextAsync(trialsPath, string.Join(Environment.NewLine, lines), cancellationToken);

        var metric = experiment.RankingMetric;
        var ranked = metric is not null && MetricDirection.IsKnown(metric)
            ? Rank(experiment.Trials, metric)
            : null;

        var leaderboardPath = _fileSystem.CombinePath(experimentPath, LeaderboardFileName);
        await _fileSystem.WriteJsonAsync(leaderboardPath, new
        {
            Metric = metric,
            Direction = metric is not null && MetricDirection.IsKnown(metric)
                ? (MetricDirection.IsLowerBetter(metric) ? "lower_is_better" : "higher_is_better")
                : "unknown",
            TrialCount = experiment.Trials.Count,
            Trials = ranked ?? experiment.Trials
        }, cancellationToken);
    }

    /// <summary>
    /// Best first. A trial that did not produce the ranking metric sorts last rather than being
    /// dropped — it ran, and the trials file records it.
    /// </summary>
    private static List<TrialRecord> Rank(IReadOnlyList<TrialRecord> trials, string metric)
    {
        var lowerIsBetter = MetricDirection.IsLowerBetter(metric);
        return [.. trials.OrderBy(t => t.Metrics.ContainsKey(metric) ? 0 : 1)
            .ThenBy(t => t.Metrics.TryGetValue(metric, out var v)
                ? (lowerIsBetter ? v : -v)
                : double.MaxValue)];
    }

    /// <summary>
    /// NDJSON is written directly rather than through <c>WriteJsonAsync</c> (one object per line, not
    /// one document), so the naming policy has to be restated here — every other artifact this store
    /// writes is camelCase, and a trials file in PascalCase would make the two halves of the same
    /// history disagree about what a field is called.
    /// </summary>
    private static readonly JsonSerializerOptions TrialJsonOptions = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

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

        // Reconstruct the training result (D5). SaveAsync persists a "result" block
        // (bestTrainer + trainingTimeSeconds), but LoadAsync previously dropped it, leaving
        // experiment.Result null — so promote re-wrote production/metadata.json and registry.json
        // with bestTrainer=null (observed for anomaly RandomizedPca). A completed experiment's
        // trainer identity must survive the round-trip; an experiment with no result block (in
        // progress/failed, or "result": null) keeps Result null.
        var result = ParseResult(metadata);

        return new ExperimentData
        {
            ModelName = resolvedName,
            ExperimentId = experimentId,
            Timestamp = DateTime.Parse(metadata["timestamp"].ToString()!),
            Status = metadata["status"].ToString()!,
            Task = metadata["task"].ToString()!,
            Config = config,
            Result = result,
            Metrics = metrics
        };
    }

    /// <summary>
    /// Reconstructs <see cref="ExperimentResult"/> from the persisted metadata "result" block
    /// (D5). Returns null when there is no completed result — missing key, JSON null, or a block
    /// lacking bestTrainer — so callers keep Result null rather than fabricating a trainer.
    /// </summary>
    private static ExperimentResult? ParseResult(Dictionary<string, object> metadata)
    {
        if (!metadata.TryGetValue("result", out var resultObj) ||
            resultObj is not JsonElement resultElement ||
            resultElement.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        var bestTrainer = resultElement.TryGetProperty("bestTrainer", out var bt) &&
                          bt.ValueKind == JsonValueKind.String
            ? bt.GetString()
            : null;

        if (string.IsNullOrEmpty(bestTrainer))
            return null;

        var trainingTime = resultElement.TryGetProperty("trainingTimeSeconds", out var tt) &&
                           tt.ValueKind == JsonValueKind.Number
            ? tt.GetDouble()
            : 0.0;

        return new ExperimentResult
        {
            BestTrainer = bestTrainer,
            TrainingTimeSeconds = trainingTime
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
            // Canonical primary-metric value (matches MetricName), not the insertion-order-dependent
            // first dictionary entry — so ranking sorts by the metric the experiment optimized (F-28).
            BestMetric = TaskMetadata.ResolvePrimaryMetricValue(
                experiment.Metrics, experiment.Config.Metric, experiment.Task),
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
