using MLoop.CLI.Infrastructure.Configuration;
using MLoop.Core.Evaluation;
using MLoop.Core.Storage;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Model registry implementation for multi-model production deployment.
/// Each model has its own production slot at models/{modelName}/production/.
/// Registry stored per model at models/{modelName}/registry.json.
/// </summary>
public class ModelRegistry : IModelRegistry
{
    // Layout names — including the production registry filename — delegate to the single
    // ExperimentLayout authority so the writer here cannot drift from ModelNameResolver's reader
    // (the registry-name drift class; cycle-93/95).
    private const string ModelsDirectory = ExperimentLayout.ModelsDirectory;
    private const string ProductionDirectory = ExperimentLayout.ProductionDirectory;
    private const string RegistryFileName = ExperimentLayout.RegistryFileName;
    private const string ModelFileName = ExperimentLayout.ModelFileName;
    private const string MetadataFileName = ExperimentLayout.MetadataFileName;

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

        if (experiment.Metrics == null)
        {
            return false; // Cannot promote without metrics
        }

        // User-facing metric aliases (e.g. "f1") differ from the canonical keys the
        // EvaluationEngine stores (e.g. "f1_score"). Resolve to the stored key so the
        // quality gate and production comparison operate on the right value instead of
        // silently failing the lookup (which blocked first-model auto-promotion — BUG-45).
        // When the project metric is the deferred "auto" (init leaves image/OD tasks as auto),
        // fall back to the task's canonical metric so the gate still engages instead of being
        // silently skipped (BUG-46).
        var metricKey = ResolveCanonicalMetricKey(primaryMetric, experiment.Task, experiment.Metrics.Keys);

        // Extract class count from schema for dynamic thresholds
        int? classCount = null;
        if (experiment.Config?.InputSchema?.Columns != null && experiment.Config.LabelColumn != null)
        {
            var labelSchema = experiment.Config.InputSchema.Columns
                .FirstOrDefault(s => s.Name.Equals(experiment.Config.LabelColumn, StringComparison.OrdinalIgnoreCase));
            if (labelSchema?.UniqueValueCount > 0)
            {
                classCount = labelSchema.UniqueValueCount;
            }
        }

        // Check minimum metric threshold (quality gate) — only when the metric is present.
        // Threshold and error-direction are computed from the RESOLVED canonical key, not the
        // raw user/project metric: "auto" and aliases ("f1", "r2") have no entry in the
        // threshold table, so using the raw value silently skipped the gate (BUG-45/46 root).
        if (metricKey != null && !IsErrorMetric(metricKey))
        {
            var minThreshold = GetMinimumMetricThreshold(metricKey, classCount);
            if (minThreshold.HasValue && experiment.Metrics[metricKey] < minThreshold.Value)
            {
                return false; // Below minimum viable threshold
            }
        }

        // Degenerate model detection: high accuracy but zero F1
        // This indicates the model only predicts the majority class
        if (IsClassificationDegenerateModel(experiment.Metrics))
        {
            return false; // Block promotion for degenerate models
        }

        // Get current production model
        var currentProduction = await GetProductionAsync(resolvedName, cancellationToken);

        if (currentProduction?.Metrics == null)
        {
            return true; // No production model yet (or it has no metrics), promote first model
        }

        // Compare against production using the resolved metric key. If either side is
        // missing the metric, fall back to promoting the new (quality-gated) model.
        if (metricKey == null ||
            !experiment.Metrics.TryGetValue(metricKey, out var newMetricValue) ||
            !currentProduction.Metrics.TryGetValue(metricKey, out var currentMetricValue))
        {
            return true;
        }

        // Promote if new model is better
        // For most metrics (accuracy, r_squared, auc, f1), higher is better
        // For error metrics (mae, rmse, mse), lower is better. Use the resolved canonical key
        // so the direction matches the value actually compared (metricKey is non-null here).
        return IsErrorMetric(metricKey)
            ? newMetricValue < currentMetricValue  // Lower error is better
            : newMetricValue > currentMetricValue; // Higher score is better
    }

    /// <summary>
    /// Resolves a user-facing metric name/alias (e.g. "f1", "r2", "log-loss") to the
    /// canonical key actually present among <paramref name="availableKeys"/> (e.g.
    /// "f1_score", "r_squared", "log_loss"). The EvaluationEngine stores canonical keys,
    /// while the CLI accepts aliases — without this mapping a raw lookup silently misses
    /// (the root cause of BUG-45's blocked auto-promotion and Compare's ignored --sort).
    /// Returns the matching canonical key, or null if no known variant is present.
    /// </summary>
    public static string? ResolveMetricKey(string metricName, IEnumerable<string> availableKeys)
    {
        if (string.IsNullOrWhiteSpace(metricName))
        {
            return null;
        }

        var keys = availableKeys as ICollection<string> ?? availableKeys.ToList();

        // Exact match wins.
        if (keys.Contains(metricName))
        {
            return metricName;
        }

        var normalized = metricName.Trim().ToLowerInvariant().Replace("-", "_");

        // Map common aliases to their canonical stored keys (most-specific first).
        var candidates = normalized switch
        {
            "f1" or "f1score" => new[] { "f1_score", "macro_f1" },
            "r2" or "rsquared" => new[] { "r_squared" },
            "logloss" => new[] { "log_loss" },
            "accuracy" => new[] { "accuracy", "micro_accuracy", "macro_accuracy" },
            "area_under_roc_curve" => new[] { "auc" },
            _ => new[] { normalized }
        };

        foreach (var candidate in candidates)
        {
            if (keys.Contains(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    /// <summary>
    /// Resolves the canonical metric key for the quality gate, falling back to the task's
    /// default metric when <paramref name="metricName"/> is the deferred "auto" (or otherwise
    /// unresolvable). <see cref="ResolveMetricKey"/> handles explicit metrics and aliases;
    /// this adds task-awareness so directory-based tasks — which init leaves as "auto" — still
    /// engage the gate instead of silently skipping it (BUG-46). Returns null when neither the
    /// requested metric nor the task default is present (e.g. object detection's mAP has no
    /// universal threshold).
    /// </summary>
    public static string? ResolveCanonicalMetricKey(string metricName, string? task, IEnumerable<string> availableKeys)
    {
        var keys = availableKeys as ICollection<string> ?? availableKeys.ToList();

        var resolved = ResolveMetricKey(metricName, keys);
        if (resolved != null)
        {
            return resolved;
        }

        // Task→primary-metric mapping lives in the shared TaskMetadata source of truth so the
        // promotion gate evaluates the same metric init writes and AutoML optimizes (TD-06).
        var taskDefault = TaskMetadata.PrimaryMetric(task);
        return taskDefault != null ? ResolveMetricKey(taskDefault, keys) : null;
    }

    /// <summary>Convenience overload resolving against a metrics dictionary's keys.</summary>
    private static string? ResolveMetricKey(string metricName, Dictionary<string, double> metrics)
        => ResolveMetricKey(metricName, metrics.Keys);

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

    /// <summary>
    /// Returns the minimum viable metric threshold for quality gate.
    /// Models scoring below this threshold are not promoted to production.
    /// Returns null for metrics without a universal minimum (e.g., error metrics).
    /// </summary>
    public static double? GetMinimumMetricThreshold(string metricName, int? classCount = null)
    {
        return metricName.ToLowerInvariant() switch
        {
            "r_squared" or "r2" => 0.0,                // Must be better than mean prediction
            "auc" or "area_under_roc_curve" => 0.5,     // Must be better than random
            "accuracy" or "micro_accuracy" => classCount.HasValue && classCount.Value > 1
                ? 1.0 / classCount.Value                 // Must be better than random (1/N)
                : 0.0,
            "macro_accuracy" => classCount.HasValue && classCount.Value > 1
                ? 1.0 / classCount.Value                 // Must be better than random (1/N)
                : 0.0,
            "f1" or "f1_score" => classCount.HasValue && classCount.Value > 1
                ? 1.0 / classCount.Value                 // Must be better than random (1/N)
                : 0.0,
            "macro_f1" => classCount.HasValue && classCount.Value > 1
                ? 1.0 / classCount.Value                 // Must be better than random (1/N)
                : 0.0,
            _ => null                                    // No threshold for unknown/error metrics
        };
    }

    private static bool IsErrorMetric(string metricName)
        => MetricDirection.IsLowerBetter(metricName);

    /// <summary>
    /// Detects degenerate classification models that achieve high accuracy by only
    /// predicting the majority class. Returns true if accuracy > 0.5 but F1 ≈ 0.
    /// </summary>
    public static bool IsClassificationDegenerateModel(Dictionary<string, double> metrics)
    {
        // Check binary: accuracy > 0.5 but f1_score == 0
        if (metrics.TryGetValue("f1_score", out var f1) &&
            metrics.TryGetValue("accuracy", out var acc))
        {
            if (acc > 0.5 && f1 < 0.001)
                return true;
        }

        // Check multiclass: macro_accuracy > random but macro_f1 == 0
        if (metrics.TryGetValue("macro_f1", out var macroF1) &&
            metrics.TryGetValue("macro_accuracy", out var macroAcc))
        {
            if (macroAcc > 0.3 && macroF1 < 0.001)
                return true;
        }

        return false;
    }

    private static string ResolveModelName(string? name)
    {
        return string.IsNullOrWhiteSpace(name)
            ? ConfigDefaults.DefaultModelName
            : name.Trim().ToLowerInvariant();
    }
}
