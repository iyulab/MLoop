using System.Globalization;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.ML;
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
    // Layout names delegate to the single ExperimentLayout authority so this writer/reader cannot drift
    // from the layout constants.
    private const string ModelsDirectory = ExperimentLayout.ModelsDirectory;
    private const string ProductionDirectory = ExperimentLayout.ProductionDirectory;
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
    public Task PromoteAsync(
        string modelName,
        string experimentId,
        CancellationToken cancellationToken = default)
        => PromoteAsync(modelName, experimentId, MLoop.Ops.Interfaces.PromotionActions.Promote, null, cancellationToken);

    /// <inheritdoc />
    public async Task PromoteAsync(
        string modelName,
        string experimentId,
        string action,
        string? gatePolicyRefusal = null,
        CancellationToken cancellationToken = default)
    {
        var resolvedName = ResolveModelName(modelName);
        var previous = (await ProductionMetadata.ReadAsync(GetProductionPath(resolvedName), cancellationToken))?.ExperimentId;

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

        // ② regression wave (heteroscedastic): promote the σ(x) model alongside when present, so the
        // production predict path can widen the conformal band per row. Non-regression / homoscedastic
        // experiments have no such file — the production model then serves the constant-width band.
        var sourceResidualPath = _fileSystem.CombinePath(experimentPath, ExperimentLayout.ResidualModelFileName);
        if (_fileSystem.FileExists(sourceResidualPath))
        {
            var targetResidualPath = _fileSystem.CombinePath(targetPath, ExperimentLayout.ResidualModelFileName);
            await _fileSystem.CopyFileAsync(sourceResidualPath, targetResidualPath, overwrite: true, cancellationToken);
        }

        // The drift reference travels with the model it describes. A production slot promoted from an
        // experiment without one must not keep the previous model's.
        var sourceDriftPath = _fileSystem.CombinePath(experimentPath, ExperimentLayout.DriftProfileFileName);
        var targetDriftPath = _fileSystem.CombinePath(targetPath, ExperimentLayout.DriftProfileFileName);
        if (_fileSystem.FileExists(sourceDriftPath))
            await _fileSystem.CopyFileAsync(sourceDriftPath, targetDriftPath, overwrite: true, cancellationToken);
        else if (_fileSystem.FileExists(targetDriftPath))
            File.Delete(targetDriftPath);

        // So does the decision threshold: production answers at the threshold chosen for this model, or at
        // the trainer's own — never at the previous model's.
        // The calibration too, for the same reason.
        // An image-anomaly model is its bank and the record that names its vector space: they move together.
        foreach (var layer in new[]
        {
            ExperimentLayout.DecisionThresholdFileName, ExperimentLayout.CalibrationFileName, ExperimentLayout.TextEncodersFileName,
            ExperimentLayout.ImageAnomalyFileName, ExperimentLayout.PatchBankFileName,
        })
        {
            var sourceLayerPath = _fileSystem.CombinePath(experimentPath, layer);
            var targetLayerPath = _fileSystem.CombinePath(targetPath, layer);
            if (_fileSystem.FileExists(sourceLayerPath))
                await _fileSystem.CopyFileAsync(sourceLayerPath, targetLayerPath, overwrite: true, cancellationToken);
            else if (_fileSystem.FileExists(targetLayerPath))
                File.Delete(targetLayerPath);
        }

        // Load experiment metadata
        var experiment = await _experimentStore.LoadAsync(resolvedName, experimentId, cancellationToken);

        // Save production metadata — the single authority for the production pointer + its snapshot
        // (experimentId/metrics/task/trainer/label), read back via ProductionMetadata. This is the only
        // production-state artifact written; the parallel registry.json 'production' entry that used to
        // duplicate it was removed (ProductionMetadata remarks).
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

        await new MLoop.Ops.Services.FilePromotionManager(_projectRoot)
            .RecordAsync(HistoryRecord(resolvedName, experiment, previous, action, gatePolicyRefusal), cancellationToken);
    }

    /// <summary>
    /// The history entry for <paramref name="experiment"/> taking production from <paramref name="previous"/>:
    /// the gate's outcome is judged here, at the moment of the change, by the same rule <c>promote</c> applies.
    /// A gate refusal on a promotion that went ahead means it was forced; a rollback is not gated.
    /// </summary>
    internal static MLoop.Ops.Interfaces.PromotionRecord HistoryRecord(
        string modelName, ExperimentData experiment, string? previous, string action, string? gatePolicyRefusal = null)
    {
        var metric = experiment.Metrics is { Count: > 0 } metrics
            ? MetricPolicy.ResolveCanonicalMetricKey(experiment.Config?.Metric ?? MetricNames.Auto, experiment.Task, metrics.Keys)
            : null;
        double? value = metric is not null && experiment.Metrics!.TryGetValue(metric, out var v) ? v : null;

        string gate;
        string? detail;
        if (action == MLoop.Ops.Interfaces.PromotionActions.Rollback)
        {
            (gate, detail) = (MLoop.Ops.Interfaces.PromotionGate.NotApplied, null);
        }
        else if ((QualityGateRefusal(experiment, experiment.Config?.Metric) ?? gatePolicyRefusal) is { } refusal)
        {
            (gate, detail) = (MLoop.Ops.Interfaces.PromotionGate.Forced, refusal);
        }
        else
        {
            (gate, detail) = (MLoop.Ops.Interfaces.PromotionGate.Passed, QualityGateWarning(experiment));
        }

        return new MLoop.Ops.Interfaces.PromotionRecord(
            modelName, experiment.ExperimentId, previous, action,
            $"{action}: {experiment.ExperimentId} (previous: {previous ?? "none"})",
            DateTimeOffset.UtcNow, gate, detail, metric, value);
    }

    /// <inheritdoc />
    public async Task<ModelInfo?> GetProductionAsync(
        string modelName,
        CancellationToken cancellationToken = default)
    {
        var resolvedName = ResolveModelName(modelName);
        var metadata = await ProductionMetadata.ReadAsync(GetProductionPath(resolvedName), cancellationToken);

        if (metadata?.ExperimentId is null or "")
        {
            return null;
        }

        return new ModelInfo
        {
            ModelName = resolvedName,
            ExperimentId = metadata.ExperimentId,
            PromotedAt = metadata.PromotedAt ?? DateTime.UtcNow,
            Metrics = metadata.Metrics,
            ModelPath = GetProductionPath(resolvedName),
            Task = metadata.Task,
            BestTrainer = metadata.BestTrainer
        };
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
        // The project's own rules last: they cost an evaluation, which a model the fixed gate or the
        // production comparison already holds back does not need.
        => await ShouldPromoteOnMetricsAsync(modelName, experimentId, primaryMetric, cancellationToken)
           && await GatePolicyRefusalAsync(modelName, experimentId, cancellationToken) is null;

    /// <inheritdoc />
    public async Task<string?> GatePolicyRefusalAsync(string modelName, string experimentId, CancellationToken cancellationToken = default)
    {
        var resolvedName = ResolveModelName(modelName);
        var config = await new Configuration.ConfigLoader(_fileSystem, _projectDiscovery).LoadUserConfigAsync(cancellationToken);
        if (!config.Models.TryGetValue(resolvedName, out var definition) || definition.Gate is not { HasRules: true } policy)
            return null;
        var experiment = await _experimentStore.LoadAsync(resolvedName, experimentId, cancellationToken);
        return await GatePolicyCheck.RefusalAsync(policy, _projectRoot, resolvedName, experimentId, experiment, _experimentStore, this, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<string?> KeptInStagingReasonAsync(
        string modelName,
        string experimentId,
        string primaryMetric,
        CancellationToken cancellationToken = default)
    {
        var resolvedName = ResolveModelName(modelName);
        var experiment = await _experimentStore.LoadAsync(resolvedName, experimentId, cancellationToken);
        if (QualityGateRefusal(experiment, primaryMetric) is { } gateRefusal)
            return $"the quality gate refuses it: {gateRefusal}";

        var production = await GetProductionAsync(resolvedName, cancellationToken);
        if (await IncomparableWithProductionAsync(resolvedName, experiment, production, cancellationToken) is { } incomparable)
            return $"its metrics cannot be compared with production's: {incomparable}";

        if (await ShouldPromoteOnMetricsAsync(resolvedName, experimentId, primaryMetric, cancellationToken))
            return await GatePolicyRefusalAsync(resolvedName, experimentId, cancellationToken) is { } policyRefusal
                ? $"the gate policy refuses it: {policyRefusal}"
                : null;

        var metricKey = MetricPolicy.ResolveCanonicalMetricKey(primaryMetric, experiment.Task, experiment.Metrics!.Keys);
        if (metricKey == null || production?.Metrics is null)
            return "there is no metric to judge it against production by — promote it explicitly";
        return experiment.Metrics.TryGetValue(metricKey, out var candidate) && production.Metrics.TryGetValue(metricKey, out var current)
            ? string.Create(CultureInfo.InvariantCulture,
                $"it does not beat production ({production.ExperimentId}) on {metricKey}: {candidate:0.####} vs {current:0.####}")
            : $"it does not beat production ({production.ExperimentId}) on {metricKey}";
    }

    /// <summary>
    /// Whether the metrics alone — the fixed gate and the comparison with production — would promote
    /// <paramref name="experimentId"/>; <see cref="ShouldPromoteAsync"/> adds the gate policy on top.
    /// </summary>
    public async Task<bool> ShouldPromoteOnMetricsAsync(
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

        if (QualityGateRefusal(experiment, primaryMetric) is not null)
        {
            return false;
        }

        var metricKey = MetricPolicy.ResolveCanonicalMetricKey(primaryMetric, experiment.Task, experiment.Metrics.Keys);

        // Get current production model
        var currentProduction = await GetProductionAsync(resolvedName, cancellationToken);

        if (currentProduction?.Metrics == null)
        {
            return true; // No production model yet (or it has no metrics), promote first model
        }

        // Metrics weighted by different columns (or one weighted, one not) describe different
        // distributions; "better" between them is not a judgement either. An explicit promote decides.
        if (await IncomparableWithProductionAsync(resolvedName, experiment, currentProduction, cancellationToken) is not null)
        {
            return false;
        }

        // No metric to judge by means no judgement was made: neither the quality gate above nor a
        // comparison can run. Replacing a production model on that basis is a decision nobody took
        // — it once let a model scoring below zero replace one at 0.73 because the metric name was
        // the unresolved "auto". The first model still goes in (there is nothing to protect); after
        // that, promotion without a metric is left to an explicit `mloop promote`.
        if (metricKey == null)
        {
            return false;
        }

        // Production predates this metric (an older experiment): the new model passed the gate on
        // it, and there is nothing to compare against, so it goes in.
        if (!experiment.Metrics.TryGetValue(metricKey, out var newMetricValue) ||
            !currentProduction.Metrics.TryGetValue(metricKey, out var currentMetricValue))
        {
            return true;
        }

        // Promote if new model is better
        // For most metrics (accuracy, r_squared, auc, f1), higher is better
        // For error metrics (mae, rmse, mse), lower is better. Use the resolved canonical key
        // so the direction matches the value actually compared (metricKey is non-null here).
        return MetricPolicy.IsErrorMetric(metricKey)
            ? newMetricValue < currentMetricValue  // Lower error is better
            : newMetricValue > currentMetricValue; // Higher score is better
    }

    /// <summary>
    /// Why the quality gate refuses <paramref name="experiment"/> as a production model, or
    /// <c>null</c> when it passes. The one statement of the gate: training's automatic promotion and
    /// an explicit <c>mloop promote</c> both read it. Before it was shared, only training applied it,
    /// so a model that predicted a single class (accuracy 0.99 on 0.6% positives, F1 0) went to
    /// production through an explicit promote without a word.
    /// </summary>
    public static string? QualityGateRefusal(ExperimentData experiment, string? primaryMetric)
    {
        if (experiment.Metrics is not { Count: > 0 } metrics)
        {
            return "it has no recorded metrics";
        }

        // User-facing metric aliases (e.g. "f1") differ from the canonical keys the
        // EvaluationEngine stores (e.g. "f1_score"). Resolve to the stored key so the
        // quality gate and production comparison operate on the right value instead of
        // silently failing the lookup (which blocked first-model auto-promotion).
        // When the project metric is the deferred "auto" (init leaves image/OD tasks as auto),
        // fall back to the task's canonical metric so the gate still engages instead of being
        // silently skipped.
        var metricKey = MetricPolicy.ResolveCanonicalMetricKey(primaryMetric ?? MetricNames.Auto, experiment.Task, metrics.Keys);

        // Extract the label's shape from the schema for the dynamic thresholds: how many classes
        // there are, and how strong a model that always answers with the most common one would be.
        int? classCount = null;
        double? majorityClassRatio = null;
        if (experiment.Config?.InputSchema?.Columns != null && experiment.Config.LabelColumn != null)
        {
            var labelSchema = experiment.Config.InputSchema.Columns
                .FirstOrDefault(s => s.Name.Equals(experiment.Config.LabelColumn, StringComparison.OrdinalIgnoreCase));
            if (labelSchema?.UniqueValueCount > 0)
            {
                classCount = labelSchema.UniqueValueCount;
            }
            // Absent on experiments trained before the ratio was recorded, and on numeric labels —
            // left null so the gate falls back to 1/N rather than to a fabricated 0.
            majorityClassRatio = labelSchema?.MajorityClassRatio;
        }

        // An undefined-metric sentinel (MetricSanitizer's direction-aware worst-case for a NaN/∞
        // measurement — e.g. an all-NaN-scoring degenerate model whose evaluator saw zero usable
        // rows) must never pass the gate, per the sanitizer's own contract. Checked explicitly
        // because the floor-threshold check below only gates higher-is-better metrics: a regression
        // optimized on rmse/mae/mse had NO floor at all, so rmse = MaxValue still auto-promoted as
        // the first production model.
        if (metricKey != null && MetricPolicy.IsUndefinedMetricSentinel(metricKey, metrics[metricKey]))
        {
            return $"its {metricKey} could not be measured (the evaluator produced no usable value)";
        }

        // Check minimum metric threshold (quality gate) — only when the metric is present.
        // Threshold and error-direction are computed from the RESOLVED canonical key, not the
        // raw user/project metric: "auto" and aliases ("f1", "r2") have no entry in the
        // threshold table, so using the raw value silently skipped the gate — the root of both.
        if (metricKey != null && !MetricPolicy.IsErrorMetric(metricKey))
        {
            // <=, not <: every threshold here is a "must be better than the trivial baseline"
            // floor (R² > 0 beats mean-prediction, AUC > 0.5 beats random, accuracy beats the
            // majority-class predictor, …) — sitting exactly on the floor means the model is
            // exactly as good as the trivial baseline, not better than it. A strict `<` let a
            // model that scored precisely at the floor (e.g. R² == 0.0 on a degenerate test
            // split) through as if it had passed.
            var minThreshold = MetricPolicy.GetMinimumMetricThreshold(metricKey, classCount, majorityClassRatio);
            if (minThreshold.HasValue && metrics[metricKey] <= minThreshold.Value)
            {
                return string.Create(CultureInfo.InvariantCulture,
                    $"its {metricKey} {metrics[metricKey]:F4} is at or below {minThreshold.Value:F4}, what a model that learned nothing scores");
            }
        }

        // An error metric has no floor of its own; R² at or below 0 says the model is no better than the mean.
        if (metricKey != null && MetricPolicy.NoBetterThanTheMean(metricKey, metrics) is { } rSquared)
        {
            return string.Create(CultureInfo.InvariantCulture,
                $"its r_squared {rSquared:F4} is no better than predicting the mean");
        }

        // Degenerate model detection: high accuracy but the model only ever predicts one
        // class (zero F1 on the positive class, or zero recall on the negative class)
        if (MetricPolicy.IsClassificationDegenerateModel(metrics))
        {
            return "it predicts only one class (its accuracy reflects the class balance, not learning)";
        }

        return null;

    }

    /// <summary>
    /// Why <paramref name="experiment"/>'s metrics cannot be compared with production's, or
    /// <c>null</c> when they can: the two were measured with rows weighted by different columns, or
    /// one weighted and the other not — the same metric names over different distributions.
    /// </summary>
    public async Task<string?> IncomparableWithProductionAsync(
        string modelName, ExperimentData experiment, ModelInfo? production, CancellationToken cancellationToken = default)
    {
        if (production is null || production.ExperimentId == experiment.ExperimentId)
            return null;

        string? productionWeight;
        try
        {
            productionWeight = (await _experimentStore.LoadAsync(modelName, production.ExperimentId, cancellationToken)).Config?.WeightColumn;
        }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException or InvalidOperationException)
        {
            return null; // production's experiment is gone; its recorded metrics are all there is to compare
        }

        var weighting = Weighting.Of(experiment.Config?.WeightColumn);
        var productionWeighting = Weighting.Of(productionWeight);
        if (weighting == productionWeighting)
            return null;

        return $"its metrics are {weighting} and production's ({production.ExperimentId}) are {productionWeighting} — " +
               "the same names over different distributions";
    }

    /// <summary>
    /// What the quality gate lets through but a reader of the model should still be told, or
    /// <c>null</c>. Today one shape: a binary model that ranks above chance yet answers one class at
    /// its default threshold (<see cref="MetricPolicy.SingleClassAtThreshold"/>). Every surface that
    /// shows <see cref="QualityGateRefusal"/> shows this beside it.
    /// </summary>
    public static string? QualityGateWarning(ExperimentData experiment)
        => experiment.Metrics is { Count: > 0 } metrics ? MetricPolicy.SingleClassAtThreshold(metrics) : null;

    /// <inheritdoc />
    public async Task<bool> AutoPromoteAsync(
        string modelName,
        string experimentId,
        string primaryMetric,
        CancellationToken cancellationToken = default)
    {
        if (await ShouldPromoteAsync(modelName, experimentId, primaryMetric, cancellationToken))
        {
            await PromoteAsync(modelName, experimentId, MLoop.Ops.Interfaces.PromotionActions.AutoPromote, null, cancellationToken);
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
