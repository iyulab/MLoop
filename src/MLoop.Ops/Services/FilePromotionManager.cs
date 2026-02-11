using System.Text.Json;
using MLoop.Ops.Interfaces;

namespace MLoop.Ops.Services;

/// <summary>
/// Filesystem-based promotion manager.
/// Manages model promotion lifecycle with backup, rollback, and history tracking.
/// </summary>
public sealed class FilePromotionManager : IPromotionManager
{
    private readonly string _projectRoot;
    private readonly IModelComparer _comparer;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true
    };

    public FilePromotionManager(string projectRoot, IModelComparer comparer)
    {
        _projectRoot = projectRoot ?? throw new ArgumentNullException(nameof(projectRoot));
        _comparer = comparer ?? throw new ArgumentNullException(nameof(comparer));
    }

    public async Task<PromotionDecision> EvaluatePromotionAsync(
        string modelName,
        string candidateExpId,
        PromotionPolicy policy,
        CancellationToken cancellationToken = default)
    {
        var checksPassed = new List<string>();
        var checksFailed = new List<string>();

        // Check 1: Candidate experiment exists and has metrics
        var candidateMetricsPath = GetMetricsPath(modelName, candidateExpId);
        if (!File.Exists(candidateMetricsPath))
        {
            checksFailed.Add($"Candidate {candidateExpId} has no metrics file");
            return new PromotionDecision(false, "Candidate has no metrics", checksPassed, checksFailed);
        }
        checksPassed.Add("Candidate has metrics");

        // Check 2: Compare with production if required
        if (policy.RequireComparisonWithProduction)
        {
            var productionExpId = await GetProductionExperimentIdAsync(modelName, cancellationToken);

            if (productionExpId != null)
            {
                var comparison = await _comparer.CompareAsync(
                    modelName, candidateExpId, productionExpId, cancellationToken);

                if (!comparison.CandidateIsBetter)
                {
                    checksFailed.Add($"Candidate is not better than production ({productionExpId})");
                    return new PromotionDecision(false, comparison.Recommendation, checksPassed, checksFailed);
                }

                if (comparison.Improvement < policy.MinimumImprovement)
                {
                    checksFailed.Add($"Improvement {comparison.Improvement:F2}% below minimum {policy.MinimumImprovement:F2}%");
                    return new PromotionDecision(false,
                        $"Insufficient improvement: {comparison.Improvement:F2}% (minimum: {policy.MinimumImprovement:F2}%)",
                        checksPassed, checksFailed);
                }

                checksPassed.Add($"Better than production by {comparison.Improvement:F2}%");
            }
            else
            {
                checksPassed.Add("No existing production model - first promotion");
            }
        }

        // Check 3: Required metrics present
        if (policy.RequiredMetrics is { Count: > 0 })
        {
            var metrics = await LoadMetricsAsync(modelName, candidateExpId, cancellationToken);
            foreach (var required in policy.RequiredMetrics)
            {
                if (metrics.ContainsKey(required))
                {
                    checksPassed.Add($"Required metric '{required}' present");
                }
                else
                {
                    checksFailed.Add($"Required metric '{required}' missing");
                }
            }

            if (checksFailed.Count > 0)
            {
                return new PromotionDecision(false, "Missing required metrics", checksPassed, checksFailed);
            }
        }

        return new PromotionDecision(true,
            $"All {checksPassed.Count} checks passed for {candidateExpId}",
            checksPassed, checksFailed);
    }

    public async Task<PromotionOutcome> PromoteAsync(
        string modelName,
        string experimentId,
        bool createBackup = true,
        CancellationToken cancellationToken = default)
    {
        var modelPath = GetModelPath(modelName);
        var productionPath = Path.Combine(modelPath, "production");
        string? previousExpId = null;
        string? backupPath = null;

        // Get current production experiment ID
        previousExpId = await GetProductionExperimentIdAsync(modelName, cancellationToken);

        // Create backup of current production
        if (createBackup && Directory.Exists(productionPath) && previousExpId != null)
        {
            backupPath = Path.Combine(modelPath, "backups", $"{previousExpId}-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}");
            Directory.CreateDirectory(Path.GetDirectoryName(backupPath)!);
            CopyDirectory(productionPath, backupPath);
        }

        // Copy experiment to production
        var experimentPath = GetExperimentPath(modelName, experimentId);
        if (!Directory.Exists(experimentPath))
        {
            return new PromotionOutcome(false, modelName, experimentId, previousExpId, null, DateTimeOffset.UtcNow);
        }

        if (Directory.Exists(productionPath))
        {
            Directory.Delete(productionPath, recursive: true);
        }

        CopyDirectory(experimentPath, productionPath);

        // Update model registry
        await UpdateRegistryAsync(modelName, experimentId, productionPath, cancellationToken);

        // Record promotion history
        await RecordHistoryAsync(modelName, experimentId, previousExpId, "promote",
            $"Promoted {experimentId} (previous: {previousExpId ?? "none"})", cancellationToken);

        return new PromotionOutcome(true, modelName, experimentId, previousExpId, backupPath, DateTimeOffset.UtcNow);
    }

    public async Task<RollbackOutcome> RollbackAsync(
        string modelName,
        string? targetExpId = null,
        CancellationToken cancellationToken = default)
    {
        var currentExpId = await GetProductionExperimentIdAsync(modelName, cancellationToken);

        if (currentExpId == null)
        {
            return new RollbackOutcome(false, modelName, "", null, DateTimeOffset.UtcNow);
        }

        // If no target specified, find the most recent backup
        if (targetExpId == null)
        {
            var history = await GetHistoryAsync(modelName, 10, cancellationToken);
            var previousPromotion = history
                .Where(h => h.Action == "promote" && h.ExperimentId != currentExpId)
                .OrderByDescending(h => h.Timestamp)
                .FirstOrDefault();

            if (previousPromotion == null)
            {
                return new RollbackOutcome(false, modelName, "", currentExpId, DateTimeOffset.UtcNow);
            }

            targetExpId = previousPromotion.ExperimentId;
        }

        // Promote the target experiment (rollback is essentially re-promoting an older experiment)
        var modelPath = GetModelPath(modelName);
        var productionPath = Path.Combine(modelPath, "production");
        var targetPath = GetExperimentPath(modelName, targetExpId);

        if (!Directory.Exists(targetPath))
        {
            return new RollbackOutcome(false, modelName, targetExpId, currentExpId, DateTimeOffset.UtcNow);
        }

        if (Directory.Exists(productionPath))
        {
            Directory.Delete(productionPath, recursive: true);
        }

        CopyDirectory(targetPath, productionPath);
        await UpdateRegistryAsync(modelName, targetExpId, productionPath, cancellationToken);

        await RecordHistoryAsync(modelName, targetExpId, currentExpId, "rollback",
            $"Rolled back from {currentExpId} to {targetExpId}", cancellationToken);

        return new RollbackOutcome(true, modelName, targetExpId, currentExpId, DateTimeOffset.UtcNow);
    }

    public async Task<IReadOnlyList<PromotionRecord>> GetHistoryAsync(
        string modelName,
        int limit = 10,
        CancellationToken cancellationToken = default)
    {
        var historyPath = GetHistoryPath(modelName);

        if (!File.Exists(historyPath))
        {
            return Array.Empty<PromotionRecord>();
        }

        var json = await File.ReadAllTextAsync(historyPath, cancellationToken);
        var records = JsonSerializer.Deserialize<List<PromotionRecord>>(json, JsonOptions)
            ?? new List<PromotionRecord>();

        return records
            .OrderByDescending(r => r.Timestamp)
            .Take(limit)
            .ToList();
    }

    private async Task RecordHistoryAsync(
        string modelName,
        string experimentId,
        string? previousExpId,
        string action,
        string reason,
        CancellationToken cancellationToken)
    {
        var historyPath = GetHistoryPath(modelName);
        Directory.CreateDirectory(Path.GetDirectoryName(historyPath)!);

        var records = new List<PromotionRecord>();
        if (File.Exists(historyPath))
        {
            var json = await File.ReadAllTextAsync(historyPath, cancellationToken);
            records = JsonSerializer.Deserialize<List<PromotionRecord>>(json, JsonOptions)
                ?? new List<PromotionRecord>();
        }

        records.Add(new PromotionRecord(modelName, experimentId, previousExpId, action, reason, DateTimeOffset.UtcNow));

        await File.WriteAllTextAsync(historyPath,
            JsonSerializer.Serialize(records, JsonOptions), cancellationToken);
    }

    private async Task UpdateRegistryAsync(
        string modelName,
        string experimentId,
        string productionPath,
        CancellationToken cancellationToken)
    {
        var registryPath = Path.Combine(GetModelPath(modelName), "model-registry.json");
        Directory.CreateDirectory(Path.GetDirectoryName(registryPath)!);

        var registry = new Dictionary<string, object>
        {
            ["production"] = new
            {
                experimentId,
                modelPath = productionPath,
                promotedAt = DateTimeOffset.UtcNow
            }
        };

        await File.WriteAllTextAsync(registryPath,
            JsonSerializer.Serialize(registry, JsonOptions), cancellationToken);
    }

    private async Task<Dictionary<string, double>> LoadMetricsAsync(
        string modelName,
        string experimentId,
        CancellationToken cancellationToken)
    {
        var metricsPath = GetMetricsPath(modelName, experimentId);
        if (!File.Exists(metricsPath))
        {
            return new Dictionary<string, double>();
        }

        var json = await File.ReadAllTextAsync(metricsPath, cancellationToken);
        return JsonSerializer.Deserialize<Dictionary<string, double>>(json, JsonOptions)
            ?? new Dictionary<string, double>();
    }

    private async Task<string?> GetProductionExperimentIdAsync(
        string modelName,
        CancellationToken cancellationToken)
    {
        var registryPath = Path.Combine(GetModelPath(modelName), "model-registry.json");
        if (!File.Exists(registryPath))
        {
            return null;
        }

        var json = await File.ReadAllTextAsync(registryPath, cancellationToken);
        var registry = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json, JsonOptions);

        if (registry == null || !registry.TryGetValue("production", out var productionEntry))
        {
            return null;
        }

        if (productionEntry.ValueKind == JsonValueKind.Object &&
            productionEntry.TryGetProperty("experimentId", out var expIdElement))
        {
            return expIdElement.GetString();
        }

        return null;
    }

    private static void CopyDirectory(string source, string destination)
    {
        Directory.CreateDirectory(destination);

        foreach (var file in Directory.GetFiles(source))
        {
            File.Copy(file, Path.Combine(destination, Path.GetFileName(file)), overwrite: true);
        }

        foreach (var dir in Directory.GetDirectories(source))
        {
            CopyDirectory(dir, Path.Combine(destination, Path.GetFileName(dir)));
        }
    }

    private string GetModelPath(string modelName)
        => Path.Combine(_projectRoot, "models", modelName.ToLowerInvariant());

    private string GetExperimentPath(string modelName, string experimentId)
        => Path.Combine(GetModelPath(modelName), "experiments", experimentId);

    private string GetMetricsPath(string modelName, string experimentId)
        => Path.Combine(GetExperimentPath(modelName, experimentId), "metrics.json");

    private string GetHistoryPath(string modelName)
        => Path.Combine(GetModelPath(modelName), "promotion-history.json");
}
