using System.Text.Json;
using MLoop.Ops.Interfaces;

namespace MLoop.Ops.Services;

/// <summary>
/// Filesystem-based model comparison service.
/// Reads experiment metrics directly from the filesystem structure:
/// models/{modelName}/experiments/{expId}/metrics.json
/// </summary>
public sealed class FileModelComparer : IModelComparer
{
    private readonly string _projectRoot;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    /// <summary>
    /// Creates a new FileModelComparer for the specified project root.
    /// </summary>
    /// <param name="projectRoot">Absolute path to the MLoop project root</param>
    public FileModelComparer(string projectRoot)
    {
        _projectRoot = projectRoot ?? throw new ArgumentNullException(nameof(projectRoot));
    }

    /// <inheritdoc/>
    public async Task<ComparisonResult> CompareAsync(
        string modelName,
        string candidateExpId,
        string baselineExpId,
        CancellationToken cancellationToken = default)
    {
        var candidateMetrics = await LoadMetricsAsync(modelName, candidateExpId, cancellationToken);
        var baselineMetrics = await LoadMetricsAsync(modelName, baselineExpId, cancellationToken);

        return CompareMetrics(candidateExpId, baselineExpId, candidateMetrics, baselineMetrics);
    }

    /// <inheritdoc/>
    public async Task<ComparisonResult> CompareWithProductionAsync(
        string modelName,
        string candidateExpId,
        CancellationToken cancellationToken = default)
    {
        var productionExpId = await GetProductionExperimentIdAsync(modelName, cancellationToken);

        if (string.IsNullOrEmpty(productionExpId))
        {
            // No production model - candidate wins by default
            var candidateMetrics = await LoadMetricsAsync(modelName, candidateExpId, cancellationToken);
            var primaryMetric = candidateMetrics.Keys.FirstOrDefault() ?? "unknown";
            var candidateScore = candidateMetrics.Values.FirstOrDefault();

            return new ComparisonResult(
                CandidateExpId: candidateExpId,
                BaselineExpId: "(none)",
                CandidateIsBetter: true,
                CandidateScore: candidateScore,
                BaselineScore: 0.0,
                Improvement: double.PositiveInfinity,
                MetricDetails: candidateMetrics.ToDictionary(
                    kv => kv.Key,
                    kv => new MetricComparison(kv.Key, kv.Value, 0.0, kv.Value, true)
                ),
                Recommendation: $"Promote {candidateExpId} - no existing production model"
            );
        }

        return await CompareAsync(modelName, candidateExpId, productionExpId, cancellationToken);
    }

    /// <inheritdoc/>
    public async Task<string?> FindBestExperimentAsync(
        string modelName,
        ComparisonCriteria criteria,
        CancellationToken cancellationToken = default)
    {
        var experimentsPath = GetExperimentsPath(modelName);

        if (!Directory.Exists(experimentsPath))
        {
            return null;
        }

        var experimentDirs = Directory.GetDirectories(experimentsPath);
        string? bestExpId = null;
        double bestScore = double.NegativeInfinity;

        foreach (var expDir in experimentDirs)
        {
            var expId = Path.GetFileName(expDir);

            try
            {
                var metrics = await LoadMetricsAsync(modelName, expId, cancellationToken);

                if (metrics.TryGetValue(criteria.PrimaryMetric, out var score))
                {
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestExpId = expId;
                    }
                }
            }
            catch
            {
                // Skip experiments with missing or invalid metrics
                continue;
            }
        }

        // Apply minimum improvement threshold if we have a current best
        if (bestExpId != null && criteria.MinimumImprovement > 0)
        {
            var productionExpId = await GetProductionExperimentIdAsync(modelName, cancellationToken);

            if (!string.IsNullOrEmpty(productionExpId))
            {
                var productionMetrics = await LoadMetricsAsync(modelName, productionExpId, cancellationToken);

                if (productionMetrics.TryGetValue(criteria.PrimaryMetric, out var productionScore))
                {
                    var improvement = (bestScore - productionScore) / Math.Abs(productionScore);

                    if (improvement < criteria.MinimumImprovement)
                    {
                        return null; // Best candidate doesn't meet minimum improvement threshold
                    }
                }
            }
        }

        return bestExpId;
    }

    private async Task<Dictionary<string, double>> LoadMetricsAsync(
        string modelName,
        string experimentId,
        CancellationToken cancellationToken)
    {
        var metricsPath = Path.Combine(GetExperimentPath(modelName, experimentId), "metrics.json");

        if (!File.Exists(metricsPath))
        {
            throw new FileNotFoundException($"Metrics not found for experiment '{experimentId}'", metricsPath);
        }

        var json = await File.ReadAllTextAsync(metricsPath, cancellationToken);
        var metrics = JsonSerializer.Deserialize<Dictionary<string, double>>(json, JsonOptions);

        return metrics ?? new Dictionary<string, double>();
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

    private ComparisonResult CompareMetrics(
        string candidateExpId,
        string baselineExpId,
        Dictionary<string, double> candidateMetrics,
        Dictionary<string, double> baselineMetrics)
    {
        var metricDetails = new Dictionary<string, MetricComparison>();
        var allMetrics = candidateMetrics.Keys.Union(baselineMetrics.Keys).ToList();

        double candidateTotalScore = 0;
        double baselineTotalScore = 0;
        int betterCount = 0;
        int totalCount = 0;

        foreach (var metric in allMetrics)
        {
            var candidateValue = candidateMetrics.GetValueOrDefault(metric, 0.0);
            var baselineValue = baselineMetrics.GetValueOrDefault(metric, 0.0);
            var difference = candidateValue - baselineValue;

            // Higher is better for most metrics (accuracy, f1, etc.)
            // Lower is better for error metrics - handle by metric name
            var isLowerBetter = IsLowerBetterMetric(metric);
            var isBetter = isLowerBetter
                ? candidateValue < baselineValue
                : candidateValue > baselineValue;

            metricDetails[metric] = new MetricComparison(
                metric,
                candidateValue,
                baselineValue,
                difference,
                isBetter
            );

            candidateTotalScore += candidateValue;
            baselineTotalScore += baselineValue;

            if (isBetter) betterCount++;
            totalCount++;
        }

        var candidateScore = totalCount > 0 ? candidateTotalScore / totalCount : 0;
        var baselineScore = totalCount > 0 ? baselineTotalScore / totalCount : 0;
        var improvement = baselineScore != 0
            ? (candidateScore - baselineScore) / Math.Abs(baselineScore) * 100
            : (candidateScore > 0 ? 100 : 0);

        var candidateIsBetter = betterCount > totalCount / 2;

        var recommendation = candidateIsBetter
            ? $"Promote {candidateExpId} - better on {betterCount}/{totalCount} metrics ({improvement:F1}% improvement)"
            : $"Keep {baselineExpId} - baseline is better on {totalCount - betterCount}/{totalCount} metrics";

        return new ComparisonResult(
            CandidateExpId: candidateExpId,
            BaselineExpId: baselineExpId,
            CandidateIsBetter: candidateIsBetter,
            CandidateScore: candidateScore,
            BaselineScore: baselineScore,
            Improvement: improvement,
            MetricDetails: metricDetails,
            Recommendation: recommendation
        );
    }

    private static bool IsLowerBetterMetric(string metricName)
    {
        var lowerName = metricName.ToLowerInvariant();
        return lowerName.Contains("error")
            || lowerName.Contains("loss")
            || lowerName.Contains("mae")
            || lowerName.Contains("mse")
            || lowerName.Contains("rmse");
    }

    private string GetModelPath(string modelName)
    {
        return Path.Combine(_projectRoot, "models", SanitizeModelName(modelName));
    }

    private string GetExperimentsPath(string modelName)
    {
        return Path.Combine(GetModelPath(modelName), "experiments");
    }

    private string GetExperimentPath(string modelName, string experimentId)
    {
        return Path.Combine(GetExperimentsPath(modelName), experimentId);
    }

    private static string SanitizeModelName(string modelName)
    {
        // Replace invalid path characters
        var invalid = Path.GetInvalidFileNameChars();
        var sanitized = modelName;

        foreach (var c in invalid)
        {
            sanitized = sanitized.Replace(c, '_');
        }

        return sanitized.ToLowerInvariant();
    }
}
