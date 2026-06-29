using System.Text.Json;
using MLoop.Core.Storage;
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
        var candidateMetrics = await LoadMetricsAsync(modelName, candidateExpId, cancellationToken).ConfigureAwait(false);
        var baselineMetrics = await LoadMetricsAsync(modelName, baselineExpId, cancellationToken).ConfigureAwait(false);

        return CompareMetrics(candidateExpId, baselineExpId, candidateMetrics, baselineMetrics);
    }

    private async Task<Dictionary<string, double>> LoadMetricsAsync(
        string modelName,
        string experimentId,
        CancellationToken cancellationToken)
    {
        var metricsPath = Path.Combine(GetExperimentPath(modelName, experimentId), ExperimentLayout.MetricsFileName);

        if (!File.Exists(metricsPath))
        {
            throw new FileNotFoundException($"Metrics not found for experiment '{experimentId}'", metricsPath);
        }

        var json = await File.ReadAllTextAsync(metricsPath, cancellationToken).ConfigureAwait(false);
        var metrics = JsonSerializer.Deserialize<Dictionary<string, double>>(json, JsonOptions);

        return metrics ?? new Dictionary<string, double>();
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
        => MLoop.Core.Evaluation.MetricDirection.IsLowerBetter(metricName);

    private string GetModelPath(string modelName)
    {
        return Path.Combine(_projectRoot, "models", SanitizeModelName(modelName));
    }

    private string GetExperimentsPath(string modelName)
    {
        // Experiments live under "staging" (the layout ExperimentStore writes), not "experiments" —
        // the old path never existed on a real project, so model comparison (and the REST /compare
        // endpoint) failed to find any metrics (F-33).
        return Path.Combine(GetModelPath(modelName), ExperimentLayout.StagingDirectory);
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
