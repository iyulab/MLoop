using System.Text.Json;
using MLoop.Core.Evaluation;
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
        var (task, metric) = await LoadObjectiveAsync(modelName, candidateExpId, cancellationToken).ConfigureAwait(false);

        return CompareMetrics(candidateExpId, baselineExpId, candidateMetrics, baselineMetrics, task, metric);
    }

    // The candidate's task and the metric its run optimized, from its metadata.json — what the comparison is judged on,
    // as promotion judges it. Absent fields leave the task's primary metric.
    private async Task<(string? Task, string? Metric)> LoadObjectiveAsync(string modelName, string experimentId, CancellationToken cancellationToken)
    {
        var path = Path.Combine(GetExperimentPath(modelName, experimentId), ExperimentLayout.MetadataFileName);
        if (!File.Exists(path))
            return (null, null);
        try
        {
            using var doc = JsonDocument.Parse(await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false));
            var root = doc.RootElement;
            string? Text(JsonElement e, string name) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
            return (Text(root, "task"), root.TryGetProperty("config", out var config) ? Text(config, "metric") : null);
        }
        catch (JsonException)
        {
            return (null, null);
        }
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

    // One metric decides — the one the candidate's run optimized (MetricNames.Canonical, as promotion reads it), or the
    // task's primary one — in its direction. Averaging every metric added accuracies to RMSEs, and a majority of metrics
    // let several correlated ones outvote the one the project chose; a metric one side lacked was read as 0, a win for
    // any lower-is-better metric. A metric one side lacks is now left out.
    private static ComparisonResult CompareMetrics(
        string candidateExpId,
        string baselineExpId,
        Dictionary<string, double> candidateMetrics,
        Dictionary<string, double> baselineMetrics,
        string? task,
        string? optimized)
    {
        var shared = candidateMetrics.Keys.Where(baselineMetrics.ContainsKey).ToList();
        var metricDetails = shared.ToDictionary(m => m, m =>
        {
            var (candidate, baseline) = (candidateMetrics[m], baselineMetrics[m]);
            return new MetricComparison(m, candidate, baseline, candidate - baseline,
                IsLowerBetterMetric(m) ? candidate < baseline : candidate > baseline);
        });

        var wanted = MetricNames.Canonical(task, optimized ?? MetricNames.Auto);
        var metric = wanted is not null && wanted != MetricNames.Auto && metricDetails.ContainsKey(wanted)
            ? wanted
            : TaskMetadata.PrimaryMetric(task) is { } primary && metricDetails.ContainsKey(primary) ? primary : null;
        if (metric is null)
        {
            return new ComparisonResult(candidateExpId, baselineExpId, CandidateIsBetter: false, 0, 0, 0, metricDetails,
                $"Keep {baselineExpId} - the two experiments share no metric to judge on" +
                (wanted is null or MetricNames.Auto ? "" : $" ({wanted} is missing from one of them)"));
        }

        var (candidateScore, baselineScore) = (candidateMetrics[metric], baselineMetrics[metric]);
        var lowerIsBetter = IsLowerBetterMetric(metric);
        var candidateIsBetter = metricDetails[metric].IsBetter;
        var gain = lowerIsBetter ? baselineScore - candidateScore : candidateScore - baselineScore;
        var improvement = baselineScore != 0 ? gain / Math.Abs(baselineScore) * 100 : 0;

        var recommendation = candidateIsBetter
            ? $"Promote {candidateExpId} - better on {metric}: {candidateScore:0.####} vs {baselineScore:0.####} ({improvement:+0.0;-0.0}%)"
            : $"Keep {baselineExpId} - {candidateExpId} does not beat it on {metric}: {candidateScore:0.####} vs {baselineScore:0.####}";

        return new ComparisonResult(candidateExpId, baselineExpId, candidateIsBetter, candidateScore, baselineScore, improvement,
            metricDetails, recommendation, metric);
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
        // endpoint) failed to find any metrics.
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
