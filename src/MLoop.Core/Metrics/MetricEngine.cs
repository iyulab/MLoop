using MLoop.Core.Scripting;
using MLoop.Extensibility.Metrics;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Metrics;

/// <summary>
/// Executes custom business metric scripts after model evaluation.
/// Discovers and runs metrics from .mloop/scripts/metrics/*.cs
/// </summary>
public class MetricEngine
{
    private readonly string _projectRoot;
    private ScriptLoader? _scriptLoader;
    private readonly ILogger _logger;

    /// <summary>
    /// Initializes a new instance of MetricEngine.
    /// </summary>
    /// <param name="projectRoot">Project root directory</param>
    /// <param name="logger">Logger for user feedback</param>
    /// <param name="scriptLoader">Optional ScriptLoader instance</param>
    public MetricEngine(
        string projectRoot,
        ILogger logger,
        ScriptLoader? scriptLoader = null)
    {
        _projectRoot = projectRoot;
        _logger = logger;
        _scriptLoader = scriptLoader;  // created on first use, as HookEngine does — most projects have no metric scripts
    }

    private ScriptLoader Loader => _scriptLoader ??= new ScriptLoader(ScriptLoader.CacheDirectoryFor(_projectRoot), _logger.Warning);

    /// <summary>
    /// Computes the project's metric scripts over scored held-out rows and adds each result to
    /// <paramref name="metrics"/> under the script's <c>Name</c>. A name the run already reports, or a
    /// value that is not finite, is skipped with a warning — a script cannot replace a measured metric,
    /// and a non-finite value cannot be recorded. Nothing happens without scripts; a task the metric
    /// contract does not cover (classification or regression predictions) is said once and skipped.
    /// </summary>
    /// <returns>The names added.</returns>
    public async Task<IReadOnlyList<string>> AddToAsync(
        Dictionary<string, double> metrics,
        Microsoft.ML.MLContext mlContext,
        Microsoft.ML.IDataView scored,
        string task,
        string labelColumn,
        string? modelName = null,
        string? experimentId = null)
    {
        if (!HasMetrics())
            return [];
        if (CustomMetricInputs.Of(scored, task, labelColumn) is not { } inputs)
        {
            _logger.Warning($"The metric scripts in .mloop/scripts/metrics/ were not computed: they take classification or regression predictions, and this model's task is {task}.");
            return [];
        }

        var results = await ExecuteMetricsAsync(new MetricContext
        {
            MLContext = mlContext,
            Predictions = inputs.Predictions,
            Labels = inputs.Labels,
            TaskType = task,
            ModelName = modelName,
            ExperimentId = experimentId,
            Logger = _logger,
        }).ConfigureAwait(false);

        var added = new List<string>();
        foreach (var result in results)
        {
            if (metrics.ContainsKey(result.Name))
                _logger.Warning($"Metric script result '{result.Name}' was not recorded: the run already reports a metric of that name.");
            else if (!double.IsFinite(result.Value))
                _logger.Warning($"Metric script result '{result.Name}' was not recorded: its value is {result.Value}.");
            else
            {
                metrics[result.Name] = result.Value;
                added.Add(result.Name);
            }
        }
        return added;
    }

    /// <summary>
    /// Gets the directory path for custom metrics.
    /// </summary>
    /// <returns>Full path to metrics directory</returns>
    public string GetMetricsDirectory()
    {
        return Path.Combine(_projectRoot, ".mloop", "scripts", "metrics");
    }

    /// <summary>
    /// Checks if any custom metrics exist.
    /// </summary>
    /// <returns>True if metrics exist, false otherwise</returns>
    public bool HasMetrics()
    {
        var metricsPath = GetMetricsDirectory();

        if (!Directory.Exists(metricsPath))
            return false;

        var scriptFiles = Directory.GetFiles(metricsPath, "*.cs", SearchOption.TopDirectoryOnly);
        return scriptFiles.Length > 0;
    }

    /// <summary>
    /// Initializes metrics directory structure.
    /// </summary>
    public void InitializeDirectories()
    {
        var metricsPath = GetMetricsDirectory();
        Directory.CreateDirectory(metricsPath);
    }

    /// <summary>
    /// Executes all custom metrics and returns results.
    /// Metrics run in the order of their files' names.
    /// </summary>
    /// <param name="context">Metric calculation context with predictions and labels</param>
    /// <returns>List of metric results from all executed metrics</returns>
    /// <exception cref="InvalidOperationException">Thrown when metric execution fails critically</exception>
    public async Task<List<MetricResult>> ExecuteMetricsAsync(MetricContext context)
    {
        var results = new List<MetricResult>();
        if (!HasMetrics())
            return results;  // No metrics = empty list (< 1ms performance)

        // Found the way every script kind is found — every implementation in every file, files in name order.
        var metrics = await new ScriptDiscovery(_projectRoot, Loader, _logger.Debug).DiscoverMetricsAsync().ConfigureAwait(false);
        if (metrics.Count == 0)
        {
            _logger.Warning("No metric implementations found in .mloop/scripts/metrics/");
            return results;
        }

        foreach (var metric in metrics)
        {
            try
            {
                _logger.Info($"📊 Calculating metric: {metric.Name}");
                var result = await metric.CalculateAsync(context).ConfigureAwait(false);
                results.Add(result);
                _logger.Info($"   {result}");
            }
            catch (Exception ex)
            {
                // Graceful degradation: Log error and continue with remaining metrics
                _logger.Warning($"Metric '{metric.Name}' failed: {ex.Message}");
                _logger.Debug($"Stack trace: {ex.StackTrace}");
            }
        }

        return results;
    }
}
