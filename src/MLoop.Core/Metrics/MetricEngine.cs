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
    private readonly ScriptLoader _scriptLoader;
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
        _scriptLoader = scriptLoader ?? new ScriptLoader();
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
    /// Metrics are executed in alphabetical order.
    /// </summary>
    /// <param name="context">Metric calculation context with predictions and labels</param>
    /// <returns>List of metric results from all executed metrics</returns>
    /// <exception cref="InvalidOperationException">Thrown when metric execution fails critically</exception>
    public async Task<List<MetricResult>> ExecuteMetricsAsync(MetricContext context)
    {
        var results = new List<MetricResult>();
        var metricsPath = GetMetricsDirectory();

        // Zero-overhead check: if directory doesn't exist, return immediately
        if (!Directory.Exists(metricsPath))
        {
            return results;  // No metrics = empty list (< 1ms performance)
        }

        var scriptFiles = Directory.GetFiles(metricsPath, "*.cs", SearchOption.TopDirectoryOnly);

        if (scriptFiles.Length == 0)
        {
            return results;  // No metrics found
        }

        // Execute metrics in alphabetical order
        Array.Sort(scriptFiles, StringComparer.Ordinal);

        foreach (var scriptFile in scriptFiles)
        {
            var scriptName = Path.GetFileName(scriptFile);

            try
            {
                // Load and execute metric
                var metrics = await _scriptLoader.LoadScriptAsync<IMLoopMetric>(scriptFile);

                if (metrics.Count == 0)
                {
                    _logger.Warning($"⚠️  No metric implementations found in {scriptName}");
                    continue;
                }

                var metric = metrics[0];  // Use first implementation
                _logger.Info($"📊 Calculating metric: {metric.Name}");

                var result = await metric.CalculateAsync(context);
                results.Add(result);

                _logger.Info($"   {result}");
            }
            catch (Exception ex)
            {
                // Graceful degradation: Log error and continue with remaining metrics
                _logger.Warning($"⚠️  Metric {scriptName} failed: {ex.Message}");
                _logger.Debug($"Stack trace: {ex.StackTrace}");
            }
        }

        return results;
    }
}
