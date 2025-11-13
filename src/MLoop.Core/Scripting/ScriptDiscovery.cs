using MLoop.Extensibility;

namespace MLoop.Core.Scripting;

/// <summary>
/// Discovers hooks and metrics from filesystem using convention-based patterns.
/// Searches .mloop/scripts/hooks/*.cs and .mloop/scripts/metrics/*.cs
/// </summary>
public class ScriptDiscovery
{
    private readonly ScriptLoader _scriptLoader;
    private readonly string _projectRoot;

    /// <summary>
    /// Initializes a new instance of ScriptDiscovery.
    /// </summary>
    /// <param name="projectRoot">Project root directory (default: current directory)</param>
    /// <param name="scriptLoader">Optional ScriptLoader instance (default: new instance)</param>
    public ScriptDiscovery(string? projectRoot = null, ScriptLoader? scriptLoader = null)
    {
        _projectRoot = projectRoot ?? Directory.GetCurrentDirectory();
        _scriptLoader = scriptLoader ?? new ScriptLoader();
    }

    // NOTE: Phase 1 (Hooks & Metrics) - Disabled for Phase 0 (Preprocessing)
    // TODO: Re-enable when implementing Phase 1

    ///// <summary>
    ///// Discovers all hooks from .mloop/scripts/hooks/*.cs
    ///// </summary>
    ///// <returns>List of discovered hook instances</returns>
    //public async Task<List<IMLoopHook>> DiscoverHooksAsync()
    //{
    //    var hooksPath = Path.Combine(_projectRoot, ".mloop", "scripts", "hooks");
    //    return await DiscoverScriptsAsync<IMLoopHook>(hooksPath, "hooks");
    //}

    ///// <summary>
    ///// Discovers all metrics from .mloop/scripts/metrics/*.cs
    ///// </summary>
    ///// <returns>List of discovered metric instances</returns>
    //public async Task<List<IMLoopMetric>> DiscoverMetricsAsync()
    //{
    //    var metricsPath = Path.Combine(_projectRoot, ".mloop", "scripts", "metrics");
    //    return await DiscoverScriptsAsync<IMLoopMetric>(metricsPath, "metrics");
    //}

    /// <summary>
    /// Generic discovery method for any interface type.
    /// </summary>
    /// <typeparam name="T">Interface type to discover</typeparam>
    /// <param name="searchPath">Directory to search for .cs files</param>
    /// <param name="typeName">Type name for logging (e.g., "hooks", "metrics")</param>
    /// <returns>List of discovered instances</returns>
    private async Task<List<T>> DiscoverScriptsAsync<T>(string searchPath, string typeName) where T : class
    {
        var instances = new List<T>();

        // Zero-overhead check: if directory doesn't exist, return empty list immediately
        if (!Directory.Exists(searchPath))
        {
            return instances;  // < 1ms performance guarantee
        }

        try
        {
            var scriptFiles = Directory.GetFiles(searchPath, "*.cs", SearchOption.TopDirectoryOnly);

            if (scriptFiles.Length == 0)
            {
                return instances;
            }

            Console.WriteLine($"🔍 Discovering {typeName} from {scriptFiles.Length} script(s)...");

            foreach (var scriptFile in scriptFiles)
            {
                try
                {
                    var loadedInstances = await _scriptLoader.LoadScriptAsync<T>(scriptFile);
                    instances.AddRange(loadedInstances);

                    if (loadedInstances.Count > 0)
                    {
                        var names = string.Join(", ", loadedInstances.Select(GetInstanceName));
                        Console.WriteLine($"  ✅ Loaded {loadedInstances.Count} {typeName} from {Path.GetFileName(scriptFile)}: {names}");
                    }
                }
                catch (Exception ex)
                {
                    // Graceful degradation: log error but continue with other scripts
                    Console.WriteLine($"  ⚠️ Failed to load {Path.GetFileName(scriptFile)}: {ex.Message}");
                }
            }

            if (instances.Count > 0)
            {
                Console.WriteLine($"✨ Total {instances.Count} {typeName} discovered");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error during {typeName} discovery: {ex.Message}");
        }

        return instances;
    }

    // NOTE: Phase 1 (Hooks & Metrics) - Disabled for Phase 0 (Preprocessing)
    // TODO: Re-enable when implementing Phase 1

    ///// <summary>
    ///// Gets the display name of a hook or metric instance.
    ///// </summary>
    //private string GetInstanceName(object instance)
    //{
    //    return instance switch
    //    {
    //        IMLoopHook hook => hook.Name,
    //        IMLoopMetric metric => metric.Name,
    //        _ => instance.GetType().Name
    //    };
    //}

    /// <summary>
    /// Gets the display name of an instance.
    /// </summary>
    private string GetInstanceName(object instance)
    {
        return instance.GetType().Name;
    }

    /// <summary>
    /// Checks if extensibility features are available in the project.
    /// Fast check (less than 1ms) that only checks directory existence.
    /// </summary>
    /// <returns>True if .mloop/scripts directory exists</returns>
    public bool IsExtensibilityAvailable()
    {
        var scriptsPath = Path.Combine(_projectRoot, ".mloop", "scripts");
        return Directory.Exists(scriptsPath);
    }

    /// <summary>
    /// Gets the full path to the scripts directory.
    /// </summary>
    public string GetScriptsDirectory()
    {
        return Path.Combine(_projectRoot, ".mloop", "scripts");
    }

    /// <summary>
    /// Gets the full path to the hooks directory.
    /// </summary>
    public string GetHooksDirectory()
    {
        return Path.Combine(_projectRoot, ".mloop", "scripts", "hooks");
    }

    /// <summary>
    /// Gets the full path to the metrics directory.
    /// </summary>
    public string GetMetricsDirectory()
    {
        return Path.Combine(_projectRoot, ".mloop", "scripts", "metrics");
    }

    /// <summary>
    /// Creates the standard directory structure for extensibility.
    /// </summary>
    public void InitializeDirectories()
    {
        Directory.CreateDirectory(GetHooksDirectory());
        Directory.CreateDirectory(GetMetricsDirectory());
        Console.WriteLine($"✅ Created extensibility directories:");
        Console.WriteLine($"  📁 {GetHooksDirectory()}");
        Console.WriteLine($"  📁 {GetMetricsDirectory()}");
    }
}
