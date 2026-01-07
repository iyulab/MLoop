using MLoop.Core.Scripting;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Hooks;

/// <summary>
/// Executes lifecycle hook scripts at specific pipeline stages.
/// Discovers and runs hooks from .mloop/scripts/hooks/{hook-type}/*.cs
/// </summary>
public class HookEngine
{
    private readonly string _projectRoot;
    private readonly ScriptLoader _scriptLoader;
    private readonly ILogger _logger;

    // Hook type to directory name mapping
    private static readonly Dictionary<HookType, string> HookDirectories = new()
    {
        [HookType.PreTrain] = "pre-train",
        [HookType.PostTrain] = "post-train",
        [HookType.PrePredict] = "pre-predict",
        [HookType.PostEvaluate] = "post-evaluate"
    };

    /// <summary>
    /// Initializes a new instance of HookEngine.
    /// </summary>
    /// <param name="projectRoot">Project root directory</param>
    /// <param name="logger">Logger for user feedback</param>
    /// <param name="scriptLoader">Optional ScriptLoader instance</param>
    public HookEngine(
        string projectRoot,
        ILogger logger,
        ScriptLoader? scriptLoader = null)
    {
        _projectRoot = projectRoot;
        _logger = logger;
        _scriptLoader = scriptLoader ?? new ScriptLoader();
    }

    /// <summary>
    /// Gets the directory path for a specific hook type.
    /// </summary>
    /// <param name="hookType">Type of hook</param>
    /// <returns>Full path to hook directory</returns>
    public string GetHookDirectory(HookType hookType)
    {
        var directoryName = HookDirectories[hookType];
        return Path.Combine(_projectRoot, ".mloop", "scripts", "hooks", directoryName);
    }

    /// <summary>
    /// Checks if any hooks exist for the specified type.
    /// </summary>
    /// <param name="hookType">Type of hook to check</param>
    /// <returns>True if hooks exist, false otherwise</returns>
    public bool HasHooks(HookType hookType)
    {
        var hookPath = GetHookDirectory(hookType);

        if (!Directory.Exists(hookPath))
            return false;

        var scriptFiles = Directory.GetFiles(hookPath, "*.cs", SearchOption.TopDirectoryOnly);
        return scriptFiles.Length > 0;
    }

    /// <summary>
    /// Initializes hook directory structure for all hook types.
    /// </summary>
    public void InitializeDirectories()
    {
        foreach (var hookType in Enum.GetValues<HookType>())
        {
            var hookPath = GetHookDirectory(hookType);
            Directory.CreateDirectory(hookPath);
        }
    }

    /// <summary>
    /// Executes all hooks for the specified type.
    /// Hooks are executed in alphabetical order.
    /// </summary>
    /// <param name="hookType">Type of hooks to execute</param>
    /// <param name="context">Execution context with data and configuration</param>
    /// <returns>
    /// True if all hooks passed (Continue or ModifyConfig),
    /// False if any hook aborted execution
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown when hook execution fails critically</exception>
    public async Task<bool> ExecuteHooksAsync(HookType hookType, HookContext context)
    {
        var hookPath = GetHookDirectory(hookType);

        // Zero-overhead check: if directory doesn't exist, return immediately
        if (!Directory.Exists(hookPath))
        {
            return true;  // No hooks = success (< 1ms performance)
        }

        try
        {
            // Discover all hook scripts
            var scriptFiles = Directory.GetFiles(hookPath, "*.cs", SearchOption.TopDirectoryOnly)
                .OrderBy(f => Path.GetFileName(f))  // Alphabetical execution order
                .ToList();

            if (scriptFiles.Count == 0)
            {
                return true;  // No hooks = success
            }

            _logger.Info($"🔗 Executing {scriptFiles.Count} {hookType} hook(s)...");

            // Execute hooks sequentially
            int hookIndex = 1;

            foreach (var scriptFile in scriptFiles)
            {
                try
                {
                    var scriptName = Path.GetFileName(scriptFile);
                    _logger.Info($"  🪝 [{hookIndex}/{scriptFiles.Count}] {scriptName}");

                    // Load hook script
                    var hooks = await _scriptLoader.LoadScriptAsync<IMLoopHook>(scriptFile);

                    if (hooks.Count == 0)
                    {
                        _logger.Warning($"    ⚠️ No IMLoopHook implementations found in {scriptName}");
                        continue;
                    }

                    if (hooks.Count > 1)
                    {
                        _logger.Warning($"    ⚠️ Multiple implementations found, using first: {hooks[0].GetType().Name}");
                    }

                    // Execute first hook implementation
                    var hook = hooks[0];
                    var startTime = DateTime.UtcNow;

                    // Update context with hook name
                    var hookContext = new HookContext
                    {
                        HookType = context.HookType,
                        HookName = scriptName,
                        MLContext = context.MLContext,
                        DataView = context.DataView,
                        Model = context.Model,
                        ExperimentResult = context.ExperimentResult,
                        Metrics = context.Metrics,
                        ProjectRoot = context.ProjectRoot,
                        Logger = context.Logger,
                        Metadata = context.Metadata
                    };

                    var result = await hook.ExecuteAsync(hookContext);
                    var elapsedMs = (DateTime.UtcNow - startTime).TotalMilliseconds;

                    // Handle result
                    switch (result.Action)
                    {
                        case HookAction.Continue:
                            _logger.Info($"    ✅ {hook.Name} (Completed in {elapsedMs:F0}ms)");
                            if (!string.IsNullOrEmpty(result.Message))
                            {
                                _logger.Info($"       {result.Message}");
                            }
                            break;

                        case HookAction.Abort:
                            _logger.Error($"    ❌ {hook.Name}: {result.Message}");
                            _logger.Error($"    🛑 Pipeline execution aborted by hook");
                            return false;  // Abort pipeline

                        case HookAction.ModifyConfig:
                            _logger.Info($"    🔧 {hook.Name}: Configuration modified");
                            if (!string.IsNullOrEmpty(result.Message))
                            {
                                _logger.Info($"       {result.Message}");
                            }

                            // Apply configuration modifications to context metadata
                            if (result.ConfigModifications != null)
                            {
                                foreach (var kvp in result.ConfigModifications)
                                {
                                    context.Metadata[kvp.Key] = kvp.Value;
                                    _logger.Debug($"       - {kvp.Key} = {kvp.Value}");
                                }
                            }
                            break;
                    }

                    // Performance warning if hook is slow
                    if (elapsedMs > 50)
                    {
                        _logger.Warning($"    ⚠️ Hook took {elapsedMs:F0}ms (> 50ms threshold)");
                    }

                    hookIndex++;
                }
                catch (Exception ex)
                {
                    var scriptName = Path.GetFileName(scriptFile);
                    _logger.Error($"    ❌ Hook execution failed: {scriptName}");
                    _logger.Error($"       {ex.Message}");

                    // Hook failures don't abort pipeline (graceful degradation)
                    // Only explicit HookResult.Abort() stops execution
                    _logger.Warning($"    ⚠️ Continuing despite hook failure");
                    continue;
                }
            }

            _logger.Info($"✅ All {hookType} hooks completed");
            return true;
        }
        catch (Exception ex)
        {
            _logger.Error($"❌ Hook engine failure: {ex.Message}");
            throw new InvalidOperationException($"Hook execution failed for {hookType}", ex);
        }
    }
}
