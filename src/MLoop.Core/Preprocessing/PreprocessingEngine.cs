using MLoop.Core.Data;
using MLoop.Core.Scripting;
using MLoop.Extensibility;

namespace MLoop.Core.Preprocessing;

/// <summary>
/// Executes preprocessing scripts in sequential order before AutoML training.
/// Discovers and runs scripts from .mloop/scripts/preprocess/*.cs
/// </summary>
public class PreprocessingEngine
{
    private readonly string _projectRoot;
    private readonly ScriptLoader _scriptLoader;
    private readonly ICsvHelper _csvHelper;
    private readonly ILogger _logger;

    /// <summary>
    /// Initializes a new instance of PreprocessingEngine.
    /// </summary>
    /// <param name="projectRoot">Project root directory</param>
    /// <param name="logger">Logger for user feedback</param>
    /// <param name="scriptLoader">Optional ScriptLoader instance</param>
    /// <param name="csvHelper">Optional CSV helper instance</param>
    public PreprocessingEngine(
        string projectRoot,
        ILogger logger,
        ScriptLoader? scriptLoader = null,
        ICsvHelper? csvHelper = null)
    {
        _projectRoot = projectRoot;
        _logger = logger;
        _scriptLoader = scriptLoader ?? new ScriptLoader();
        _csvHelper = csvHelper ?? new CsvHelperImpl();
    }

    /// <summary>
    /// Executes all preprocessing scripts sequentially on the input data.
    /// Each script receives the output of the previous script as input.
    /// </summary>
    /// <param name="inputPath">Path to the raw input CSV file</param>
    /// <param name="labelColumn">Optional label column name</param>
    /// <returns>Path to the final preprocessed CSV file, or original path if no scripts found</returns>
    public async Task<string> ExecuteAsync(string inputPath, string? labelColumn = null)
    {
        var preprocessPath = Path.Combine(_projectRoot, ".mloop", "scripts", "preprocess");

        // Zero-overhead check: if directory doesn't exist, return input path immediately
        if (!Directory.Exists(preprocessPath))
        {
            return inputPath;  // < 1ms performance guarantee
        }

        try
        {
            // Discover all preprocessing scripts
            var scriptFiles = Directory.GetFiles(preprocessPath, "*.cs", SearchOption.TopDirectoryOnly)
                .OrderBy(f => Path.GetFileName(f))  // 01_*.cs, 02_*.cs, 03_*.cs
                .ToList();

            if (scriptFiles.Count == 0)
            {
                _logger.Info("📋 No preprocessing scripts found");
                return inputPath;
            }

            _logger.Info($"🔄 Running {scriptFiles.Count} preprocessing script(s)...");

            // Create temporary output directory
            var tempDirectory = Path.Combine(_projectRoot, ".mloop", "temp", "preprocess");
            Directory.CreateDirectory(tempDirectory);

            // Execute scripts sequentially
            string currentInputPath = inputPath;
            int scriptIndex = 1;

            foreach (var scriptFile in scriptFiles)
            {
                try
                {
                    var scriptName = Path.GetFileName(scriptFile);
                    _logger.Info($"  📝 [{scriptIndex}/{scriptFiles.Count}] {scriptName}");

                    // Load script
                    var scripts = await _scriptLoader.LoadScriptAsync<IPreprocessingScript>(scriptFile);

                    if (scripts.Count == 0)
                    {
                        _logger.Warning($"    ⚠️ No IPreprocessingScript implementations found in {scriptName}");
                        continue;
                    }

                    if (scripts.Count > 1)
                    {
                        _logger.Warning($"    ⚠️ Multiple implementations found, using first: {scripts[0].GetType().Name}");
                    }

                    var script = scripts[0];

                    // Create context
                    var context = new PreprocessContext
                    {
                        InputPath = currentInputPath,
                        OutputDirectory = tempDirectory,
                        ProjectRoot = _projectRoot,
                        Csv = _csvHelper,
                        Logger = _logger
                    };

                    // Initialize metadata
                    var metadata = new Dictionary<string, object>
                    {
                        ["ScriptSequence"] = scriptIndex,
                        ["TotalScripts"] = scriptFiles.Count,
                        ["ScriptName"] = scriptName
                    };

                    if (!string.IsNullOrEmpty(labelColumn))
                    {
                        metadata["LabelColumn"] = labelColumn;
                    }

                    context.InitializeMetadata(metadata);

                    // Execute script
                    var outputPath = await script.ExecuteAsync(context);

                    if (string.IsNullOrEmpty(outputPath) || !File.Exists(outputPath))
                    {
                        throw new InvalidOperationException(
                            $"Script {scriptName} did not produce a valid output file");
                    }

                    _logger.Info($"    ✅ Output: {Path.GetFileName(outputPath)}");

                    // Update input path for next script
                    currentInputPath = outputPath;
                    scriptIndex++;
                }
                catch (Exception ex)
                {
                    _logger.Error($"    ❌ Failed: {ex.Message}");
                    throw new InvalidOperationException(
                        $"Preprocessing failed at script {Path.GetFileName(scriptFile)}: {ex.Message}",
                        ex);
                }
            }

            _logger.Info($"✅ Preprocessing complete");
            return currentInputPath;
        }
        catch (Exception ex)
        {
            _logger.Error($"❌ Preprocessing error: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Checks if preprocessing scripts are available in the project.
    /// Fast check (less than 1ms) that only checks directory existence.
    /// </summary>
    public bool HasPreprocessingScripts()
    {
        var preprocessPath = Path.Combine(_projectRoot, ".mloop", "scripts", "preprocess");
        if (!Directory.Exists(preprocessPath))
        {
            return false;
        }

        var scriptFiles = Directory.GetFiles(preprocessPath, "*.cs", SearchOption.TopDirectoryOnly);
        return scriptFiles.Length > 0;
    }

    /// <summary>
    /// Gets the preprocessing scripts directory path.
    /// </summary>
    public string GetPreprocessingDirectory()
    {
        return Path.Combine(_projectRoot, ".mloop", "scripts", "preprocess");
    }

    /// <summary>
    /// Initializes the preprocessing scripts directory.
    /// </summary>
    public void InitializeDirectory()
    {
        var preprocessPath = GetPreprocessingDirectory();
        Directory.CreateDirectory(preprocessPath);
        _logger.Info($"✅ Created preprocessing directory: {preprocessPath}");
    }

    /// <summary>
    /// Cleans up temporary preprocessing files.
    /// </summary>
    public void CleanupTempFiles()
    {
        try
        {
            var tempDirectory = Path.Combine(_projectRoot, ".mloop", "temp", "preprocess");
            if (Directory.Exists(tempDirectory))
            {
                Directory.Delete(tempDirectory, recursive: true);
                _logger.Debug($"🧹 Cleaned up preprocessing temp files");
            }
        }
        catch (Exception ex)
        {
            _logger.Warning($"⚠️ Failed to cleanup temp files: {ex.Message}");
        }
    }
}
