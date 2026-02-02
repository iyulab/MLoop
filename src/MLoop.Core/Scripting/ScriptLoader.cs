using System.Reflection;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;
using MLoop.Extensibility;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Scripting;

/// <summary>
/// Loads and compiles .cs scripts into executable types with DLL caching for performance.
/// Implements hybrid strategy: Roslyn compilation on first run, cached DLL loading on subsequent runs.
/// </summary>
public class ScriptLoader
{
    private readonly string _cacheDirectory;
    private ScriptOptions? _scriptOptions;
    private readonly object _scriptOptionsLock = new();

    /// <summary>
    /// Initializes a new instance of ScriptLoader.
    /// </summary>
    /// <param name="cacheDirectory">Directory for cached DLL storage (default: .mloop/.cache/scripts/)</param>
    public ScriptLoader(string? cacheDirectory = null)
    {
        _cacheDirectory = cacheDirectory ?? Path.Combine(".mloop", ".cache", "scripts");
        // Note: _scriptOptions is lazily initialized to avoid Assembly.Location issues in single-file publish
    }

    /// <summary>
    /// Gets or creates the ScriptOptions instance (lazy initialization).
    /// This avoids Assembly.Location issues in single-file publish scenarios.
    /// </summary>
    private ScriptOptions GetScriptOptions()
    {
        if (_scriptOptions == null)
        {
            lock (_scriptOptionsLock)
            {
                if (_scriptOptions == null)
                {
                    // Ensure cache directory exists
                    Directory.CreateDirectory(_cacheDirectory);

                    // Configure script compilation options with necessary references
                    _scriptOptions = ScriptOptions.Default
                        .AddReferences(
                            typeof(object).Assembly,                    // System
                            typeof(IPreprocessingScript).Assembly,      // MLoop.Extensibility
                            typeof(Microsoft.ML.MLContext).Assembly,    // Microsoft.ML
                            typeof(FilePrepper.Pipeline.DataPipeline).Assembly  // FilePrepper
                        )
                        .AddImports(
                            "System",
                            "System.IO",
                            "System.Linq",
                            "System.Threading.Tasks",
                            "Microsoft.ML",
                            "MLoop.Extensibility",
                            "MLoop.Extensibility.Preprocessing",
                            "FilePrepper.Pipeline"
                        );
                }
            }
        }
        return _scriptOptions;
    }

    /// <summary>
    /// Loads a script and returns instances of all types implementing the specified interface.
    /// Uses DLL caching to improve performance on subsequent loads.
    /// </summary>
    /// <typeparam name="T">Interface type to search for (IMLoopHook or IMLoopMetric)</typeparam>
    /// <param name="scriptPath">Path to .cs script file</param>
    /// <returns>List of instances implementing T, or empty list on failure</returns>
    public async Task<List<T>> LoadScriptAsync<T>(string scriptPath) where T : class
    {
        try
        {
            if (!File.Exists(scriptPath))
            {
                Console.WriteLine($"⚠️ Script not found: {scriptPath}");
                return new List<T>();
            }

            var scriptContent = await File.ReadAllTextAsync(scriptPath);
            var scriptHash = ComputeHash(scriptContent);
            var cachedDllPath = GetCachedDllPath(scriptPath, scriptHash);

            Assembly assembly;

            // Check if cached DLL exists and is up-to-date
            if (File.Exists(cachedDllPath))
            {
                // Load from cache (fast path) - use byte array to avoid file locking
                var assemblyBytes = await File.ReadAllBytesAsync(cachedDllPath);
                assembly = Assembly.Load(assemblyBytes);
            }
            else
            {
                // Compile and cache (slow path)
                assembly = await CompileAndCacheAsync(scriptPath, scriptContent, cachedDllPath);
            }

            // Find and instantiate all types implementing T
            return InstantiateTypes<T>(assembly);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Failed to load script {scriptPath}: {ex.Message}");
            return new List<T>();  // Graceful degradation
        }
    }

    /// <summary>
    /// Compiles a .cs script using Roslyn and caches the resulting DLL.
    /// </summary>
    private async Task<Assembly> CompileAndCacheAsync(string scriptPath, string scriptContent, string cachedDllPath)
    {
        // Parse the script to extract class definitions
        var syntaxTree = CSharpSyntaxTree.ParseText(scriptContent);

        // Create compilation with necessary references
        var mlAssembly = typeof(Microsoft.ML.MLContext).Assembly;
        var filePrepperAssembly = typeof(FilePrepper.Pipeline.DataPipeline).Assembly;

        // Get references - handle single-file publish where Assembly.Location is empty
        var additionalReferences = new List<MetadataReference>();
        additionalReferences.AddRange(GetMetadataReferences(
            typeof(object).Assembly,
            typeof(IPreprocessingScript).Assembly,
            mlAssembly,
            filePrepperAssembly,
            typeof(System.Threading.Tasks.Task).Assembly,
            typeof(System.Linq.Enumerable).Assembly,
            typeof(System.IO.Path).Assembly,
            typeof(Console).Assembly
        ));

        // Add ML.NET DataView assembly if available
        var dataViewAssembly = mlAssembly.GetReferencedAssemblies()
            .FirstOrDefault(a => a.Name == "Microsoft.ML.DataView");
        if (dataViewAssembly != null)
        {
            try
            {
                var assembly = Assembly.Load(dataViewAssembly);
                additionalReferences.AddRange(GetMetadataReferences(assembly));
            }
            catch { /* Ignore if DataView assembly cannot be loaded */ }
        }

        // Add runtime assemblies for .NET
        var runtimePath = GetRuntimeDirectory();
        var runtimeReferences = new[]
        {
            "System.Runtime.dll",
            "System.Collections.dll",
            "System.Collections.Immutable.dll",
            "netstandard.dll"
        }
        .Select(name => Path.Combine(runtimePath, name))
        .Where(File.Exists)
        .Select(path => MetadataReference.CreateFromFile(path));

        var compilation = CSharpCompilation.Create(
            assemblyName: Path.GetFileNameWithoutExtension(scriptPath),
            syntaxTrees: new[] { syntaxTree },
            references: additionalReferences.Concat(runtimeReferences),
            options: new CSharpCompilationOptions(
                OutputKind.DynamicallyLinkedLibrary,
                optimizationLevel: OptimizationLevel.Release
            )
        );

        // Compile to memory stream
        using var ms = new MemoryStream();
        var result = compilation.Emit(ms);

        if (!result.Success)
        {
            var errors = string.Join("\n", result.Diagnostics
                .Where(d => d.Severity == DiagnosticSeverity.Error)
                .Select(d => $"  {d.Location.GetLineSpan().StartLinePosition}: {d.GetMessage()}"));

            throw new InvalidOperationException($"Compilation failed:\n{errors}");
        }

        // Save to cache - ensure directory exists
        ms.Seek(0, SeekOrigin.Begin);
        var assemblyBytes = ms.ToArray();
        var cacheDir = Path.GetDirectoryName(cachedDllPath);
        if (!string.IsNullOrEmpty(cacheDir) && !Directory.Exists(cacheDir))
        {
            Directory.CreateDirectory(cacheDir);
        }
        await File.WriteAllBytesAsync(cachedDllPath, assemblyBytes);

        // Load from memory to avoid file locking
        return Assembly.Load(assemblyBytes);
    }

    /// <summary>
    /// Finds all types in assembly implementing T and creates instances.
    /// </summary>
    private List<T> InstantiateTypes<T>(Assembly assembly) where T : class
    {
        var instances = new List<T>();

        try
        {
            var interfaceType = typeof(T);
            var types = assembly.GetTypes()
                .Where(t => t.IsClass && !t.IsAbstract && interfaceType.IsAssignableFrom(t));

            foreach (var type in types)
            {
                var instance = Activator.CreateInstance(type) as T;
                if (instance != null)
                {
                    instances.Add(instance);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"⚠️ Error instantiating types: {ex.Message}");
        }

        return instances;
    }

    /// <summary>
    /// Computes SHA256 hash of script content for cache invalidation.
    /// </summary>
    private string ComputeHash(string content)
    {
        var bytes = Encoding.UTF8.GetBytes(content);
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash);
    }

    /// <summary>
    /// Gets the cached DLL path for a script based on its hash.
    /// </summary>
    private string GetCachedDllPath(string scriptPath, string hash)
    {
        var scriptFileName = Path.GetFileNameWithoutExtension(scriptPath);
        return Path.Combine(_cacheDirectory, $"{scriptFileName}_{hash[..8]}.dll");
    }

    /// <summary>
    /// Clears all cached DLLs.
    /// </summary>
    public void ClearCache()
    {
        try
        {
            if (Directory.Exists(_cacheDirectory))
            {
                Directory.Delete(_cacheDirectory, recursive: true);
                Directory.CreateDirectory(_cacheDirectory);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"⚠️ Failed to clear cache: {ex.Message}");
        }
    }

    /// <summary>
    /// Gets MetadataReferences for assemblies, handling single-file publish scenarios.
    /// </summary>
    private static IEnumerable<MetadataReference> GetMetadataReferences(params Assembly[] assemblies)
    {
        foreach (var assembly in assemblies)
        {
            var location = assembly.Location;
            if (!string.IsNullOrEmpty(location) && File.Exists(location))
            {
                yield return MetadataReference.CreateFromFile(location);
            }
            else
            {
                // For single-file publish, try to find the assembly in the runtime directory
                var runtimeDir = GetRuntimeDirectory();
                var assemblyName = assembly.GetName().Name + ".dll";
                var runtimePath = Path.Combine(runtimeDir, assemblyName);

                if (File.Exists(runtimePath))
                {
                    yield return MetadataReference.CreateFromFile(runtimePath);
                }
                // If still not found, skip this assembly (it may be embedded)
            }
        }
    }

    /// <summary>
    /// Gets the .NET runtime directory for loading reference assemblies.
    /// </summary>
    private static string GetRuntimeDirectory()
    {
        // Try Assembly.Location first (works in normal execution)
        var coreLibLocation = typeof(object).Assembly.Location;
        if (!string.IsNullOrEmpty(coreLibLocation))
        {
            return Path.GetDirectoryName(coreLibLocation)!;
        }

        // Fallback for single-file publish: use RuntimeEnvironment
        var runtimeDir = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();
        if (!string.IsNullOrEmpty(runtimeDir) && Directory.Exists(runtimeDir))
        {
            return runtimeDir;
        }

        // Last resort: AppContext.BaseDirectory
        return AppContext.BaseDirectory;
    }
}
