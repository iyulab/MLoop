namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Project discovery implementation that searches for a .mloop/ directory.
/// </summary>
public class ProjectDiscovery : IProjectDiscovery
{
    private const string MLoopDirectoryName = ".mloop";
    private readonly IFileSystemManager _fileSystem;

    // The global runtime cache (created by `mloop runtime install`) lives at &lt;userProfile&gt;/.mloop/runtimes,
    // so the user-profile directory itself contains a .mloop directory. It is NOT a project: without
    // excluding it, FindRoot() walking up from any directory under $HOME would match that cache and treat
    // $HOME as the project root — making relative data paths resolve against $HOME and every command run
    // outside a real project silently misbehave (e.g. `mloop info data.csv` → "File not found: ~/data.csv").
    // A project at the user-profile root is impossible anyway (it would collide with the cache at the same
    // path), so excluding it is a necessary invariant, not just a heuristic.
    private readonly string? _runtimeCacheRoot;

    /// <param name="runtimeCacheRoot">Directory whose .mloop is the global runtime cache and must never be a
    /// project root. Defaults to the user-profile directory; tests inject a fake. A single constructor (with
    /// an optional parameter) is intentional — a second public constructor would make the DI container's
    /// greedy constructor selection fail to resolve the unregistered string and crash service startup.</param>
    public ProjectDiscovery(IFileSystemManager fileSystem, string? runtimeCacheRoot = null)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
        var cacheRoot = string.IsNullOrEmpty(runtimeCacheRoot)
            ? Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)
            : runtimeCacheRoot;
        _runtimeCacheRoot = string.IsNullOrEmpty(cacheRoot) ? null : cacheRoot; // empty (rare) → no exclusion
    }

    /// <summary>
    /// The environment variable that names the project explicitly, taking precedence over walking up
    /// from the working directory.
    /// </summary>
    /// <remarks>
    /// A server or a container has no meaningful "current directory" — it has a deployment. Both
    /// <c>mloop serve</c> and the Dockerfile <c>mloop docker</c> generates already set this variable;
    /// until now nothing read it, so a container worked only because its WORKDIR happened to be the
    /// project, and moving the entry point would have broken it silently.
    /// </remarks>
    public const string ProjectRootVariable = "MLOOP_PROJECT_ROOT";

    public string FindRoot()
    {
        var configured = Environment.GetEnvironmentVariable(ProjectRootVariable);
        if (!string.IsNullOrWhiteSpace(configured))
        {
            var root = Path.GetFullPath(configured.Trim());
            if (!IsProjectRoot(root))
                throw new InvalidOperationException(
                    $"{ProjectRootVariable} is set to '{root}', which is not a MLoop project "
                    + $"(no {MLoopDirectoryName} directory there). {NotInsideProjectGuidance}");

            // Deliberately not falling back to the search: a deployment that names its project and
            // names it wrongly should say so, not quietly serve whatever directory it started in.
            return root;
        }

        return FindRoot(Directory.GetCurrentDirectory());
    }

    public string FindRoot(string startingDirectory)
    {
        var directory = new DirectoryInfo(startingDirectory);

        while (directory != null)
        {
            if (IsProjectRoot(directory.FullName))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException($"{NotInsideProjectCause} {NotInsideProjectGuidance}");
    }

    /// <summary>
    /// What went wrong when no project root is found, worded once. Twelve call sites report this
    /// failure — eleven commands catching the exception plus <see cref="FindRoot(string)"/> throwing
    /// it — and each used to carry its own copy, so the guidance drifted between the thrown message
    /// ("or navigate to an existing project directory") and what the commands printed (only "run
    /// mloop init"). Report it through <c>ErrorConsole.Error(cause, guidance)</c> so the two lines
    /// reach a machine consumer as one event.
    /// </summary>
    public const string NotInsideProjectCause = "Not inside a MLoop project.";

    /// <inheritdoc cref="NotInsideProjectCause"/>
    public const string NotInsideProjectGuidance =
        "Run 'mloop init' to create a new project, or navigate to an existing project directory.";

    public bool IsProjectRoot(string path)
    {
        var mloopPath = _fileSystem.CombinePath(path, MLoopDirectoryName);
        if (!_fileSystem.DirectoryExists(mloopPath))
            return false;

        // The user-profile root's .mloop is the global runtime cache, not a project (see field comment).
        if (_runtimeCacheRoot != null && PathsEqual(path, _runtimeCacheRoot))
            return false;

        return true;
    }

    public void EnsureProjectRoot()
    {
        // This will throw if not in a project
        _ = FindRoot();
    }

    public string GetMLoopDirectory(string projectRoot)
    {
        return _fileSystem.CombinePath(projectRoot, MLoopDirectoryName);
    }

    private static bool PathsEqual(string a, string b)
    {
        var comparison = OperatingSystem.IsWindows()
            ? StringComparison.OrdinalIgnoreCase
            : StringComparison.Ordinal;
        return string.Equals(
            Path.TrimEndingDirectorySeparator(Path.GetFullPath(a)),
            Path.TrimEndingDirectorySeparator(Path.GetFullPath(b)),
            comparison);
    }
}
