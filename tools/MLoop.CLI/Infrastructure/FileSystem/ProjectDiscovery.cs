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

    public string FindRoot()
    {
        var currentDirectory = Directory.GetCurrentDirectory();
        return FindRoot(currentDirectory);
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

        throw new InvalidOperationException(
            $"Not inside a MLoop project. Run 'mloop init' to create a new project, " +
            $"or navigate to an existing project directory.");
    }

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
