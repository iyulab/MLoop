namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Discovers MLoop project root by searching for .mloop/ directory
/// </summary>
public interface IProjectDiscovery
{
    /// <summary>
    /// Finds the project root by traversing up from current directory
    /// </summary>
    /// <returns>Absolute path to project root</returns>
    /// <exception cref="InvalidOperationException">Thrown when project root not found</exception>
    string FindRoot();

    /// <summary>
    /// Finds the project root starting from a specific directory
    /// </summary>
    string FindRoot(string startingDirectory);

    /// <summary>
    /// Checks if a directory is a MLoop project root
    /// </summary>
    bool IsProjectRoot(string path);

    /// <summary>
    /// Ensures we are inside a MLoop project, throws if not
    /// </summary>
    void EnsureProjectRoot();

    /// <summary>
    /// Gets the .mloop directory path
    /// </summary>
    string GetMLoopDirectory(string projectRoot);
}
