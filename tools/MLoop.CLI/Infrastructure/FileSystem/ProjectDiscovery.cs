namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Project discovery implementation that searches for .mloop/ directory
/// </summary>
public class ProjectDiscovery : IProjectDiscovery
{
    private const string MLoopDirectoryName = ".mloop";
    private readonly IFileSystemManager _fileSystem;

    public ProjectDiscovery(IFileSystemManager fileSystem)
    {
        _fileSystem = fileSystem ?? throw new ArgumentNullException(nameof(fileSystem));
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
        return _fileSystem.DirectoryExists(mloopPath);
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
}
