namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Filesystem operations abstraction for MLoop project management
/// </summary>
public interface IFileSystemManager
{
    /// <summary>
    /// Creates a directory recursively if it doesn't exist
    /// </summary>
    Task CreateDirectoryAsync(string path, CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if a directory exists
    /// </summary>
    bool DirectoryExists(string path);

    /// <summary>
    /// Checks if a file exists
    /// </summary>
    bool FileExists(string path);

    /// <summary>
    /// Writes JSON content to a file
    /// </summary>
    Task WriteJsonAsync<T>(string path, T content, CancellationToken cancellationToken = default);

    /// <summary>
    /// Reads JSON content from a file
    /// </summary>
    Task<T> ReadJsonAsync<T>(string path, CancellationToken cancellationToken = default);

    /// <summary>
    /// Writes text content to a file
    /// </summary>
    Task WriteTextAsync(string path, string content, CancellationToken cancellationToken = default);

    /// <summary>
    /// Reads text content from a file
    /// </summary>
    Task<string> ReadTextAsync(string path, CancellationToken cancellationToken = default);

    /// <summary>
    /// Copies a file to a destination
    /// </summary>
    Task CopyFileAsync(string sourcePath, string destinationPath, bool overwrite = false, CancellationToken cancellationToken = default);

    /// <summary>
    /// Deletes a directory recursively
    /// </summary>
    Task DeleteDirectoryAsync(string path, CancellationToken cancellationToken = default);

    /// <summary>
    /// Lists all files in a directory matching a pattern
    /// </summary>
    IEnumerable<string> GetFiles(string path, string searchPattern = "*", bool recursive = false);

    /// <summary>
    /// Gets the absolute path from a relative path
    /// </summary>
    string GetAbsolutePath(string path);

    /// <summary>
    /// Combines path segments
    /// </summary>
    string CombinePath(params string[] paths);
}
