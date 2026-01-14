using System.Text.Json;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// Default filesystem manager implementation with thread-safe operations
/// </summary>
public class FileSystemManager : IFileSystemManager
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true
    };

    public Task CreateDirectoryAsync(string path, CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
        return Task.CompletedTask;
    }

    public bool DirectoryExists(string path)
    {
        return Directory.Exists(path);
    }

    public bool FileExists(string path)
    {
        return File.Exists(path);
    }

    public async Task WriteJsonAsync<T>(string path, T content, CancellationToken cancellationToken = default)
    {
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
        {
            await CreateDirectoryAsync(directory, cancellationToken);
        }

        var json = JsonSerializer.Serialize(content, JsonOptions);
        await File.WriteAllTextAsync(path, json, cancellationToken);
    }

    public async Task<T> ReadJsonAsync<T>(string path, CancellationToken cancellationToken = default)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"File not found: {path}");
        }

        var json = await File.ReadAllTextAsync(path, cancellationToken);
        return JsonSerializer.Deserialize<T>(json, JsonOptions)
            ?? throw new InvalidOperationException($"Failed to deserialize JSON from {path}");
    }

    public async Task WriteTextAsync(string path, string content, CancellationToken cancellationToken = default)
    {
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
        {
            await CreateDirectoryAsync(directory, cancellationToken);
        }

        await File.WriteAllTextAsync(path, content, cancellationToken);
    }

    public async Task<string> ReadTextAsync(string path, CancellationToken cancellationToken = default)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"File not found: {path}");
        }

        return await File.ReadAllTextAsync(path, cancellationToken);
    }

    public async Task CopyFileAsync(string sourcePath, string destinationPath, bool overwrite = false, CancellationToken cancellationToken = default)
    {
        if (!File.Exists(sourcePath))
        {
            throw new FileNotFoundException($"Source file not found: {sourcePath}");
        }

        var directory = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrEmpty(directory))
        {
            await CreateDirectoryAsync(directory, cancellationToken);
        }

        await Task.Run(() => File.Copy(sourcePath, destinationPath, overwrite), cancellationToken);
    }

    public Task DeleteDirectoryAsync(string path, CancellationToken cancellationToken = default)
    {
        if (Directory.Exists(path))
        {
            Directory.Delete(path, recursive: true);
        }
        return Task.CompletedTask;
    }

    public IEnumerable<string> GetFiles(string path, string searchPattern = "*", bool recursive = false)
    {
        if (!Directory.Exists(path))
        {
            return Enumerable.Empty<string>();
        }

        var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        return Directory.GetFiles(path, searchPattern, searchOption);
    }

    public string GetAbsolutePath(string path)
    {
        return Path.GetFullPath(path);
    }

    public string CombinePath(params string[] paths)
    {
        return Path.Combine(paths);
    }
}
