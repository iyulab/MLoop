using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.FileSystem;

public class FileSystemManagerTests : IDisposable
{
    private readonly FileSystemManager _fs = new();
    private readonly string _tempDir;

    public FileSystemManagerTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-fsm-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    #region Directory Operations

    [Fact]
    public async Task CreateDirectoryAsync_CreatesDirectory()
    {
        var path = Path.Combine(_tempDir, "sub", "nested");

        await _fs.CreateDirectoryAsync(path);

        Assert.True(Directory.Exists(path));
    }

    [Fact]
    public async Task CreateDirectoryAsync_AlreadyExists_NoError()
    {
        var path = Path.Combine(_tempDir, "existing");
        Directory.CreateDirectory(path);

        await _fs.CreateDirectoryAsync(path);

        Assert.True(Directory.Exists(path));
    }

    [Fact]
    public void DirectoryExists_ExistingDir_ReturnsTrue()
    {
        Assert.True(_fs.DirectoryExists(_tempDir));
    }

    [Fact]
    public void DirectoryExists_NonExistent_ReturnsFalse()
    {
        Assert.False(_fs.DirectoryExists(Path.Combine(_tempDir, "nonexistent")));
    }

    [Fact]
    public async Task DeleteDirectoryAsync_RemovesDirectory()
    {
        var path = Path.Combine(_tempDir, "to-delete");
        Directory.CreateDirectory(path);
        await File.WriteAllTextAsync(Path.Combine(path, "file.txt"), "data");

        await _fs.DeleteDirectoryAsync(path);

        Assert.False(Directory.Exists(path));
    }

    [Fact]
    public async Task DeleteDirectoryAsync_NonExistent_NoError()
    {
        await _fs.DeleteDirectoryAsync(Path.Combine(_tempDir, "nonexistent"));
    }

    #endregion

    #region File Existence

    [Fact]
    public void FileExists_ExistingFile_ReturnsTrue()
    {
        var path = Path.Combine(_tempDir, "exists.txt");
        File.WriteAllText(path, "content");

        Assert.True(_fs.FileExists(path));
    }

    [Fact]
    public void FileExists_NonExistent_ReturnsFalse()
    {
        Assert.False(_fs.FileExists(Path.Combine(_tempDir, "nonexistent.txt")));
    }

    #endregion

    #region JSON Operations

    [Fact]
    public async Task WriteJsonAsync_ReadJsonAsync_RoundTrip()
    {
        var path = Path.Combine(_tempDir, "test.json");
        var data = new TestData { Name = "hello", Value = 42 };

        await _fs.WriteJsonAsync(path, data);
        var result = await _fs.ReadJsonAsync<TestData>(path);

        Assert.Equal("hello", result.Name);
        Assert.Equal(42, result.Value);
    }

    [Fact]
    public async Task WriteJsonAsync_CreatesParentDirectory()
    {
        var path = Path.Combine(_tempDir, "new-dir", "test.json");
        var data = new TestData { Name = "test" };

        await _fs.WriteJsonAsync(path, data);

        Assert.True(File.Exists(path));
    }

    [Fact]
    public async Task ReadJsonAsync_FileNotFound_Throws()
    {
        var path = Path.Combine(_tempDir, "nonexistent.json");

        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _fs.ReadJsonAsync<TestData>(path));
    }

    #endregion

    #region Text Operations

    [Fact]
    public async Task WriteTextAsync_ReadTextAsync_RoundTrip()
    {
        var path = Path.Combine(_tempDir, "test.txt");

        await _fs.WriteTextAsync(path, "hello world");
        var result = await _fs.ReadTextAsync(path);

        Assert.Equal("hello world", result);
    }

    [Fact]
    public async Task WriteTextAsync_CreatesParentDirectory()
    {
        var path = Path.Combine(_tempDir, "text-dir", "test.txt");

        await _fs.WriteTextAsync(path, "content");

        Assert.True(File.Exists(path));
    }

    [Fact]
    public async Task ReadTextAsync_FileNotFound_Throws()
    {
        var path = Path.Combine(_tempDir, "nonexistent.txt");

        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _fs.ReadTextAsync(path));
    }

    #endregion

    #region Copy Operations

    [Fact]
    public async Task CopyFileAsync_CopiesFile()
    {
        var src = Path.Combine(_tempDir, "source.txt");
        var dst = Path.Combine(_tempDir, "dest.txt");
        await File.WriteAllTextAsync(src, "data");

        await _fs.CopyFileAsync(src, dst);

        Assert.True(File.Exists(dst));
        Assert.Equal("data", await File.ReadAllTextAsync(dst));
    }

    [Fact]
    public async Task CopyFileAsync_Overwrite_Replaces()
    {
        var src = Path.Combine(_tempDir, "source.txt");
        var dst = Path.Combine(_tempDir, "dest.txt");
        await File.WriteAllTextAsync(src, "new");
        await File.WriteAllTextAsync(dst, "old");

        await _fs.CopyFileAsync(src, dst, overwrite: true);

        Assert.Equal("new", await File.ReadAllTextAsync(dst));
    }

    [Fact]
    public async Task CopyFileAsync_SourceNotFound_Throws()
    {
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _fs.CopyFileAsync(
                Path.Combine(_tempDir, "nonexistent.txt"),
                Path.Combine(_tempDir, "dest.txt")));
    }

    [Fact]
    public async Task CopyFileAsync_CreatesParentDirectory()
    {
        var src = Path.Combine(_tempDir, "source.txt");
        var dst = Path.Combine(_tempDir, "copy-dir", "dest.txt");
        await File.WriteAllTextAsync(src, "data");

        await _fs.CopyFileAsync(src, dst);

        Assert.True(File.Exists(dst));
    }

    #endregion

    #region GetFiles

    [Fact]
    public void GetFiles_ReturnsMatchingFiles()
    {
        File.WriteAllText(Path.Combine(_tempDir, "a.txt"), "");
        File.WriteAllText(Path.Combine(_tempDir, "b.txt"), "");
        File.WriteAllText(Path.Combine(_tempDir, "c.json"), "");

        var txtFiles = _fs.GetFiles(_tempDir, "*.txt").ToList();

        Assert.Equal(2, txtFiles.Count);
    }

    [Fact]
    public void GetFiles_Recursive_FindsNested()
    {
        var sub = Path.Combine(_tempDir, "sub");
        Directory.CreateDirectory(sub);
        File.WriteAllText(Path.Combine(_tempDir, "a.txt"), "");
        File.WriteAllText(Path.Combine(sub, "b.txt"), "");

        var files = _fs.GetFiles(_tempDir, "*.txt", recursive: true).ToList();

        Assert.Equal(2, files.Count);
    }

    [Fact]
    public void GetFiles_NonExistentDir_ReturnsEmpty()
    {
        var files = _fs.GetFiles(Path.Combine(_tempDir, "nonexistent"));

        Assert.Empty(files);
    }

    #endregion

    #region Path Operations

    [Fact]
    public void CombinePath_CombinesSegments()
    {
        var result = _fs.CombinePath("a", "b", "c.txt");
        Assert.Equal(Path.Combine("a", "b", "c.txt"), result);
    }

    [Fact]
    public void GetAbsolutePath_ReturnsAbsolute()
    {
        var result = _fs.GetAbsolutePath("relative/path");
        Assert.True(Path.IsPathRooted(result));
    }

    #endregion

    private class TestData
    {
        public string? Name { get; set; }
        public int Value { get; set; }
    }
}
