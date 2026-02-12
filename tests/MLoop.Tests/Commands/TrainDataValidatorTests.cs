using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;

namespace MLoop.Tests.Commands;

public class TrainDataValidatorTests : IDisposable
{
    private readonly string _tempDir;
    private readonly IFileSystemManager _fileSystem;

    public TrainDataValidatorTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-tdv-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
        Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));
        Directory.CreateDirectory(Path.Combine(_tempDir, "datasets"));
        _fileSystem = new FileSystemManager();
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private string CreateCsv(string name, string content)
    {
        var path = Path.Combine(_tempDir, name);
        var dir = Path.GetDirectoryName(path);
        if (dir != null && !Directory.Exists(dir))
            Directory.CreateDirectory(dir);
        File.WriteAllText(path, content, System.Text.Encoding.UTF8);
        return path;
    }

    // --- ResolveDataFileAsync tests ---

    [Fact]
    public async Task ResolveDataFile_ExplicitPath_ReturnsResolvedPath()
    {
        var csvPath = CreateCsv("my_data.csv", "A,B,Label\n1,2,X\n");
        var config = new MLoopConfig();
        var discovery = new DatasetDiscovery(_fileSystem);

        var result = await TrainDataValidator.ResolveDataFileAsync(
            csvPath, config, _tempDir, discovery, _fileSystem);

        Assert.Equal(csvPath, result);
    }

    [Fact]
    public async Task ResolveDataFile_ExplicitPath_NotFound_ReturnsNull()
    {
        var config = new MLoopConfig();
        var discovery = new DatasetDiscovery(_fileSystem);

        var result = await TrainDataValidator.ResolveDataFileAsync(
            "nonexistent.csv", config, _tempDir, discovery, _fileSystem);

        Assert.Null(result);
    }

    [Fact]
    public async Task ResolveDataFile_RelativePath_ResolvesFromProjectRoot()
    {
        CreateCsv("relative_data.csv", "A,B,Label\n1,2,X\n");
        var config = new MLoopConfig();
        var discovery = new DatasetDiscovery(_fileSystem);

        var result = await TrainDataValidator.ResolveDataFileAsync(
            "relative_data.csv", config, _tempDir, discovery, _fileSystem);

        Assert.NotNull(result);
        Assert.True(File.Exists(result));
    }

    [Fact]
    public async Task ResolveDataFile_ConfigTrainPath_ReturnsConfigPath()
    {
        var csvPath = CreateCsv("datasets/custom_train.csv", "A,B,Label\n1,2,X\n");
        var config = new MLoopConfig { Data = new DataSettings { Train = "datasets/custom_train.csv" } };
        var discovery = new DatasetDiscovery(_fileSystem);

        var result = await TrainDataValidator.ResolveDataFileAsync(
            null, config, _tempDir, discovery, _fileSystem);

        Assert.NotNull(result);
        Assert.Contains("custom_train.csv", result);
    }

    [Fact]
    public async Task ResolveDataFile_AutoDiscovery_FindsTrainCsv()
    {
        CreateCsv("datasets/train.csv", "A,B,Label\n1,2,X\n");
        var config = new MLoopConfig();
        var discovery = new DatasetDiscovery(_fileSystem);

        var result = await TrainDataValidator.ResolveDataFileAsync(
            null, config, _tempDir, discovery, _fileSystem);

        Assert.NotNull(result);
        Assert.Contains("train.csv", result);
    }

    [Fact]
    public async Task ResolveDataFile_NoDataAnywhere_ReturnsNull()
    {
        var config = new MLoopConfig();
        var discovery = new DatasetDiscovery(_fileSystem);

        var result = await TrainDataValidator.ResolveDataFileAsync(
            null, config, _tempDir, discovery, _fileSystem);

        Assert.Null(result);
    }

    // --- ValidateLabelColumnAsync tests ---

    [Fact]
    public async Task ValidateLabel_ExistingColumn_DoesNotThrow()
    {
        var csvPath = CreateCsv("validate_ok.csv", "Feature1,Feature2,Label\n1,2,A\n3,4,B\n");

        await TrainDataValidator.ValidateLabelColumnAsync(csvPath, "Label", "test-model");
        // Should not throw
    }

    [Fact]
    public async Task ValidateLabel_MissingColumn_ThrowsArgumentException()
    {
        var csvPath = CreateCsv("validate_missing.csv", "Feature1,Feature2,Label\n1,2,A\n3,4,B\n");

        var ex = await Assert.ThrowsAsync<ArgumentException>(
            () => TrainDataValidator.ValidateLabelColumnAsync(csvPath, "NonExistent", "test-model"));

        Assert.Contains("NonExistent", ex.Message);
        Assert.Contains("test-model", ex.Message);
    }

    [Fact]
    public async Task ValidateLabel_EmptyFile_Throws()
    {
        var csvPath = CreateCsv("validate_empty.csv", "");

        // Empty CSV throws either InvalidOperationException or CsvHelper.ReaderException
        await Assert.ThrowsAnyAsync<Exception>(
            () => TrainDataValidator.ValidateLabelColumnAsync(csvPath, "Label", "test-model"));
    }
}
