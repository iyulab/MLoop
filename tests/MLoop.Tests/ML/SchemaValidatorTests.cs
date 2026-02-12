using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.ML;

public class SchemaValidatorTests : IDisposable
{
    private readonly string _tempDir;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;

    public SchemaValidatorTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-sv-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);

        _fileSystem = new FileSystemManager();
        _projectDiscovery = new ProjectDiscovery(_fileSystem);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    private string CreateCsv(string name, string content)
    {
        var path = Path.Combine(_tempDir, name);
        File.WriteAllText(path, content, System.Text.Encoding.UTF8);
        return path;
    }

    [Fact]
    public async Task ValidateAsync_EmptyInputFile_ReturnsResult()
    {
        var validator = new SchemaValidator(_fileSystem, _projectDiscovery);
        var modelPath = Path.Combine(_tempDir, "model.zip");
        var inputPath = CreateCsv("empty.csv", "");

        var result = await validator.ValidateAsync(modelPath, inputPath, "default");

        // Should handle gracefully without crashing
        Assert.NotNull(result);
    }

    [Fact]
    public async Task ValidateAsync_NoModelNoSchema_ReturnsResult()
    {
        var validator = new SchemaValidator(_fileSystem, _projectDiscovery);
        var modelPath = Path.Combine(_tempDir, "nonexistent_model.zip");
        var inputPath = CreateCsv("data.csv", "Feature1,Feature2,Label\n1,2,A\n3,4,B\n");

        var result = await validator.ValidateAsync(modelPath, inputPath, "default");

        // Without model or experiment, should not crash
        Assert.NotNull(result);
    }

    [Fact]
    public void SchemaValidationResult_DefaultValues_AreCorrect()
    {
        var result = new SchemaValidationResult();

        Assert.False(result.IsValid); // default bool = false
        Assert.Null(result.ErrorMessage);
        Assert.NotNull(result.MissingColumns);
        Assert.Empty(result.MissingColumns);
        Assert.NotNull(result.Suggestions);
        Assert.Empty(result.Suggestions);
    }
}
