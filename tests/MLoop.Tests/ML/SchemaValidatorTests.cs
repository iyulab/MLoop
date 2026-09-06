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
        var inputPath = CreateCsv("empty.csv", "");

        var result = await validator.ValidateAsync(inputPath, "default");

        // Should handle gracefully without crashing
        Assert.NotNull(result);
    }

    [Fact]
    public async Task ValidateAsync_NoSavedSchema_SkipsAndSaysSo_RatherThanInventingAFinding()
    {
        // An experiment with no recorded input schema is not a data problem, and there is nothing to
        // compare the columns against. Reporting "missing columns" here would name a defect in data
        // that is fine — which is what comparing against the model artifact's own schema did: that
        // schema describes a featurized view, where the feature columns are folded into a single
        // vector and a label is present by construction, so present columns read as missing and
        // prediction data is asked for a label it is not supposed to carry.
        var validator = new SchemaValidator(_fileSystem, _projectDiscovery);
        var inputPath = CreateCsv("data.csv", "Feature1,Feature2,Label\n1,2,A\n3,4,B\n");

        var result = await validator.ValidateAsync(inputPath, "default");

        Assert.True(result.IsValid);
        Assert.Empty(result.MissingColumns);
        Assert.Contains("skipped", result.ErrorMessageEn, StringComparison.OrdinalIgnoreCase);
        Assert.NotEmpty(result.Suggestions);
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
