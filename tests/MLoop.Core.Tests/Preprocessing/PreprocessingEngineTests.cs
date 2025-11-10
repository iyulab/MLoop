using MLoop.Core.Preprocessing;
using MLoop.Extensibility;

namespace MLoop.Core.Tests.Preprocessing;

public class PreprocessingEngineTests : IDisposable
{
    private readonly string _tempProjectRoot;
    private readonly TestLogger _logger;
    private readonly PreprocessingEngine _engine;

    public PreprocessingEngineTests()
    {
        _tempProjectRoot = Path.Combine(Path.GetTempPath(), $"MLoopTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempProjectRoot);
        _logger = new TestLogger();
        _engine = new PreprocessingEngine(_tempProjectRoot, _logger);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempProjectRoot))
        {
            Directory.Delete(_tempProjectRoot, recursive: true);
        }
    }

    [Fact]
    public void GetPreprocessingDirectory_ReturnsCorrectPath()
    {
        // Act
        var path = _engine.GetPreprocessingDirectory();

        // Assert
        var expected = Path.Combine(_tempProjectRoot, ".mloop", "scripts", "preprocess");
        Assert.Equal(expected, path);
    }

    [Fact]
    public void InitializeDirectory_CreatesDirectory()
    {
        // Act
        _engine.InitializeDirectory();

        // Assert
        var expectedPath = Path.Combine(_tempProjectRoot, ".mloop", "scripts", "preprocess");
        Assert.True(Directory.Exists(expectedPath));
    }

    [Fact]
    public void HasPreprocessingScripts_WithNoDirectory_ReturnsFalse()
    {
        // Act
        var hasScripts = _engine.HasPreprocessingScripts();

        // Assert
        Assert.False(hasScripts);
    }

    [Fact]
    public void HasPreprocessingScripts_WithEmptyDirectory_ReturnsFalse()
    {
        // Arrange
        _engine.InitializeDirectory();

        // Act
        var hasScripts = _engine.HasPreprocessingScripts();

        // Assert
        Assert.False(hasScripts);
    }

    [Fact]
    public void HasPreprocessingScripts_WithScripts_ReturnsTrue()
    {
        // Arrange
        _engine.InitializeDirectory();
        var scriptPath = Path.Combine(_engine.GetPreprocessingDirectory(), "01_test.cs");
        File.WriteAllText(scriptPath, "// test script");

        // Act
        var hasScripts = _engine.HasPreprocessingScripts();

        // Assert
        Assert.True(hasScripts);
    }

    [Fact]
    public async Task ExecuteAsync_WithNoDirectory_ReturnsOriginalPath()
    {
        // Arrange
        var inputPath = Path.Combine(_tempProjectRoot, "input.csv");
        File.WriteAllText(inputPath, "data");

        // Act
        var result = await _engine.ExecuteAsync(inputPath);

        // Assert
        Assert.Equal(inputPath, result);
    }

    [Fact]
    public async Task ExecuteAsync_WithEmptyDirectory_ReturnsOriginalPath()
    {
        // Arrange
        _engine.InitializeDirectory();
        var inputPath = Path.Combine(_tempProjectRoot, "input.csv");
        File.WriteAllText(inputPath, "data");

        // Act
        var result = await _engine.ExecuteAsync(inputPath);

        // Assert
        Assert.Equal(inputPath, result);
    }

    [Fact]
    public void CleanupTempFiles_WithNonExistentDirectory_DoesNotThrow()
    {
        // Act & Assert - should not throw
        _engine.CleanupTempFiles();
    }

    [Fact]
    public void CleanupTempFiles_WithExistingDirectory_RemovesDirectory()
    {
        // Arrange
        var tempDir = Path.Combine(_tempProjectRoot, ".mloop", "temp", "preprocess");
        Directory.CreateDirectory(tempDir);
        var testFile = Path.Combine(tempDir, "test.csv");
        File.WriteAllText(testFile, "data");

        // Act
        _engine.CleanupTempFiles();

        // Assert
        Assert.False(Directory.Exists(tempDir));
    }

    private class TestLogger : ILogger
    {
        public List<string> InfoMessages { get; } = new();
        public List<string> WarningMessages { get; } = new();
        public List<string> ErrorMessages { get; } = new();
        public List<string> DebugMessages { get; } = new();

        public void Info(string message) => InfoMessages.Add(message);
        public void Warning(string message) => WarningMessages.Add(message);
        public void Error(string message) => ErrorMessages.Add(message);
        public void Debug(string message) => DebugMessages.Add(message);
    }
}
