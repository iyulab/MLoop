using MLoop.Core.Preprocessing;
using MLoop.Extensibility;
using MLoop.Extensibility.Preprocessing;

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

    [Fact]
    public async Task ExecuteAsync_WithSingleScript_ExecutesAndReturnsOutputPath()
    {
        // Arrange
        _engine.InitializeDirectory();
        var scriptPath = Path.Combine(_engine.GetPreprocessingDirectory(), "01_simple.cs");
        var scriptContent = @"
using System.IO;
using System.Threading.Tasks;
using MLoop.Extensibility.Preprocessing;

public class SimpleScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        var data = await context.Csv.ReadAsync(context.InputPath);

        // Add a new column
        foreach (var row in data)
        {
            row[""Processed""] = ""true"";
        }

        var outputPath = Path.Combine(context.OutputDirectory, ""01_simple.csv"");
        await context.Csv.WriteAsync(outputPath, data);
        return outputPath;
    }
}";
        File.WriteAllText(scriptPath, scriptContent);

        var inputPath = Path.Combine(_tempProjectRoot, "input.csv");
        var csvContent = @"Name,Value
Alice,100
Bob,200";
        File.WriteAllText(inputPath, csvContent);

        // Act
        var result = await _engine.ExecuteAsync(inputPath);

        // Assert
        Assert.NotEqual(inputPath, result);
        Assert.True(File.Exists(result));

        var outputContent = await File.ReadAllTextAsync(result);
        Assert.Contains("Processed", outputContent);
        Assert.Contains("true", outputContent);
    }

    [Fact]
    public async Task ExecuteAsync_WithMultipleScripts_ExecutesInOrder()
    {
        // Arrange
        _engine.InitializeDirectory();

        // Script 1: Add column A
        var script1Path = Path.Combine(_engine.GetPreprocessingDirectory(), "01_add_a.cs");
        var script1Content = @"
using System.IO;
using System.Threading.Tasks;
using MLoop.Extensibility.Preprocessing;

public class AddAScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        var data = await context.Csv.ReadAsync(context.InputPath);
        foreach (var row in data)
        {
            row[""ColumnA""] = ""A"";
        }
        var outputPath = Path.Combine(context.OutputDirectory, ""01_add_a.csv"");
        await context.Csv.WriteAsync(outputPath, data);
        return outputPath;
    }
}";
        File.WriteAllText(script1Path, script1Content);

        // Script 2: Add column B
        var script2Path = Path.Combine(_engine.GetPreprocessingDirectory(), "02_add_b.cs");
        var script2Content = @"
using System.IO;
using System.Threading.Tasks;
using MLoop.Extensibility.Preprocessing;

public class AddBScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        var data = await context.Csv.ReadAsync(context.InputPath);
        foreach (var row in data)
        {
            row[""ColumnB""] = ""B"";
        }
        var outputPath = Path.Combine(context.OutputDirectory, ""02_add_b.csv"");
        await context.Csv.WriteAsync(outputPath, data);
        return outputPath;
    }
}";
        File.WriteAllText(script2Path, script2Content);

        var inputPath = Path.Combine(_tempProjectRoot, "input.csv");
        File.WriteAllText(inputPath, "Name\nTest");

        // Act
        var result = await _engine.ExecuteAsync(inputPath);

        // Assert
        Assert.True(File.Exists(result));
        var outputContent = await File.ReadAllTextAsync(result);
        Assert.Contains("ColumnA", outputContent);
        Assert.Contains("ColumnB", outputContent);

        // Verify execution order in logs
        Assert.Contains(_logger.InfoMessages, m => m.Contains("[1/2]"));
        Assert.Contains(_logger.InfoMessages, m => m.Contains("[2/2]"));
    }

    [Fact]
    public async Task ExecuteAsync_WithFailingScript_ThrowsException()
    {
        // Arrange
        _engine.InitializeDirectory();
        var scriptPath = Path.Combine(_engine.GetPreprocessingDirectory(), "01_failing.cs");
        var scriptContent = @"
using System;
using System.Threading.Tasks;
using MLoop.Extensibility.Preprocessing;

public class FailingScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext context)
    {
        throw new InvalidOperationException(""Script intentionally fails"");
    }
}";
        File.WriteAllText(scriptPath, scriptContent);

        var inputPath = Path.Combine(_tempProjectRoot, "input.csv");
        File.WriteAllText(inputPath, "data");

        // Act & Assert
        var exception = await Assert.ThrowsAsync<InvalidOperationException>(
            () => _engine.ExecuteAsync(inputPath));

        Assert.Contains("01_failing", exception.Message);
    }

    [Fact]
    public async Task ExecuteAsync_WithNonIPreprocessingScript_LogsWarning()
    {
        // Arrange
        _engine.InitializeDirectory();
        var scriptPath = Path.Combine(_engine.GetPreprocessingDirectory(), "01_invalid.cs");
        var scriptContent = @"
public class NotAPreprocessingScript
{
    public string SomeMethod() => ""test"";
}";
        File.WriteAllText(scriptPath, scriptContent);

        var inputPath = Path.Combine(_tempProjectRoot, "input.csv");
        File.WriteAllText(inputPath, "data");

        // Act
        var result = await _engine.ExecuteAsync(inputPath);

        // Assert
        Assert.Equal(inputPath, result); // No preprocessing
        Assert.Contains(_logger.WarningMessages,
            m => m.Contains("No IPreprocessingScript implementations"));
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
        public void Error(string message, Exception exception) => ErrorMessages.Add($"{message}{Environment.NewLine}{exception}");
        public void Debug(string message) => DebugMessages.Add(message);
    }
}
