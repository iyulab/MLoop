using MLoop.Core.Preprocessing;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Tests.Preprocessing;

public class DataPipelineExecutorTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly DataPipelineExecutor _executor;

    public DataPipelineExecutorTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopPrepTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _executor = new DataPipelineExecutor(new TestLogger());
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
    }

    [Fact]
    public async Task ExecuteAsync_WithRemoveColumns_RemovesSpecifiedColumns()
    {
        // Arrange
        var inputPath = CreateTestCsv(
            "id,name,age,score",
            "1,Alice,30,90.5",
            "2,Bob,25,85.0",
            "3,Carol,35,92.3");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "remove-columns", Columns = ["name"] }
        };

        // Act
        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        // Assert
        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Equal(4, lines.Length); // header + 3 rows
        Assert.DoesNotContain("name", lines[0]);
        Assert.Contains("id", lines[0]);
        Assert.Contains("age", lines[0]);
        Assert.Contains("score", lines[0]);
    }

    [Fact]
    public async Task ExecuteAsync_WithDropDuplicates_RemovesDuplicateRows()
    {
        // Arrange
        var inputPath = CreateTestCsv(
            "id,value",
            "1,A",
            "2,B",
            "1,A",
            "3,C");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "drop-duplicates" }
        };

        // Act
        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        // Assert
        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Equal(4, lines.Length); // header + 3 unique rows
    }

    [Fact]
    public async Task ExecuteAsync_WithRenameColumns_RenamesCorrectly()
    {
        // Arrange
        var inputPath = CreateTestCsv(
            "old_name,value",
            "A,1",
            "B,2");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new()
            {
                Type = "rename-columns",
                Mapping = new Dictionary<string, string> { ["old_name"] = "new_name" }
            }
        };

        // Act
        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        // Assert
        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Contains("new_name", lines[0]);
        Assert.DoesNotContain("old_name", lines[0]);
    }

    [Fact]
    public async Task ExecuteAsync_WithMultipleSteps_ExecutesSequentially()
    {
        // Arrange
        var inputPath = CreateTestCsv(
            "id,name,value",
            "1,Alice,10",
            "2,Bob,20",
            "1,Alice,10",
            "3,Carol,30");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "drop-duplicates" },
            new() { Type = "remove-columns", Columns = ["name"] }
        };

        // Act
        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        // Assert
        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Equal(4, lines.Length); // header + 3 unique rows
        Assert.DoesNotContain("name", lines[0]);
    }

    [Fact]
    public async Task ExecuteAsync_WithFilterRows_FiltersCorrectly()
    {
        // Arrange
        var inputPath = CreateTestCsv(
            "name,score",
            "Alice,90",
            "Bob,60",
            "Carol,85",
            "Dave,45");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new()
            {
                Type = "filter-rows",
                Column = "score",
                Operator = ">=",
                Value = "80"
            }
        };

        // Act
        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        // Assert
        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Equal(3, lines.Length); // header + Alice + Carol
    }

    [Fact]
    public async Task ExecuteAsync_WithUnknownStepType_ThrowsInvalidOperationException()
    {
        // Arrange
        var inputPath = CreateTestCsv("id,value", "1,A");
        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "nonexistent-step" }
        };

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => _executor.ExecuteAsync(inputPath, steps, outputPath));
    }

    [Fact]
    public async Task ExecuteAsync_WithEmptySteps_CopiesDataUnchanged()
    {
        // Arrange
        var inputPath = CreateTestCsv(
            "id,value",
            "1,A",
            "2,B");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>();

        // Act
        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        // Assert
        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Equal(3, lines.Length); // header + 2 rows
    }

    private string CreateTestCsv(params string[] lines)
    {
        var path = Path.Combine(_tempDirectory, $"test_{Guid.NewGuid()}.csv");
        File.WriteAllLines(path, lines);
        return path;
    }

    private class TestLogger : ILogger
    {
        public void Debug(string message) { }
        public void Info(string message) { }
        public void Warning(string message) { }
        public void Error(string message) { }
        public void Error(string message, Exception exception) { }
    }
}
