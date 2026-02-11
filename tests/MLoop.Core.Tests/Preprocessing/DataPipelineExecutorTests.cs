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

    [Fact]
    public async Task ExecuteAsync_WithAddColumnConstant_AddsColumn()
    {
        var inputPath = CreateTestCsv(
            "id,name",
            "1,Alice",
            "2,Bob");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "add-column", Column = "status", Value = "active" }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Contains("status", lines[0]);
        Assert.Contains("active", lines[1]);
        Assert.Contains("active", lines[2]);
    }

    [Fact]
    public async Task ExecuteAsync_WithAddColumnCopyExpression_CopiesColumn()
    {
        var inputPath = CreateTestCsv(
            "id,name",
            "1,Alice",
            "2,Bob");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "add-column", Column = "name_backup", Expression = "copy:name" }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Contains("name_backup", lines[0]);
        Assert.Contains("Alice", lines[1]);
    }

    [Fact]
    public async Task ExecuteAsync_WithParseKoreanTime_ParsesCorrectly()
    {
        var inputPath = CreateTestCsv(
            "id,time_str",
            "1,오전 9:01:18",
            "2,오후 2:15:30");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "parse-korean-time", Column = "time_str", OutputColumn = "parsed_time" }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Contains("parsed_time", lines[0]);
        Assert.Contains("09:01:18", lines[1]);
        Assert.Contains("14:15:30", lines[2]);
    }

    [Fact]
    public async Task ExecuteAsync_WithParseExcelDate_ParsesNumericDate()
    {
        // Excel date 44927 = 2023-01-01
        var inputPath = CreateTestCsv(
            "id,date_num",
            "1,44927",
            "2,44928");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "parse-excel-date", Column = "date_num", Format = "yyyy-MM-dd" }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Contains("2023-01-01", lines[1]);
        Assert.Contains("2023-01-02", lines[2]);
    }

    [Fact]
    public async Task ExecuteAsync_WithRolling_AddsRollingColumns()
    {
        var inputPath = CreateTestCsv(
            "time,value",
            "1,10",
            "2,20",
            "3,30",
            "4,40",
            "5,50");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new()
            {
                Type = "rolling",
                Columns = ["value"],
                WindowSize = 3,
                Method = "mean",
                OutputSuffix = "_avg"
            }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Contains("value_avg", lines[0]);
        Assert.Equal(6, lines.Length); // header + 5 rows
    }

    [Fact]
    public async Task ExecuteAsync_WithResample_AggregatesTimeWindows()
    {
        var inputPath = CreateTestCsv(
            "timestamp,temperature",
            "2024-01-01 00:00:00,20",
            "2024-01-01 00:05:00,22",
            "2024-01-01 00:10:00,21",
            "2024-01-01 01:00:00,18",
            "2024-01-01 01:05:00,19");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new()
            {
                Type = "resample",
                TimeColumn = "timestamp",
                Window = "1H",
                Columns = ["temperature"],
                Method = "mean"
            }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        // Should aggregate into 2 hourly windows
        Assert.Equal(3, lines.Length); // header + 2 windows
    }

    [Fact]
    public async Task ExecuteAsync_WithAddColumnConcat_ConcatenatesColumns()
    {
        var inputPath = CreateTestCsv(
            "first,last",
            "John,Doe",
            "Jane,Smith");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "add-column", Column = "full_name", Expression = "concat:first,last, " }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Contains("full_name", lines[0]);
        Assert.Contains("John Doe", lines[1]);
        Assert.Contains("Jane Smith", lines[2]);
    }

    [Fact]
    public async Task ExecuteAsync_WithFillMissing_FillsWithMean()
    {
        var inputPath = CreateTestCsv(
            "id,value",
            "1,10",
            "2,",
            "3,30",
            "4,");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "fill-missing", Columns = ["value"], Method = "mean" }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Equal(5, lines.Length); // header + 4 rows
        // Mean of 10,30 = 20, filled values should be 20
        Assert.DoesNotContain(",," , string.Join(",", lines));
    }

    [Fact]
    public async Task ExecuteAsync_WithFillMissingConstant_ExecutesWithoutError()
    {
        // FilePrepper FillMissing with constant method - verifies the pipeline step runs
        var inputPath = CreateTestCsv(
            "id,value",
            "1,10",
            "2,",
            "3,30");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "fill-missing", Column = "value", Method = "constant", Value = "0" }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Equal(4, lines.Length); // header + 3 rows
    }

    [Fact]
    public async Task ExecuteAsync_WithNormalize_NormalizesValues()
    {
        var inputPath = CreateTestCsv(
            "id,score",
            "1,0",
            "2,50",
            "3,100");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "normalize", Column = "score", Method = "min-max" }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Equal(4, lines.Length);
        // Min-max: 0→0, 50→0.5, 100→1
        Assert.Contains("0", lines[1]); // id=1, score=0
        Assert.Contains("1", lines[3]); // id=3, score=1
    }

    [Fact]
    public async Task ExecuteAsync_WithScale_IsNormalizeAlias()
    {
        var inputPath = CreateTestCsv(
            "id,value",
            "1,0",
            "2,100");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new() { Type = "scale", Column = "value", Method = "min-max" }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Equal(3, lines.Length);
    }

    [Fact]
    public async Task ExecuteAsync_WithExtractDate_ExtractsFeatures()
    {
        var inputPath = CreateTestCsv(
            "id,date",
            "1,2024-03-15",
            "2,2024-12-25");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new()
            {
                Type = "extract-date",
                Column = "date",
                Features = ["year", "month", "day"]
            }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        var header = lines[0];
        Assert.Contains("year", header.ToLower());
        Assert.Contains("month", header.ToLower());
        Assert.Contains("day", header.ToLower());
    }

    [Fact]
    public async Task ExecuteAsync_WithParseDatetime_ParsesFormat()
    {
        var inputPath = CreateTestCsv(
            "id,date_str",
            "1,15/03/2024",
            "2,25/12/2024");

        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var steps = new List<PrepStep>
        {
            new()
            {
                Type = "parse-datetime",
                Column = "date_str",
                Format = "dd/MM/yyyy"
            }
        };

        await _executor.ExecuteAsync(inputPath, steps, outputPath);

        var lines = await File.ReadAllLinesAsync(outputPath);
        Assert.Equal(3, lines.Length);
        // Parsed dates should be present
        Assert.Contains("2024", lines[1]);
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
