using MLoop.Core.Data;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// Tests for LabelValueHandler - handling missing label values.
/// </summary>
public class LabelValueHandlerTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly CsvHelperImpl _csvHelper;
    private readonly LabelValueHandler _labelHandler;

    public LabelValueHandlerTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _csvHelper = new CsvHelperImpl();
        _labelHandler = new LabelValueHandler(_csvHelper, new TestLogger());
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
    }

    [Fact]
    public async Task AnalyzeLabelColumnAsync_WithNoMissingValues_ReturnsCorrectStats()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "test.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,A\n2,B\n3,A\n4,B");

        // Act
        var result = await _labelHandler.AnalyzeLabelColumnAsync(csvPath, "Label");

        // Assert
        Assert.Equal(4, result.TotalRows);
        Assert.Equal(0, result.MissingCount);
        Assert.Equal(4, result.ValidCount);
        Assert.False(result.HasMissingValues);
        Assert.Equal(0, result.MissingPercentage);
        Assert.Equal(2, result.UniqueValueCount);
        Assert.Equal(2, result.ValueDistribution["A"]);
        Assert.Equal(2, result.ValueDistribution["B"]);
    }

    [Fact]
    public async Task AnalyzeLabelColumnAsync_WithMissingValues_ReturnsCorrectStats()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "test.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,A\n2,\n3,A\n4,\n5,B");

        // Act
        var result = await _labelHandler.AnalyzeLabelColumnAsync(csvPath, "Label");

        // Assert
        Assert.Equal(5, result.TotalRows);
        Assert.Equal(2, result.MissingCount);
        Assert.Equal(3, result.ValidCount);
        Assert.True(result.HasMissingValues);
        Assert.Equal(40.0, result.MissingPercentage);
        Assert.Equal(2, result.UniqueValueCount);
    }

    [Fact]
    public async Task AnalyzeLabelColumnAsync_WithNonExistentColumn_ReturnsError()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "test.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label\n1,A\n2,B");

        // Act
        var result = await _labelHandler.AnalyzeLabelColumnAsync(csvPath, "NonExistent");

        // Assert
        Assert.NotNull(result.Error);
        Assert.Contains("not found", result.Error);
    }

    [Fact]
    public async Task AnalyzeLabelColumnAsync_WithNonExistentFile_ThrowsException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => _labelHandler.AnalyzeLabelColumnAsync(
                Path.Combine(_tempDirectory, "nonexistent.csv"), "Label"));
    }

    [Fact]
    public async Task DropMissingLabelsAsync_WithMissingValues_DropsRows()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "input.csv");
        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        await File.WriteAllTextAsync(inputPath, "Feature,Label\n1,A\n2,\n3,A\n4,\n5,B");

        // Act
        var result = await _labelHandler.DropMissingLabelsAsync(inputPath, outputPath, "Label");

        // Assert
        Assert.True(result.Success);
        Assert.Equal(5, result.OriginalRowCount);
        Assert.Equal(2, result.DroppedRowCount);
        Assert.Equal(3, result.FinalRowCount);
        Assert.Equal(40.0, result.DroppedPercentage);
        Assert.True(File.Exists(outputPath));

        // Verify output content
        var outputData = await _csvHelper.ReadAsync(outputPath);
        Assert.Equal(3, outputData.Count);
        Assert.All(outputData, row => Assert.False(string.IsNullOrWhiteSpace(row["Label"])));
    }

    [Fact]
    public async Task DropMissingLabelsAsync_WithNoMissingValues_DoesNotModify()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "input.csv");
        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        await File.WriteAllTextAsync(inputPath, "Feature,Label\n1,A\n2,B\n3,C");

        // Act
        var result = await _labelHandler.DropMissingLabelsAsync(inputPath, outputPath, "Label");

        // Assert
        Assert.True(result.Success);
        Assert.Equal(3, result.OriginalRowCount);
        Assert.Equal(0, result.DroppedRowCount);
        Assert.Equal(3, result.FinalRowCount);
        Assert.Contains("No missing labels found", result.Message);
    }

    [Fact]
    public async Task DropMissingLabelsAsync_WithAllMissing_ReturnsError()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "input.csv");
        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        await File.WriteAllTextAsync(inputPath, "Feature,Label\n1,\n2,\n3,");

        // Act
        var result = await _labelHandler.DropMissingLabelsAsync(inputPath, outputPath, "Label");

        // Assert
        Assert.False(result.Success);
        Assert.Contains("All rows have missing labels", result.Error);
    }

    [Fact]
    public async Task DropMissingLabelsAsync_WithWhitespaceOnlyLabels_TreatsAsMissing()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "input.csv");
        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        await File.WriteAllTextAsync(inputPath, "Feature,Label\n1,A\n2,   \n3,B\n4,\t\n5,C");

        // Act
        var result = await _labelHandler.DropMissingLabelsAsync(inputPath, outputPath, "Label");

        // Assert
        Assert.True(result.Success);
        Assert.Equal(5, result.OriginalRowCount);
        Assert.Equal(2, result.DroppedRowCount);
        Assert.Equal(3, result.FinalRowCount);
    }

    [Fact]
    public async Task DropMissingLabelsAsync_WithNonExistentColumn_ReturnsError()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "input.csv");
        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        await File.WriteAllTextAsync(inputPath, "Feature,Label\n1,A\n2,B");

        // Act
        var result = await _labelHandler.DropMissingLabelsAsync(inputPath, outputPath, "NonExistent");

        // Assert
        Assert.False(result.Success);
        Assert.Contains("not found", result.Error);
    }

    [Fact]
    public async Task AnalyzeLabelColumnAsync_WithEmptyFile_ReturnsEmptyStats()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "empty.csv");
        await File.WriteAllTextAsync(csvPath, "Feature,Label");

        // Act
        var result = await _labelHandler.AnalyzeLabelColumnAsync(csvPath, "Label");

        // Assert
        Assert.Equal(0, result.TotalRows);
        Assert.Equal(0, result.MissingCount);
        Assert.False(result.HasMissingValues);
    }

    /// <summary>
    /// Simple test logger for unit tests.
    /// </summary>
    private class TestLogger : ILogger
    {
        public void Debug(string message) { }
        public void Info(string message) { }
        public void Warning(string message) { }
        public void Error(string message) { }
        public void Error(string message, Exception exception) { }
    }
}
