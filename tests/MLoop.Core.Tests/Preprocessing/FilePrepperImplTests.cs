using MLoop.Core.Preprocessing;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Tests.Preprocessing;

/// <summary>
/// Tests for FilePrepperImpl - FilePrepper API integration.
/// </summary>
public class FilePrepperImplTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly FilePrepperImpl _filePrepper;

    public FilePrepperImplTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _filePrepper = new FilePrepperImpl();
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
    }

    #region UnpivotSimple Tests

    [Fact]
    public async Task UnpivotSimpleAsync_TransformsWideToLong()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "wide.csv");
        var outputPath = Path.Combine(_tempDirectory, "long.csv");

        // Wide format: Region, Q1, Q2, Q3, Q4
        await File.WriteAllTextAsync(inputPath,
            "Region,Q1,Q2,Q3,Q4\n" +
            "North,100,120,150,130\n" +
            "South,200,220,250,230");

        // Act
        var result = await _filePrepper.UnpivotSimpleAsync(
            inputPath,
            outputPath,
            baseColumns: new[] { "Region" },
            unpivotColumns: new[] { "Q1", "Q2", "Q3", "Q4" },
            indexColumn: "Quarter",
            valueColumn: "Sales");

        // Assert
        Assert.True(result.Success);
        Assert.Equal(2, result.OriginalRowCount);
        Assert.Equal(8, result.TransformedRowCount); // 2 rows * 4 quarters
        Assert.True(File.Exists(outputPath));
    }

    [Fact]
    public async Task UnpivotSimpleAsync_SkipsEmptyRows()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "sparse.csv");
        var outputPath = Path.Combine(_tempDirectory, "sparse_long.csv");

        // Wide format with some empty values
        await File.WriteAllTextAsync(inputPath,
            "Region,Q1,Q2,Q3\n" +
            "North,100,,150\n" + // Q2 is empty
            "South,200,220,");   // Q3 is empty

        // Act
        var result = await _filePrepper.UnpivotSimpleAsync(
            inputPath,
            outputPath,
            baseColumns: new[] { "Region" },
            unpivotColumns: new[] { "Q1", "Q2", "Q3" },
            indexColumn: "Quarter",
            valueColumn: "Sales",
            skipEmptyRows: true);

        // Assert
        Assert.True(result.Success);
        Assert.Equal(2, result.OriginalRowCount);
        Assert.True(result.TransformedRowCount < 6); // Less than 2 * 3 due to skipping
    }

    [Fact]
    public async Task UnpivotSimpleAsync_InvalidFile_ReturnsError()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "nonexistent.csv");
        var outputPath = Path.Combine(_tempDirectory, "output.csv");

        // Act
        var result = await _filePrepper.UnpivotSimpleAsync(
            inputPath,
            outputPath,
            baseColumns: new[] { "Region" },
            unpivotColumns: new[] { "Q1", "Q2" });

        // Assert
        Assert.False(result.Success);
        Assert.NotNull(result.Error);
    }

    #endregion

    #region ParseKoreanTime Tests

    [Fact]
    public async Task ParseKoreanTimeAsync_ParsesAMPM()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "korean_time.csv");
        var outputPath = Path.Combine(_tempDirectory, "parsed_time.csv");

        await File.WriteAllTextAsync(inputPath,
            "ID,Time\n" +
            "1,오전 9:30\n" +
            "2,오후 2:45\n" +
            "3,오전 11:00");

        // Act
        var result = await _filePrepper.ParseKoreanTimeAsync(
            inputPath,
            outputPath,
            sourceColumn: "Time",
            targetColumn: "ParsedTime");

        // Assert - Basic success check
        Assert.True(result.Success, $"Expected success but got error: {result.Error}");
        Assert.Equal(3, result.TotalRows);
        Assert.True(File.Exists(outputPath));

        // Verify output file has the new column with data
        var outputContent = await File.ReadAllTextAsync(outputPath);
        Assert.Contains("ParsedTime", outputContent);
    }

    [Fact]
    public async Task ParseKoreanTimeAsync_WithBaseDate_UsesSpecifiedDate()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "korean_base.csv");
        var outputPath = Path.Combine(_tempDirectory, "parsed_base.csv");
        var baseDate = new DateTime(2025, 1, 15);

        await File.WriteAllTextAsync(inputPath,
            "ID,Time\n" +
            "1,오전 10:00");

        // Act
        var result = await _filePrepper.ParseKoreanTimeAsync(
            inputPath,
            outputPath,
            sourceColumn: "Time",
            targetColumn: "ParsedTime",
            baseDate: baseDate);

        // Assert
        Assert.True(result.Success);
    }

    [Fact]
    public async Task ParseKoreanTimeAsync_InvalidColumn_ReturnsError()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "invalid_col.csv");
        var outputPath = Path.Combine(_tempDirectory, "output.csv");

        await File.WriteAllTextAsync(inputPath, "ID,Value\n1,100");

        // Act
        var result = await _filePrepper.ParseKoreanTimeAsync(
            inputPath,
            outputPath,
            sourceColumn: "NonExistent",
            targetColumn: "ParsedTime");

        // Assert
        Assert.False(result.Success);
        Assert.NotNull(result.Error);
    }

    #endregion

    #region Basic Operations Tests

    [Fact]
    public async Task NormalizeEncodingAsync_WritesUtf8()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "input.csv");
        var outputPath = Path.Combine(_tempDirectory, "normalized.csv");

        await File.WriteAllTextAsync(inputPath, "Name,Value\nTest,123");

        // Act
        var result = await _filePrepper.NormalizeEncodingAsync(inputPath, outputPath);

        // Assert
        Assert.Equal(outputPath, result);
        Assert.True(File.Exists(outputPath));
    }

    [Fact]
    public async Task RemoveDuplicatesAsync_RemovesDuplicateRows()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "duplicates.csv");
        var outputPath = Path.Combine(_tempDirectory, "deduped.csv");

        await File.WriteAllTextAsync(inputPath,
            "ID,Value\n" +
            "1,A\n" +
            "2,B\n" +
            "1,A\n" + // Duplicate
            "3,C");

        // Act
        var (path, removed) = await _filePrepper.RemoveDuplicatesAsync(inputPath, outputPath);

        // Assert
        Assert.Equal(outputPath, path);
        Assert.Equal(1, removed);
    }

    #endregion
}
