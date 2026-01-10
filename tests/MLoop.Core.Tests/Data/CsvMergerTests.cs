using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// Tests for CsvMerger - CSV file merging with schema detection.
/// </summary>
public class CsvMergerTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly CsvHelperImpl _csvHelper;
    private readonly CsvMerger _csvMerger;

    public CsvMergerTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _csvHelper = new CsvHelperImpl();
        _csvMerger = new CsvMerger(_csvHelper);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
    }

    [Fact]
    public async Task DiscoverMergeableCsvsAsync_WithSameSchemaFiles_ReturnsGroup()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "normal.csv");
        var csv2 = Path.Combine(_tempDirectory, "outlier.csv");

        await File.WriteAllTextAsync(csv1, "Feature1,Feature2,Label\n1.0,2.0,normal\n1.1,2.1,normal");
        await File.WriteAllTextAsync(csv2, "Feature1,Feature2,Label\n5.0,6.0,outlier\n5.1,6.1,outlier");

        // Act
        var groups = await _csvMerger.DiscoverMergeableCsvsAsync(_tempDirectory);

        // Assert
        Assert.Single(groups);
        Assert.Equal(2, groups[0].FilePaths.Count);
        Assert.Equal(3, groups[0].Columns.Count);
        Assert.Equal("normal_outlier", groups[0].DetectedPattern);
        Assert.True(groups[0].PatternConfidence >= 0.8);
    }

    [Fact]
    public async Task DiscoverMergeableCsvsAsync_WithDifferentSchemas_ReturnsSeparateGroups()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "data1.csv");
        var csv2 = Path.Combine(_tempDirectory, "data2.csv");

        await File.WriteAllTextAsync(csv1, "A,B,C\n1,2,3");
        await File.WriteAllTextAsync(csv2, "X,Y,Z\n4,5,6");

        // Act
        var groups = await _csvMerger.DiscoverMergeableCsvsAsync(_tempDirectory);

        // Assert
        Assert.Empty(groups); // Each has only 1 file, need 2+ to form a group
    }

    [Fact]
    public async Task DiscoverMergeableCsvsAsync_ExcludesReservedFiles()
    {
        // Arrange - should exclude train.csv, validation.csv, test.csv, predict.csv
        var train = Path.Combine(_tempDirectory, "train.csv");
        var data1 = Path.Combine(_tempDirectory, "data1.csv");
        var data2 = Path.Combine(_tempDirectory, "data2.csv");

        await File.WriteAllTextAsync(train, "A,B,C\n1,2,3");
        await File.WriteAllTextAsync(data1, "A,B,C\n4,5,6");
        await File.WriteAllTextAsync(data2, "A,B,C\n7,8,9");

        // Act
        var groups = await _csvMerger.DiscoverMergeableCsvsAsync(_tempDirectory);

        // Assert
        Assert.Single(groups);
        Assert.Equal(2, groups[0].FilePaths.Count);
        Assert.DoesNotContain(groups[0].FilePaths, p => Path.GetFileName(p).Equals("train.csv", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task MergeAsync_WithValidFiles_CombinesData()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "file1.csv");
        var csv2 = Path.Combine(_tempDirectory, "file2.csv");
        var output = Path.Combine(_tempDirectory, "merged.csv");

        await File.WriteAllTextAsync(csv1, "Name,Value\nAlice,100\nBob,200");
        await File.WriteAllTextAsync(csv2, "Name,Value\nCharlie,300\nDave,400");

        // Act
        var result = await _csvMerger.MergeAsync([csv1, csv2], output);

        // Assert
        Assert.True(result.Success);
        Assert.Equal(4, result.TotalRows);
        Assert.Equal(2, result.SourceFileCount);
        Assert.Equal(2, result.RowsPerFile["file1.csv"]);
        Assert.Equal(2, result.RowsPerFile["file2.csv"]);
        Assert.True(File.Exists(output));

        // Verify merged content
        var mergedData = await _csvHelper.ReadAsync(output);
        Assert.Equal(4, mergedData.Count);
    }

    [Fact]
    public async Task MergeAsync_WithSchemaMismatch_ReturnsError()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "file1.csv");
        var csv2 = Path.Combine(_tempDirectory, "file2.csv");
        var output = Path.Combine(_tempDirectory, "merged.csv");

        await File.WriteAllTextAsync(csv1, "A,B,C\n1,2,3");
        await File.WriteAllTextAsync(csv2, "X,Y\n4,5"); // Different schema

        // Act
        var result = await _csvMerger.MergeAsync([csv1, csv2], output);

        // Assert
        Assert.False(result.Success);
        Assert.NotNull(result.Error);
        Assert.Contains("mismatch", result.Error, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task MergeAsync_WithNoFiles_ReturnsError()
    {
        // Act
        var result = await _csvMerger.MergeAsync([], Path.Combine(_tempDirectory, "output.csv"));

        // Assert
        Assert.False(result.Success);
        Assert.NotNull(result.Error);
    }

    [Fact]
    public async Task ValidateSchemaCompatibilityAsync_WithCompatibleFiles_ReturnsSuccess()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "file1.csv");
        var csv2 = Path.Combine(_tempDirectory, "file2.csv");

        await File.WriteAllTextAsync(csv1, "A,B,C\n1,2,3");
        await File.WriteAllTextAsync(csv2, "A,B,C\n4,5,6");

        // Act
        var result = await _csvMerger.ValidateSchemaCompatibilityAsync([csv1, csv2]);

        // Assert
        Assert.True(result.IsCompatible);
        Assert.Equal(3, result.CommonColumns.Count);
        Assert.Empty(result.MismatchedColumns);
    }

    [Fact]
    public async Task ValidateSchemaCompatibilityAsync_WithIncompatibleFiles_ReturnsFailure()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "file1.csv");
        var csv2 = Path.Combine(_tempDirectory, "file2.csv");

        await File.WriteAllTextAsync(csv1, "A,B,C\n1,2,3");
        await File.WriteAllTextAsync(csv2, "A,B,D\n4,5,6"); // Different column C vs D

        // Act
        var result = await _csvMerger.ValidateSchemaCompatibilityAsync([csv1, csv2]);

        // Assert
        Assert.False(result.IsCompatible);
        Assert.NotEmpty(result.MismatchedColumns);
    }

    [Fact]
    public async Task DiscoverMergeableCsvsAsync_WithSequencePattern_DetectsPattern()
    {
        // Arrange - use pattern that matches [_-](\d+)[_.]
        var csv1 = Path.Combine(_tempDirectory, "data_1_part.csv");
        var csv2 = Path.Combine(_tempDirectory, "data_2_part.csv");
        var csv3 = Path.Combine(_tempDirectory, "data_3_part.csv");

        await File.WriteAllTextAsync(csv1, "A,B\n1,2");
        await File.WriteAllTextAsync(csv2, "A,B\n3,4");
        await File.WriteAllTextAsync(csv3, "A,B\n5,6");

        // Act
        var groups = await _csvMerger.DiscoverMergeableCsvsAsync(_tempDirectory);

        // Assert
        Assert.Single(groups);
        Assert.Equal(3, groups[0].FilePaths.Count);
        Assert.Equal("sequence", groups[0].DetectedPattern);
    }

    [Fact]
    public async Task DiscoverMergeableCsvsAsync_WithDatePattern_DetectsPattern()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "log_2024-01-01.csv");
        var csv2 = Path.Combine(_tempDirectory, "log_2024-01-02.csv");

        await File.WriteAllTextAsync(csv1, "Time,Value\n10:00,100");
        await File.WriteAllTextAsync(csv2, "Time,Value\n10:00,200");

        // Act
        var groups = await _csvMerger.DiscoverMergeableCsvsAsync(_tempDirectory);

        // Assert
        Assert.Single(groups);
        Assert.Equal(2, groups[0].FilePaths.Count);
        Assert.Equal("date_series", groups[0].DetectedPattern);
    }
}
