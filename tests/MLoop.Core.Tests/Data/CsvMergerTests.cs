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
    public async Task MergeAsync_FileLackingADefaultedColumn_TakesTheDefault()
    {
        // Training data without weights merged with weighted feedback: the unweighted rows count 1.
        var train = Path.Combine(_tempDirectory, "train.csv");
        var feedback = Path.Combine(_tempDirectory, "feedback.csv");
        await File.WriteAllTextAsync(train, "x,label\n1,a\n2,b");
        await File.WriteAllTextAsync(feedback, "x,label,weight\n3,a,4");
        var defaults = new Dictionary<string, string> { ["weight"] = "1" };

        var validation = await _csvMerger.ValidateSchemaCompatibilityAsync([train, feedback], defaults);
        var output = Path.Combine(_tempDirectory, "merged.csv");
        var result = await _csvMerger.MergeAsync([train, feedback], output, defaults);

        Assert.True(validation.IsCompatible, validation.Message);
        Assert.True(result.Success, result.Error);
        Assert.Equal(["x,label,weight", "1,a,1", "2,b,1", "3,a,4"], await File.ReadAllLinesAsync(output));
    }

    [Fact]
    public async Task MergeAsync_ColumnWithoutADefault_IsStillAMismatch()
    {
        var a = Path.Combine(_tempDirectory, "a.csv");
        var b = Path.Combine(_tempDirectory, "b.csv");
        await File.WriteAllTextAsync(a, "x,label\n1,a");
        await File.WriteAllTextAsync(b, "x,label,weight,extra\n3,a,4,z");

        var validation = await _csvMerger.ValidateSchemaCompatibilityAsync(
            [a, b], new Dictionary<string, string> { ["weight"] = "1" });

        Assert.False(validation.IsCompatible);
        Assert.Equal(["extra"], validation.MismatchedColumns["b.csv"]);
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
    public async Task MergeAsync_LaterFileInAnotherColumnOrder_LandsUnderTheFirstFilesHeader()
    {
        var first = Path.Combine(_tempDirectory, "a.csv");
        var second = Path.Combine(_tempDirectory, "b.csv");
        await File.WriteAllTextAsync(first, "x,y,label\n1,2,A\n");
        await File.WriteAllTextAsync(second, "label,x,y\nB,3,4\n");
        var output = Path.Combine(_tempDirectory, "merged.csv");

        var result = await _csvMerger.MergeAsync([first, second], output);

        Assert.True(result.Success);
        Assert.Equal(["x,y,label", "1,2,A", "3,4,B"], (await File.ReadAllLinesAsync(output)).Select(l => l.TrimStart('\uFEFF')));
        Assert.Equal(new Dictionary<string, int> { ["a.csv"] = 1, ["b.csv"] = 1 }, result.RowsPerFile);
    }

    [Fact]
    public async Task MergeAsync_FilesWithoutRows_FailsAndLeavesNoFile()
    {
        var first = Path.Combine(_tempDirectory, "a.csv");
        var second = Path.Combine(_tempDirectory, "b.csv");
        await File.WriteAllTextAsync(first, "x,y\n");
        await File.WriteAllTextAsync(second, "x,y\n");
        var output = Path.Combine(_tempDirectory, "merged.csv");

        var result = await _csvMerger.MergeAsync([first, second], output);

        Assert.False(result.Success);
        Assert.Equal(["a.csv", "b.csv"], Directory.GetFiles(_tempDirectory).Select(Path.GetFileName).Order());
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

    #region MergeWithMetadataAsync Tests

    [Fact]
    public async Task MergeWithMetadataAsync_AddsSourceColumn()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "batch_001.csv");
        var csv2 = Path.Combine(_tempDirectory, "batch_002.csv");
        var output = Path.Combine(_tempDirectory, "merged.csv");

        await File.WriteAllTextAsync(csv1, "Value\n100\n110");
        await File.WriteAllTextAsync(csv2, "Value\n200\n210");

        var options = new FilenameMetadataOptions
        {
            AddSourceColumn = true,
            SourceColumnName = "SourceFile"
        };

        // Act
        var result = await _csvMerger.MergeWithMetadataAsync([csv1, csv2], output, options);

        // Assert
        Assert.True(result.Success);
        Assert.Equal(4, result.TotalRows);

        var mergedData = await _csvHelper.ReadAsync(output);
        Assert.Equal(4, mergedData.Count);
        Assert.Contains("SourceFile", mergedData[0].Keys);
        Assert.Equal("batch_001.csv", mergedData[0]["SourceFile"]);
        Assert.Equal("batch_002.csv", mergedData[2]["SourceFile"]);
    }

    [Fact]
    public async Task MergeWithMetadataAsync_ExtractsDateFromFilename()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "sensor_2024.01.15.csv");
        var csv2 = Path.Combine(_tempDirectory, "sensor_2024.01.16.csv");
        var output = Path.Combine(_tempDirectory, "merged.csv");

        await File.WriteAllTextAsync(csv1, "Reading\n100");
        await File.WriteAllTextAsync(csv2, "Reading\n200");

        var options = new FilenameMetadataOptions
        {
            AddSourceColumn = false,
            Preset = FilenameMetadataPreset.SensorDate
        };

        // Act
        var result = await _csvMerger.MergeWithMetadataAsync([csv1, csv2], output, options);

        // Assert
        Assert.True(result.Success);

        var mergedData = await _csvHelper.ReadAsync(output);
        Assert.Equal(2, mergedData.Count);
        Assert.Contains("FileDate", mergedData[0].Keys);
        Assert.Equal("2024.01.15", mergedData[0]["FileDate"]);
        Assert.Equal("2024.01.16", mergedData[1]["FileDate"]);
    }

    [Fact]
    public async Task MergeWithMetadataAsync_WithCategoryPreset_ExtractsCategory()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "normal.csv");
        var csv2 = Path.Combine(_tempDirectory, "outlier.csv");
        var output = Path.Combine(_tempDirectory, "merged.csv");

        await File.WriteAllTextAsync(csv1, "Feature\n1.0\n1.1");
        await File.WriteAllTextAsync(csv2, "Feature\n9.0\n9.1");

        var options = new FilenameMetadataOptions
        {
            AddSourceColumn = false,
            Preset = FilenameMetadataPreset.Category
        };

        // Act
        var result = await _csvMerger.MergeWithMetadataAsync([csv1, csv2], output, options);

        // Assert
        Assert.True(result.Success);

        var mergedData = await _csvHelper.ReadAsync(output);
        Assert.Equal(4, mergedData.Count);
        Assert.Contains("Category", mergedData[0].Keys);
        Assert.Equal("normal", mergedData[0]["Category"]);
        Assert.Equal("outlier", mergedData[2]["Category"]);
    }

    [Fact]
    public async Task MergeWithMetadataAsync_WithCustomPattern_ExtractsMetadata()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "machine_A1_001.csv");
        var csv2 = Path.Combine(_tempDirectory, "machine_B2_002.csv");
        var output = Path.Combine(_tempDirectory, "merged.csv");

        await File.WriteAllTextAsync(csv1, "Value\n100");
        await File.WriteAllTextAsync(csv2, "Value\n200");

        var options = new FilenameMetadataOptions
        {
            AddSourceColumn = false,
            CustomPatterns = new Dictionary<string, string>
            {
                ["MachineId"] = @"machine_([A-Z]\d+)_",
                ["BatchNum"] = @"_(\d+)\.csv$"
            }
        };

        // Act
        var result = await _csvMerger.MergeWithMetadataAsync([csv1, csv2], output, options);

        // Assert
        Assert.True(result.Success);

        var mergedData = await _csvHelper.ReadAsync(output);
        Assert.Equal(2, mergedData.Count);
        Assert.Contains("MachineId", mergedData[0].Keys);
        Assert.Contains("BatchNum", mergedData[0].Keys);
        Assert.Equal("A1", mergedData[0]["MachineId"]);
        Assert.Equal("001", mergedData[0]["BatchNum"]);
        Assert.Equal("B2", mergedData[1]["MachineId"]);
        Assert.Equal("002", mergedData[1]["BatchNum"]);
    }

    [Fact]
    public async Task MergeWithMetadataAsync_WithSchemaMismatch_ReturnsError()
    {
        // Arrange
        var csv1 = Path.Combine(_tempDirectory, "file1.csv");
        var csv2 = Path.Combine(_tempDirectory, "file2.csv");
        var output = Path.Combine(_tempDirectory, "merged.csv");

        await File.WriteAllTextAsync(csv1, "A,B\n1,2");
        await File.WriteAllTextAsync(csv2, "X,Y\n3,4"); // Different schema

        var options = new FilenameMetadataOptions();

        // Act
        var result = await _csvMerger.MergeWithMetadataAsync([csv1, csv2], output, options);

        // Assert
        Assert.False(result.Success);
        Assert.NotNull(result.Error);
    }

    #endregion
}
