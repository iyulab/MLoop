using MLoop.Core.Diagnostics;

namespace MLoop.Core.Tests.Diagnostics;

/// <summary>
/// Tests for UnusedDataScanner - unused data file detection.
/// </summary>
public class UnusedDataScannerTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly UnusedDataScanner _scanner;

    public UnusedDataScannerTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _scanner = new UnusedDataScanner();
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void Scan_AllFilesUsed_ReturnsNoUnused()
    {
        // Arrange
        var file1 = Path.Combine(_tempDirectory, "data1.csv");
        var file2 = Path.Combine(_tempDirectory, "data2.csv");
        File.WriteAllText(file1, "A,B\n1,2");
        File.WriteAllText(file2, "A,B\n3,4");

        // Act
        var result = _scanner.Scan(_tempDirectory, new[] { file1, file2 });

        // Assert
        Assert.False(result.HasUnusedFiles);
        Assert.Empty(result.UnusedFiles);
        Assert.Equal(2, result.TotalCsvFiles);
    }

    [Fact]
    public void Scan_SomeFilesUnused_ReturnsUnusedFiles()
    {
        // Arrange
        var file1 = Path.Combine(_tempDirectory, "train.csv");
        var file2 = Path.Combine(_tempDirectory, "unused.csv");
        File.WriteAllText(file1, "A,B\n1,2");
        File.WriteAllText(file2, "A,B\n3,4");

        // Act
        var result = _scanner.Scan(_tempDirectory, new[] { file1 });

        // Assert
        Assert.True(result.HasUnusedFiles);
        Assert.Single(result.UnusedFiles);
        Assert.Equal("unused.csv", result.UnusedFiles[0].FileName);
    }

    [Fact]
    public void Scan_CategorizesBackupFiles()
    {
        // Arrange
        var train = Path.Combine(_tempDirectory, "train.csv");
        var backup = Path.Combine(_tempDirectory, "backup_data.csv");
        File.WriteAllText(train, "A,B\n1,2");
        File.WriteAllText(backup, "A,B\n3,4");

        // Act
        var result = _scanner.Scan(_tempDirectory, new[] { train });

        // Assert
        Assert.Single(result.UnusedFiles);
        Assert.Equal(UnusedFileCategory.Backup, result.UnusedFiles[0].Category);
    }

    [Fact]
    public void Scan_CategorizesTempFiles()
    {
        // Arrange
        var train = Path.Combine(_tempDirectory, "train.csv");
        var temp = Path.Combine(_tempDirectory, "temp_processing.csv");
        File.WriteAllText(train, "A,B\n1,2");
        File.WriteAllText(temp, "A,B\n3,4");

        // Act
        var result = _scanner.Scan(_tempDirectory, new[] { train });

        // Assert
        Assert.Single(result.UnusedFiles);
        Assert.Equal(UnusedFileCategory.Temporary, result.UnusedFiles[0].Category);
    }

    [Fact]
    public void Scan_CategorizesMergedFiles()
    {
        // Arrange
        var train = Path.Combine(_tempDirectory, "train.csv");
        var merged = Path.Combine(_tempDirectory, "merged_output.csv");
        File.WriteAllText(train, "A,B\n1,2");
        File.WriteAllText(merged, "A,B\n3,4");

        // Act
        var result = _scanner.Scan(_tempDirectory, new[] { train });

        // Assert
        Assert.Single(result.UnusedFiles);
        Assert.Equal(UnusedFileCategory.MergedOutput, result.UnusedFiles[0].Category);
    }

    [Fact]
    public void Scan_DetectsReservedFilesNotUsed()
    {
        // Arrange - train.csv exists but not used
        var train = Path.Combine(_tempDirectory, "train.csv");
        var other = Path.Combine(_tempDirectory, "other.csv");
        File.WriteAllText(train, "A,B\n1,2");
        File.WriteAllText(other, "A,B\n3,4");

        // Act - using 'other.csv' but not 'train.csv'
        var result = _scanner.Scan(_tempDirectory, new[] { other });

        // Assert
        Assert.Single(result.UnusedFiles);
        Assert.Equal(UnusedFileCategory.ReservedNotUsed, result.UnusedFiles[0].Category);
        Assert.Contains(result.Warnings, w => w.Contains("Standard data file"));
    }

    [Fact]
    public void Scan_NonExistentDirectory_ReturnsError()
    {
        // Act
        var result = _scanner.Scan(Path.Combine(_tempDirectory, "nonexistent"), Array.Empty<string>());

        // Assert
        Assert.NotNull(result.Error);
        Assert.Contains("not found", result.Error);
    }

    [Fact]
    public void Scan_ManyUnusedFiles_GeneratesWarning()
    {
        // Arrange - create more than 5 unused files
        var train = Path.Combine(_tempDirectory, "train.csv");
        File.WriteAllText(train, "A,B\n1,2");

        for (int i = 0; i < 10; i++)
        {
            File.WriteAllText(Path.Combine(_tempDirectory, $"data_{i}.csv"), "A,B\n1,2");
        }

        // Act
        var result = _scanner.Scan(_tempDirectory, new[] { train });

        // Assert
        Assert.Contains(result.Warnings, w => w.Contains("Large number"));
    }

    [Fact]
    public void Scan_UnusedFilesOrderedBySize()
    {
        // Arrange
        var train = Path.Combine(_tempDirectory, "train.csv");
        var small = Path.Combine(_tempDirectory, "small.csv");
        var large = Path.Combine(_tempDirectory, "large.csv");

        File.WriteAllText(train, "A,B\n1,2");
        File.WriteAllText(small, "A,B\n1,2");
        File.WriteAllText(large, "A,B\n" + string.Join("\n", Enumerable.Range(1, 1000).Select(i => $"{i},{i}")));

        // Act
        var result = _scanner.Scan(_tempDirectory, new[] { train });

        // Assert
        Assert.Equal(2, result.UnusedFiles.Count);
        Assert.True(result.UnusedFiles[0].SizeBytes >= result.UnusedFiles[1].SizeBytes);
    }

    [Fact]
    public void Scan_CalculatesCorrectFileSizes()
    {
        // Arrange
        var train = Path.Combine(_tempDirectory, "train.csv");
        var data = Path.Combine(_tempDirectory, "data.csv");
        var content = "A,B,C,D,E\n" + string.Join("\n", Enumerable.Range(1, 100).Select(i => $"{i},{i},{i},{i},{i}"));

        File.WriteAllText(train, "A,B\n1,2");
        File.WriteAllText(data, content);

        // Act
        var result = _scanner.Scan(_tempDirectory, new[] { train });

        // Assert
        Assert.Single(result.UnusedFiles);
        Assert.True(result.UnusedFiles[0].SizeBytes > 0);
        Assert.NotEmpty(result.UnusedFiles[0].SizeFormatted);
    }

    [Fact]
    public void Scan_SuggestsMergeForMultipleUnknownFiles()
    {
        // Arrange
        var train = Path.Combine(_tempDirectory, "train.csv");
        File.WriteAllText(train, "A,B\n1,2");

        // Create multiple "unknown" category files
        File.WriteAllText(Path.Combine(_tempDirectory, "batch1.csv"), "A,B\n1,2");
        File.WriteAllText(Path.Combine(_tempDirectory, "batch2.csv"), "A,B\n3,4");
        File.WriteAllText(Path.Combine(_tempDirectory, "batch3.csv"), "A,B\n5,6");

        // Act
        var result = _scanner.Scan(_tempDirectory, new[] { train });

        // Assert
        Assert.Contains(result.Suggestions, s => s.Contains("--auto-merge"));
    }

    [Fact]
    public void ScanRecursive_ScansSubdirectories()
    {
        // Arrange
        var subDir = Path.Combine(_tempDirectory, "subdir");
        Directory.CreateDirectory(subDir);

        var train = Path.Combine(_tempDirectory, "train.csv");
        var subFile = Path.Combine(subDir, "subdata.csv");

        File.WriteAllText(train, "A,B\n1,2");
        File.WriteAllText(subFile, "A,B\n3,4");

        // Act
        var result = _scanner.ScanRecursive(_tempDirectory, new[] { train });

        // Assert
        Assert.True(result.HasUnusedFiles);
        Assert.Contains(result.UnusedFiles, f => f.FileName == "subdata.csv");
    }

    [Fact]
    public void Scan_IgnoresNonCsvFiles()
    {
        // Arrange
        var train = Path.Combine(_tempDirectory, "train.csv");
        var readme = Path.Combine(_tempDirectory, "README.txt");
        var json = Path.Combine(_tempDirectory, "config.json");

        File.WriteAllText(train, "A,B\n1,2");
        File.WriteAllText(readme, "This is a readme");
        File.WriteAllText(json, "{}");

        // Act
        var result = _scanner.Scan(_tempDirectory, new[] { train });

        // Assert
        Assert.False(result.HasUnusedFiles);
        Assert.Equal(1, result.TotalCsvFiles);
    }
}
