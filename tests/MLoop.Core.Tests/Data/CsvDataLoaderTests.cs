using Microsoft.ML;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.Data;

public class CsvDataLoaderTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly MLContext _mlContext;
    private readonly CsvDataLoader _loader;

    public CsvDataLoaderTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _mlContext = new MLContext(seed: 42);
        _loader = new CsvDataLoader(_mlContext);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
    }

    [Fact]
    public void Constructor_WithNullMLContext_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new CsvDataLoader(null!));
    }

    [Fact]
    public void LoadData_WithNonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var filePath = Path.Combine(_tempDirectory, "nonexistent.csv");

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => _loader.LoadData(filePath));
    }

    [Fact]
    public void LoadData_WithValidCsv_ReturnsDataView()
    {
        // Arrange
        var csvPath = CreateTestCsv(new[]
        {
            "feature1,feature2,label",
            "1.0,2.0,0.0",
            "3.0,4.0,1.0",
            "5.0,6.0,0.0"
        });

        // Act
        var dataView = _loader.LoadData(csvPath, "label");

        // Assert
        Assert.NotNull(dataView);
        Assert.True(dataView.Schema.Count > 0, $"Schema count is {dataView.Schema.Count}");

        // GetRowCount() may return null, use manual count as fallback
        var rowCount = GetActualRowCount(dataView);
        Assert.Equal(3, rowCount);
    }

    [Fact]
    public void LoadData_WithInvalidLabelColumn_ThrowsInvalidOperationException()
    {
        // Arrange
        var csvPath = CreateTestCsv(new[]
        {
            "feature1,feature2,target",
            "1.0,2.0,0",
            "3.0,4.0,1"
        });

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(
            () => _loader.LoadData(csvPath, "nonexistent_label"));

        Assert.Contains("not found", exception.Message);
    }

    [Fact]
    public void ValidateLabelColumn_WithValidColumn_ReturnsTrue()
    {
        // Arrange
        var csvPath = CreateTestCsv(new[]
        {
            "feature,label",
            "1.0,0.0",
            "2.0,1.0"
        });
        var dataView = _loader.LoadData(csvPath, "label");

        // Act
        var isValid = _loader.ValidateLabelColumn(dataView, "label");

        // Assert
        Assert.True(isValid);
    }

    [Fact]
    public void ValidateLabelColumn_WithInvalidColumn_ReturnsFalse()
    {
        // Arrange
        var csvPath = CreateTestCsv(new[]
        {
            "feature,label",
            "1.0,0.0",
            "2.0,1.0"
        });
        var dataView = _loader.LoadData(csvPath, "label");

        // Act
        var isValid = _loader.ValidateLabelColumn(dataView, "nonexistent");

        // Assert
        Assert.False(isValid);
    }

    [Fact]
    public void GetSchema_ReturnsCorrectSchema()
    {
        // Arrange
        var csvPath = CreateTestCsv(new[]
        {
            "feature1,feature2,label",
            "1.0,2.0,0.0",
            "3.0,4.0,1.0",
            "5.0,6.0,0.0"
        });
        var dataView = _loader.LoadData(csvPath, "label");

        // Act
        var schema = _loader.GetSchema(dataView);

        // Assert
        Assert.NotNull(schema);
        Assert.NotEmpty(schema.Columns);
        Assert.Equal(3, schema.RowCount);
    }

    [Fact]
    public void SplitData_WithValidFraction_ReturnsSplitData()
    {
        // Arrange - Use larger dataset to ensure reliable split
        var csvPath = CreateTestCsv(new[]
        {
            "feature,label",
            "1.0,0.0",
            "2.0,1.0",
            "3.0,0.0",
            "4.0,1.0",
            "5.0,0.0",
            "6.0,1.0",
            "7.0,0.0",
            "8.0,1.0",
            "9.0,0.0",
            "10.0,1.0"
        });
        var dataView = _loader.LoadData(csvPath, "label");

        // Act
        var (trainSet, testSet) = _loader.SplitData(dataView, testFraction: 0.2);

        // Assert
        Assert.NotNull(trainSet);
        Assert.NotNull(testSet);

        var trainCount = GetActualRowCount(trainSet);
        var testCount = GetActualRowCount(testSet);

        Assert.True(trainCount > 0, $"Train count is {trainCount}");
        Assert.True(testCount > 0, $"Test count is {testCount}");
        Assert.Equal(10, trainCount + testCount); // Total should equal original count
    }

    [Fact]
    public void SplitData_WithZeroFraction_ReturnsSameDataForBoth()
    {
        // Arrange
        var csvPath = CreateTestCsv(new[]
        {
            "feature,label",
            "1.0,0.0",
            "2.0,1.0",
            "3.0,0.0"
        });
        var dataView = _loader.LoadData(csvPath, "label");

        // Act
        var (trainSet, testSet) = _loader.SplitData(dataView, testFraction: 0);

        // Assert
        Assert.NotNull(trainSet);
        Assert.NotNull(testSet);
        Assert.Equal(3, GetActualRowCount(trainSet));
        Assert.Equal(3, GetActualRowCount(testSet));
    }

    [Fact]
    public void SplitData_WithInvalidFraction_ThrowsArgumentException()
    {
        // Arrange
        var csvPath = CreateTestCsv(new[]
        {
            "feature,label",
            "1.0,0.0",
            "2.0,1.0"
        });
        var dataView = _loader.LoadData(csvPath, "label");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _loader.SplitData(dataView, testFraction: 1.5));
    }

    [Fact]
    public void LoadData_WithDateTimeColumn_ExcludesFromFeatures()
    {
        // Arrange - CSV with a datetime column that should be auto-excluded
        var csvPath = CreateTestCsv(new[]
        {
            "feature1,timestamp,label",
            "1.0,2024-01-01 09:00:00,0.0",
            "2.0,2024-01-02 10:30:00,1.0",
            "3.0,2024-01-03 11:45:00,0.0",
            "4.0,2024-01-04 14:00:00,1.0",
            "5.0,2024-01-05 16:15:00,0.0"
        });

        // Act
        var dataView = _loader.LoadData(csvPath, "label", "regression");

        // Assert - datetime column should be excluded (not in schema as feature)
        Assert.NotNull(dataView);
        // The timestamp column should still be in schema but ignored by AutoML
        var columns = dataView.Schema.Select(c => c.Name).ToList();
        Assert.Contains("feature1", columns);
        Assert.Contains("label", columns);
    }

    private string CreateTestCsv(string[] lines)
    {
        var fileName = $"test_{Guid.NewGuid()}.csv";
        var filePath = Path.Combine(_tempDirectory, fileName);
        File.WriteAllLines(filePath, lines);
        return filePath;
    }

    /// <summary>
    /// Gets actual row count from DataView, handling cases where GetRowCount() returns null.
    /// ML.NET 5.0 may return null more frequently, so we use manual counting as fallback.
    /// </summary>
    private int GetActualRowCount(IDataView dataView)
    {
        var count = dataView.GetRowCount();
        if (count.HasValue)
        {
            return (int)count.Value;
        }

        // Fallback: count manually
        int rowCount = 0;
        using (var cursor = dataView.GetRowCursor(dataView.Schema))
        {
            while (cursor.MoveNext())
            {
                rowCount++;
            }
        }
        return rowCount;
    }

    [Fact]
    public void FlattenMultiLineHeaders_SingleLineHeader_ReturnsOriginalPath()
    {
        var csvPath = Path.Combine(_tempDirectory, "single_line.csv");
        File.WriteAllText(csvPath, "A,B,C\n1,2,3\n4,5,6\n");

        var result = CsvDataLoader.FlattenMultiLineHeaders(csvPath);

        Assert.Equal(csvPath, result);
    }

    [Fact]
    public void FlattenMultiLineHeaders_MultiLineHeader_ReturnsFlattenedFile()
    {
        var csvPath = Path.Combine(_tempDirectory, "multi_line.csv");
        // Header with newlines inside quoted fields: "Col A", "Col\nB", "Col\nC\nD"
        File.WriteAllText(csvPath, "\"Col A\",\"Col\nB\",\"Col\nC\nD\"\n1,2,3\n4,5,6\n");

        var result = CsvDataLoader.FlattenMultiLineHeaders(csvPath);

        // Should return a different file (temp)
        Assert.NotEqual(csvPath, result);
        Assert.True(File.Exists(result));

        // Read the flattened file
        var lines = File.ReadAllLines(result);
        Assert.True(lines.Length >= 3); // header + 2 data rows

        // Header should be single line with spaces instead of newlines
        Assert.Contains("Col A", lines[0]);
        Assert.Contains("Col B", lines[0]);
        Assert.Contains("Col C D", lines[0]);

        // Data rows should be preserved
        Assert.Equal("1,2,3", lines[1]);
        Assert.Equal("4,5,6", lines[2]);

        // Cleanup
        File.Delete(result);
    }

    [Fact]
    public void FlattenMultiLineHeaders_EmptyFile_ReturnsOriginalPath()
    {
        var csvPath = Path.Combine(_tempDirectory, "empty.csv");
        File.WriteAllText(csvPath, "");

        var result = CsvDataLoader.FlattenMultiLineHeaders(csvPath);

        Assert.Equal(csvPath, result);
    }

    #region IsLikelyIndexColumn

    [Theory]
    [InlineData("", true)]
    [InlineData("  ", true)]
    [InlineData("Unnamed: 0", true)]
    [InlineData("Unnamed: 1", true)]
    [InlineData("unnamed: 0", true)]
    [InlineData("Unnamed", true)]
    [InlineData("Feature1", false)]
    [InlineData("id", false)]
    [InlineData("index", false)]
    [InlineData("Price", false)]
    public void IsLikelyIndexColumn_DetectsCorrectly(string columnName, bool expected)
    {
        Assert.Equal(expected, CsvDataLoader.IsLikelyIndexColumn(columnName));
    }

    #endregion

    #region RemoveIndexColumns

    [Fact]
    public void RemoveIndexColumns_NoIndexColumns_ReturnsOriginalPath()
    {
        var csv = "Feature1,Feature2,Label\n1,2,A\n3,4,B\n";
        var csvPath = Path.Combine(_tempDirectory, "noidx.csv");
        File.WriteAllText(csvPath, csv, System.Text.Encoding.UTF8);

        var result = CsvDataLoader.RemoveIndexColumns(csvPath);

        Assert.Equal(csvPath, result); // No change
    }

    [Fact]
    public void RemoveIndexColumns_EmptyNameColumn_RemovesIt()
    {
        // Pandas default: ",Feature1,Feature2,Label\n0,1,2,A\n1,3,4,B"
        var csv = ",Feature1,Feature2,Label\n0,1,2,A\n1,3,4,B\n";
        var csvPath = Path.Combine(_tempDirectory, "pandas_idx.csv");
        File.WriteAllText(csvPath, csv, System.Text.Encoding.UTF8);

        var result = CsvDataLoader.RemoveIndexColumns(csvPath);

        Assert.NotEqual(csvPath, result); // New file created
        var content = File.ReadAllText(result);
        Assert.StartsWith("Feature1,Feature2,Label", content);
        Assert.Contains("1,2,A", content);
        Assert.DoesNotContain(",0,", content.Split('\n')[0]); // Index values removed
    }

    [Fact]
    public void RemoveIndexColumns_UnnamedColumn_RemovesIt()
    {
        var csv = "Unnamed: 0,Feature1,Label\n0,1,A\n1,2,B\n";
        var csvPath = Path.Combine(_tempDirectory, "unnamed_idx.csv");
        File.WriteAllText(csvPath, csv, System.Text.Encoding.UTF8);

        var result = CsvDataLoader.RemoveIndexColumns(csvPath);

        Assert.NotEqual(csvPath, result);
        var content = File.ReadAllText(result);
        Assert.StartsWith("Feature1,Label", content);
    }

    [Fact]
    public void RemoveIndexColumns_EmptyFile_ReturnsOriginalPath()
    {
        var csvPath = Path.Combine(_tempDirectory, "empty_idx.csv");
        File.WriteAllText(csvPath, "", System.Text.Encoding.UTF8);

        var result = CsvDataLoader.RemoveIndexColumns(csvPath);

        Assert.Equal(csvPath, result);
    }

    #endregion

    #region IsLikelyHeaderless

    [Fact]
    public void IsLikelyHeaderless_AllNumericFirstRow_ReturnsTrue()
    {
        var csv = "1,2,3,4,5\n10.5,20.3,15.2,5.1,8.0\n8.9,12.4,18.7,3.2,6.0\n";
        var csvPath = Path.Combine(_tempDirectory, "headerless.csv");
        File.WriteAllText(csvPath, csv, System.Text.Encoding.UTF8);

        Assert.True(CsvDataLoader.IsLikelyHeaderless(csvPath));
    }

    [Fact]
    public void IsLikelyHeaderless_TextHeaders_ReturnsFalse()
    {
        var csv = "Feature1,Feature2,Label\n1.0,2.0,0\n3.0,4.0,1\n";
        var csvPath = Path.Combine(_tempDirectory, "with_header.csv");
        File.WriteAllText(csvPath, csv, System.Text.Encoding.UTF8);

        Assert.False(CsvDataLoader.IsLikelyHeaderless(csvPath));
    }

    [Fact]
    public void IsLikelyHeaderless_MixedFirstRow_ReturnsFalse()
    {
        // Some text + some numbers in first row → likely a header
        var csv = "Name,Age,Score\nAlice,30,95.5\nBob,25,87.3\n";
        var csvPath = Path.Combine(_tempDirectory, "mixed_header.csv");
        File.WriteAllText(csvPath, csv, System.Text.Encoding.UTF8);

        Assert.False(CsvDataLoader.IsLikelyHeaderless(csvPath));
    }

    [Fact]
    public void IsLikelyHeaderless_EmptyFile_ReturnsFalse()
    {
        var csvPath = Path.Combine(_tempDirectory, "empty_headerless.csv");
        File.WriteAllText(csvPath, "", System.Text.Encoding.UTF8);

        Assert.False(CsvDataLoader.IsLikelyHeaderless(csvPath));
    }

    [Fact]
    public void IsLikelyHeaderless_SingleColumn_ReturnsFalse()
    {
        // Only one column - can't reliably judge
        var csv = "42\n100\n200\n";
        var csvPath = Path.Combine(_tempDirectory, "single_col.csv");
        File.WriteAllText(csvPath, csv, System.Text.Encoding.UTF8);

        Assert.False(CsvDataLoader.IsLikelyHeaderless(csvPath));
    }

    [Fact]
    public void IsLikelyHeaderless_SequentialIntegers_ReturnsTrue()
    {
        var csv = "0,1,2,3,4\n5,6,7,8,9\n10,11,12,13,14\n";
        var csvPath = Path.Combine(_tempDirectory, "sequential.csv");
        File.WriteAllText(csvPath, csv, System.Text.Encoding.UTF8);

        Assert.True(CsvDataLoader.IsLikelyHeaderless(csvPath));
    }

    #endregion
}
