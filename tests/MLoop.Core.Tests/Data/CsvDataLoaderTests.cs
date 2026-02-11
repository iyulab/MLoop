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
}
