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
        Assert.True(dataView.GetRowCount() == 3, $"Row count is {dataView.GetRowCount()}");
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
        var exception = Assert.Throws<InvalidOperationException>(
            () => _loader.LoadData(csvPath, "nonexistent_label"));

        Assert.Contains("not found in the data", exception.Message);
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
        // Arrange
        var csvPath = CreateTestCsv(new[]
        {
            "feature,label",
            "1.0,0.0",
            "2.0,1.0",
            "3.0,0.0",
            "4.0,1.0",
            "5.0,0.0"
        });
        var dataView = _loader.LoadData(csvPath, "label");

        // Act
        var (trainSet, testSet) = _loader.SplitData(dataView, testFraction: 0.2);

        // Assert
        Assert.NotNull(trainSet);
        Assert.NotNull(testSet);

        var trainCount = trainSet.GetRowCount();
        var testCount = testSet.GetRowCount();

        Assert.True(trainCount > 0);
        Assert.True(testCount > 0);
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
        Assert.Equal(3, trainSet.GetRowCount());
        Assert.Equal(3, testSet.GetRowCount());
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

    private string CreateTestCsv(string[] lines)
    {
        var fileName = $"test_{Guid.NewGuid()}.csv";
        var filePath = Path.Combine(_tempDirectory, fileName);
        File.WriteAllLines(filePath, lines);
        return filePath;
    }
}
