using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// Tests for CsvHelperImpl - CSV reading/writing for preprocessing scripts.
/// </summary>
public class CsvHelperImplTests : IDisposable
{
    private readonly string _tempDirectory;
    private readonly CsvHelperImpl _csvHelper;

    public CsvHelperImplTests()
    {
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"MLoopTests_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempDirectory);
        _csvHelper = new CsvHelperImpl();
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
    }

    [Fact]
    public async Task ReadAsync_WithValidCsv_ReturnsData()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "test.csv");
        var csvContent = @"Name,Age,City
Alice,25,Seoul
Bob,30,Busan
Charlie,35,Incheon";
        await File.WriteAllTextAsync(csvPath, csvContent);

        // Act
        var data = await _csvHelper.ReadAsync(csvPath);

        // Assert
        Assert.Equal(3, data.Count);
        Assert.Equal("Alice", data[0]["Name"]);
        Assert.Equal("25", data[0]["Age"]);
        Assert.Equal("Seoul", data[0]["City"]);
        Assert.Equal("Bob", data[1]["Name"]);
        Assert.Equal("30", data[1]["Age"]);
        Assert.Equal("Busan", data[1]["City"]);
    }

    [Fact]
    public async Task ReadAsync_WithNonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "nonexistent.csv");

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() => _csvHelper.ReadAsync(csvPath));
    }

    [Fact]
    public async Task ReadAsync_WithCommaFormattedNumbers_CleansNumbers()
    {
        // Arrange - Korean-style number formatting with commas (quoted to preserve)
        var csvPath = Path.Combine(_tempDirectory, "numbers.csv");
        var csvContent = @"Product,Price,Quantity
Laptop,""1,500,000"",10
Phone,""800,000"",20
Tablet,""500,000.50"",15";
        await File.WriteAllTextAsync(csvPath, csvContent);

        // Act
        var data = await _csvHelper.ReadAsync(csvPath);

        // Assert
        Assert.Equal("1500000", data[0]["Price"]);  // Commas removed
        Assert.Equal("800000", data[1]["Price"]);
        Assert.Equal("500000.50", data[2]["Price"]);  // Decimal preserved
        Assert.Equal("10", data[0]["Quantity"]);
    }

    [Fact]
    public async Task ReadAsync_WithEmptyFields_ReturnsEmptyStrings()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "empty_fields.csv");
        var csvContent = @"Name,Age,City
Alice,25,
Bob,,Busan
,,";
        await File.WriteAllTextAsync(csvPath, csvContent);

        // Act
        var data = await _csvHelper.ReadAsync(csvPath);

        // Assert
        Assert.Equal(3, data.Count);
        Assert.Equal("", data[0]["City"]);
        Assert.Equal("", data[1]["Age"]);
        Assert.Equal("", data[2]["Name"]);
        Assert.Equal("", data[2]["Age"]);
        Assert.Equal("", data[2]["City"]);
    }

    [Fact]
    public async Task WriteAsync_WithValidData_CreatesFile()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "output.csv");
        var data = new List<Dictionary<string, string>>
        {
            new() { ["Name"] = "Alice", ["Age"] = "25", ["City"] = "Seoul" },
            new() { ["Name"] = "Bob", ["Age"] = "30", ["City"] = "Busan" }
        };

        // Act
        var result = await _csvHelper.WriteAsync(csvPath, data);

        // Assert
        Assert.True(File.Exists(csvPath));
        Assert.Equal(Path.GetFullPath(csvPath), result);

        var content = await File.ReadAllTextAsync(csvPath);
        Assert.Contains("Name,Age,City", content);
        Assert.Contains("Alice,25,Seoul", content);
        Assert.Contains("Bob,30,Busan", content);
    }

    [Fact]
    public async Task WriteAsync_WithNullData_ThrowsArgumentException()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "output.csv");

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _csvHelper.WriteAsync(csvPath, null!));
    }

    [Fact]
    public async Task WriteAsync_WithEmptyData_ThrowsArgumentException()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "output.csv");
        var data = new List<Dictionary<string, string>>();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _csvHelper.WriteAsync(csvPath, data));
    }

    [Fact]
    public async Task WriteAsync_CreatesDirectoryIfNotExists()
    {
        // Arrange
        var subDir = Path.Combine(_tempDirectory, "subdir", "nested");
        var csvPath = Path.Combine(subDir, "output.csv");
        var data = new List<Dictionary<string, string>>
        {
            new() { ["Name"] = "Alice", ["Age"] = "25" }
        };

        // Act
        await _csvHelper.WriteAsync(csvPath, data);

        // Assert
        Assert.True(Directory.Exists(subDir));
        Assert.True(File.Exists(csvPath));
    }

    [Fact]
    public async Task WriteAsync_WithMissingColumns_FillsEmptyStrings()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "partial.csv");
        var data = new List<Dictionary<string, string>>
        {
            new() { ["Name"] = "Alice", ["Age"] = "25", ["City"] = "Seoul" },
            new() { ["Name"] = "Bob", ["Age"] = "30" },  // Missing City
            new() { ["Name"] = "Charlie" }  // Missing Age and City
        };

        // Act
        await _csvHelper.WriteAsync(csvPath, data);

        // Assert
        var content = await File.ReadAllTextAsync(csvPath);
        Assert.Contains("Name,Age,City", content);
        Assert.Contains("Alice,25,Seoul", content);
        Assert.Contains("Bob,30,", content);  // Empty City
        Assert.Contains("Charlie,,", content);  // Empty Age and City
    }

    [Fact]
    public async Task ReadHeadersAsync_ReturnsColumnNames()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "headers.csv");
        var csvContent = @"Name,Age,City,Country
Alice,25,Seoul,Korea
Bob,30,Busan,Korea";
        await File.WriteAllTextAsync(csvPath, csvContent);

        // Act
        var headers = await _csvHelper.ReadHeadersAsync(csvPath);

        // Assert
        Assert.Equal(4, headers.Count);
        Assert.Equal("Name", headers[0]);
        Assert.Equal("Age", headers[1]);
        Assert.Equal("City", headers[2]);
        Assert.Equal("Country", headers[3]);
    }

    [Fact]
    public async Task ReadHeadersAsync_WithNonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "nonexistent.csv");

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _csvHelper.ReadHeadersAsync(csvPath));
    }

    [Fact]
    public async Task ReadAsync_WriteAsync_RoundTrip_PreservesData()
    {
        // Arrange
        var inputPath = Path.Combine(_tempDirectory, "input.csv");
        var outputPath = Path.Combine(_tempDirectory, "output.csv");
        var originalData = new List<Dictionary<string, string>>
        {
            new() { ["Name"] = "Alice", ["Age"] = "25", ["City"] = "Seoul" },
            new() { ["Name"] = "Bob", ["Age"] = "30", ["City"] = "Busan" },
            new() { ["Name"] = "Charlie", ["Age"] = "35", ["City"] = "Incheon" }
        };

        // Act - Write then Read
        await _csvHelper.WriteAsync(inputPath, originalData);
        var readData = await _csvHelper.ReadAsync(inputPath);
        await _csvHelper.WriteAsync(outputPath, readData);
        var finalData = await _csvHelper.ReadAsync(outputPath);

        // Assert - Data preserved through round trip
        Assert.Equal(originalData.Count, finalData.Count);
        for (int i = 0; i < originalData.Count; i++)
        {
            Assert.Equal(originalData[i]["Name"], finalData[i]["Name"]);
            Assert.Equal(originalData[i]["Age"], finalData[i]["Age"]);
            Assert.Equal(originalData[i]["City"], finalData[i]["City"]);
        }
    }

    [Fact]
    public async Task ReadAsync_WithWhitespace_TrimsValues()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "whitespace.csv");
        var csvContent = @"Name,Age,City
 Alice ,  25  ,  Seoul
  Bob  ,30,Busan";
        await File.WriteAllTextAsync(csvPath, csvContent);

        // Act
        var data = await _csvHelper.ReadAsync(csvPath);

        // Assert - Whitespace trimmed
        Assert.Equal("Alice", data[0]["Name"]);
        Assert.Equal("25", data[0]["Age"]);
        Assert.Equal("Seoul", data[0]["City"]);
        Assert.Equal("Bob", data[1]["Name"]);
    }

    [Fact]
    public async Task ReadAsync_WithSpecialCharacters_PreservesCharacters()
    {
        // Arrange
        var csvPath = Path.Combine(_tempDirectory, "special.csv");
        var csvContent = "Name,Description\n" +
                         "Alice,\"Contains \"\"quotes\"\"\"\n" +
                         "Bob,\"Has, comma\"\n" +
                         "Charlie,Normal text";
        await File.WriteAllTextAsync(csvPath, csvContent);

        // Act
        var data = await _csvHelper.ReadAsync(csvPath);

        // Assert
        Assert.Equal(3, data.Count);
        Assert.Equal("Alice", data[0]["Name"]);
        Assert.Contains("quotes", data[0]["Description"]);
        Assert.Equal("Bob", data[1]["Name"]);
        Assert.Contains("comma", data[1]["Description"]);
        Assert.Equal("Charlie", data[2]["Name"]);
    }
}
