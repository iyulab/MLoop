using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Infrastructure.FileSystem;

public class DatasetDiscoveryTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly IFileSystemManager _fileSystem;
    private readonly DatasetDiscovery _datasetDiscovery;

    public DatasetDiscoveryTests()
    {
        // Create temporary test directory
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testProjectRoot);

        _fileSystem = new FileSystemManager();
        _datasetDiscovery = new DatasetDiscovery(_fileSystem);
    }

    public void Dispose()
    {
        // Cleanup test directory
        if (Directory.Exists(_testProjectRoot))
        {
            Directory.Delete(_testProjectRoot, recursive: true);
        }
    }

    [Fact]
    public void FindDirectoryDataset_ObjectDetection_PrefersCocoThenYolo()
    {
        // datasets/coco wins over datasets/yolo and datasets/ for object detection.
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "datasets", "yolo"));
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "datasets", "coco"));

        var result = DatasetDiscovery.FindDirectoryDataset(_testProjectRoot, "object-detection");

        Assert.Equal(Path.Combine(_testProjectRoot, "datasets", "coco"), result);
    }

    [Fact]
    public void FindDirectoryDataset_ObjectDetection_FallsBackToYolo()
    {
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "datasets", "yolo"));

        var result = DatasetDiscovery.FindDirectoryDataset(_testProjectRoot, "object-detection");

        Assert.Equal(Path.Combine(_testProjectRoot, "datasets", "yolo"), result);
    }

    [Fact]
    public void FindDirectoryDataset_ImageClassification_PrefersImagesDir()
    {
        // The image-classification convention is datasets/images, not coco/yolo.
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "datasets", "images"));
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "datasets", "coco"));

        var result = DatasetDiscovery.FindDirectoryDataset(_testProjectRoot, "image-classification");

        Assert.Equal(Path.Combine(_testProjectRoot, "datasets", "images"), result);
    }

    [Fact]
    public void FindDirectoryDataset_FallsBackToDatasets()
    {
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, "datasets"));

        var result = DatasetDiscovery.FindDirectoryDataset(_testProjectRoot, "image-classification");

        Assert.Equal(Path.Combine(_testProjectRoot, "datasets"), result);
    }

    [Fact]
    public void FindDirectoryDataset_NoDirectory_ReturnsNull()
    {
        var result = DatasetDiscovery.FindDirectoryDataset(_testProjectRoot, "object-detection");

        Assert.Null(result);
    }

    [Fact]
    public void FindDatasets_WithTrainCsv_ReturnsDatasetPaths()
    {
        // Arrange
        var datasetsDir = Path.Combine(_testProjectRoot, "datasets");
        Directory.CreateDirectory(datasetsDir);

        var trainPath = Path.Combine(datasetsDir, "train.csv");
        File.WriteAllText(trainPath, "col1,col2\n1,2\n");

        // Act
        var result = _datasetDiscovery.FindDatasets(_testProjectRoot);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(trainPath, result.TrainPath);
        Assert.Null(result.ValidationPath);
        Assert.Null(result.TestPath);
        Assert.Null(result.PredictPath);
    }

    [Fact]
    public void FindDatasets_WithAllFiles_ReturnsAllPaths()
    {
        // Arrange
        var datasetsDir = Path.Combine(_testProjectRoot, "datasets");
        Directory.CreateDirectory(datasetsDir);

        var trainPath = Path.Combine(datasetsDir, "train.csv");
        var validationPath = Path.Combine(datasetsDir, "validation.csv");
        var testPath = Path.Combine(datasetsDir, "test.csv");
        var predictPath = Path.Combine(datasetsDir, "predict.csv");

        File.WriteAllText(trainPath, "col1,col2\n1,2\n");
        File.WriteAllText(validationPath, "col1,col2\n3,4\n");
        File.WriteAllText(testPath, "col1,col2\n5,6\n");
        File.WriteAllText(predictPath, "col1,col2\n7,8\n");

        // Act
        var result = _datasetDiscovery.FindDatasets(_testProjectRoot);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(trainPath, result.TrainPath);
        Assert.Equal(validationPath, result.ValidationPath);
        Assert.Equal(testPath, result.TestPath);
        Assert.Equal(predictPath, result.PredictPath);
    }

    [Fact]
    public void FindDatasets_WithoutDatasetsDirectory_ReturnsNull()
    {
        // Act
        var result = _datasetDiscovery.FindDatasets(_testProjectRoot);

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void FindDatasets_WithoutTrainCsv_ReturnsNull()
    {
        // Arrange
        var datasetsDir = Path.Combine(_testProjectRoot, "datasets");
        Directory.CreateDirectory(datasetsDir);

        var testPath = Path.Combine(datasetsDir, "test.csv");
        File.WriteAllText(testPath, "col1,col2\n1,2\n");

        // Act
        var result = _datasetDiscovery.FindDatasets(_testProjectRoot);

        // Assert
        Assert.Null(result); // train.csv is mandatory
    }

    [Fact]
    public void HasDatasetsDirectory_WhenExists_ReturnsTrue()
    {
        // Arrange
        var datasetsDir = Path.Combine(_testProjectRoot, "datasets");
        Directory.CreateDirectory(datasetsDir);

        // Act
        var result = _datasetDiscovery.HasDatasetsDirectory(_testProjectRoot);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void HasDatasetsDirectory_WhenNotExists_ReturnsFalse()
    {
        // Act
        var result = _datasetDiscovery.HasDatasetsDirectory(_testProjectRoot);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GetDatasetsPath_ReturnsCorrectPath()
    {
        // Act
        var result = _datasetDiscovery.GetDatasetsPath(_testProjectRoot);

        // Assert
        var expected = Path.Combine(_testProjectRoot, "datasets");
        Assert.Equal(expected, result);
    }
}
