using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Infrastructure.FileSystem;

public class ExperimentStoreTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly ExperimentStore _experimentStore;

    public ExperimentStoreTests()
    {
        // Create temporary test directory with .mloop
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testProjectRoot);

        var mloopDir = Path.Combine(_testProjectRoot, ".mloop");
        Directory.CreateDirectory(mloopDir);

        _fileSystem = new FileSystemManager();
        _projectDiscovery = new ProjectDiscovery(_fileSystem);

        // Change to test directory so ProjectDiscovery can find it
        Directory.SetCurrentDirectory(_testProjectRoot);

        _experimentStore = new ExperimentStore(_fileSystem, _projectDiscovery);
    }

    public void Dispose()
    {
        // Cleanup test directory
        if (Directory.Exists(_testProjectRoot))
        {
            try
            {
                Directory.Delete(_testProjectRoot, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    [Fact]
    public async Task GenerateIdAsync_FirstId_ReturnsExp000()
    {
        // Act
        var experimentId = await _experimentStore.GenerateIdAsync(CancellationToken.None);

        // Assert
        Assert.Equal("exp-001", experimentId);
    }

    [Fact]
    public async Task GenerateIdAsync_Sequential_ReturnsIncrementingIds()
    {
        // Act
        var id1 = await _experimentStore.GenerateIdAsync(CancellationToken.None);
        var id2 = await _experimentStore.GenerateIdAsync(CancellationToken.None);
        var id3 = await _experimentStore.GenerateIdAsync(CancellationToken.None);

        // Assert
        Assert.Equal("exp-001", id1);
        Assert.Equal("exp-002", id2);
        Assert.Equal("exp-003", id3);
    }

    [Fact]
    public async Task SaveAsync_ValidExperiment_SavesSuccessfully()
    {
        // Arrange
        var experimentId = await _experimentStore.GenerateIdAsync(CancellationToken.None);
        var experimentData = CreateExperimentData(experimentId);

        // Act
        await _experimentStore.SaveAsync(experimentData, CancellationToken.None);

        // Assert
        Assert.True(_experimentStore.ExperimentExists(experimentId));
    }

    [Fact]
    public async Task LoadAsync_ExistingExperiment_ReturnsExperimentData()
    {
        // Arrange
        var experimentId = await _experimentStore.GenerateIdAsync(CancellationToken.None);
        var originalData = CreateExperimentData(experimentId);
        await _experimentStore.SaveAsync(originalData, CancellationToken.None);

        // Act
        var loadedData = await _experimentStore.LoadAsync(experimentId, CancellationToken.None);

        // Assert
        Assert.NotNull(loadedData);
        Assert.Equal(experimentId, loadedData.ExperimentId);
        Assert.Equal(originalData.Status, loadedData.Status);
        Assert.Equal(originalData.Task, loadedData.Task);
        Assert.Equal(originalData.Config.LabelColumn, loadedData.Config.LabelColumn);
    }

    [Fact]
    public async Task LoadAsync_NonExistingExperiment_ThrowsFileNotFoundException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => _experimentStore.LoadAsync("exp-999", CancellationToken.None));
    }

    [Fact]
    public async Task ListAsync_NoExperiments_ReturnsEmptyList()
    {
        // Act
        var experiments = await _experimentStore.ListAsync(CancellationToken.None);

        // Assert
        Assert.Empty(experiments);
    }

    [Fact]
    public async Task ListAsync_MultipleExperiments_ReturnsAllExperiments()
    {
        // Arrange
        var exp1 = await CreateAndSaveExperimentAsync("exp-001", "Completed", 0.85);
        var exp2 = await CreateAndSaveExperimentAsync("exp-002", "Completed", 0.90);
        var exp3 = await CreateAndSaveExperimentAsync("exp-003", "Failed", null);

        // Act
        var experiments = await _experimentStore.ListAsync(CancellationToken.None);
        var experimentsList = experiments.ToList();

        // Assert
        Assert.Equal(3, experimentsList.Count);
        Assert.Contains(experimentsList, e => e.ExperimentId == "exp-001" && e.Status == "Completed");
        Assert.Contains(experimentsList, e => e.ExperimentId == "exp-002" && e.Status == "Completed");
        Assert.Contains(experimentsList, e => e.ExperimentId == "exp-003" && e.Status == "Failed");
    }

    [Fact]
    public async Task ListAsync_WithMetrics_IncludesMetricsInSummary()
    {
        // Arrange
        await CreateAndSaveExperimentAsync("exp-001", "Completed", 0.95);

        // Act
        var experiments = await _experimentStore.ListAsync(CancellationToken.None);
        var experiment = experiments.FirstOrDefault();

        // Assert
        Assert.NotNull(experiment);
        Assert.Equal(0.95, experiment.BestMetric);
    }

    [Fact]
    public async Task ExperimentExists_ExistingExperiment_ReturnsTrue()
    {
        // Arrange
        var experimentId = await CreateAndSaveExperimentAsync("exp-001", "Completed", 0.85);

        // Act
        var exists = _experimentStore.ExperimentExists(experimentId);

        // Assert
        Assert.True(exists);
    }

    [Fact]
    public void ExperimentExists_NonExistingExperiment_ReturnsFalse()
    {
        // Act
        var exists = _experimentStore.ExperimentExists("exp-999");

        // Assert
        Assert.False(exists);
    }

    [Fact]
    public void GetExperimentPath_ReturnsCorrectPath()
    {
        // Arrange
        var experimentId = "exp-001";

        // Act
        var path = _experimentStore.GetExperimentPath(experimentId);

        // Assert
        Assert.Contains("models", path);
        Assert.Contains("staging", path);
        Assert.Contains(experimentId, path);
    }

    private async Task<string> CreateAndSaveExperimentAsync(string experimentId, string status, double? metric)
    {
        var experimentData = CreateExperimentData(experimentId, status, metric);
        await _experimentStore.SaveAsync(experimentData, CancellationToken.None);
        return experimentId;
    }

    private ExperimentData CreateExperimentData(string experimentId, string status = "Completed", double? metric = 0.85)
    {
        return new ExperimentData
        {
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = status,
            Task = "regression",
            Config = new ExperimentConfig
            {
                DataFile = "test.csv",
                LabelColumn = "label",
                TimeLimitSeconds = 60,
                Metric = "r_squared",
                TestSplit = 0.2
            },
            Metrics = metric.HasValue ? new Dictionary<string, double>
            {
                ["r_squared"] = metric.Value
            } : null
        };
    }
}
