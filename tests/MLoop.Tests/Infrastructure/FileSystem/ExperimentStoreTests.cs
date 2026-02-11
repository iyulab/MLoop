using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Infrastructure.FileSystem;

[Collection("FileSystem")]
public class ExperimentStoreTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly ExperimentStore _experimentStore;
    private const string DefaultModelName = "default";

    public ExperimentStoreTests()
    {
        // Store original directory first
        _originalDirectory = Directory.GetCurrentDirectory();

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
        // Restore original directory BEFORE deleting the temp directory
        try
        {
            Directory.SetCurrentDirectory(_originalDirectory);
        }
        catch
        {
            // If original directory doesn't exist, try to set to temp path
            try
            {
                Directory.SetCurrentDirectory(Path.GetTempPath());
            }
            catch
            {
                // Ignore if we can't restore directory
            }
        }

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
    public async Task GenerateIdAsync_FirstId_ReturnsExp001()
    {
        // Act
        var experimentId = await _experimentStore.GenerateIdAsync(DefaultModelName, CancellationToken.None);

        // Assert
        Assert.Equal("exp-001", experimentId);
    }

    [Fact]
    public async Task GenerateIdAsync_Sequential_ReturnsIncrementingIds()
    {
        // Act
        var id1 = await _experimentStore.GenerateIdAsync(DefaultModelName, CancellationToken.None);
        var id2 = await _experimentStore.GenerateIdAsync(DefaultModelName, CancellationToken.None);
        var id3 = await _experimentStore.GenerateIdAsync(DefaultModelName, CancellationToken.None);

        // Assert
        Assert.Equal("exp-001", id1);
        Assert.Equal("exp-002", id2);
        Assert.Equal("exp-003", id3);
    }

    [Fact]
    public async Task GenerateIdAsync_DifferentModels_IndependentIds()
    {
        // Act
        var defaultId = await _experimentStore.GenerateIdAsync("default", CancellationToken.None);
        var churnId = await _experimentStore.GenerateIdAsync("churn", CancellationToken.None);
        var revenueId = await _experimentStore.GenerateIdAsync("revenue", CancellationToken.None);

        // Assert - each model starts from exp-001
        Assert.Equal("exp-001", defaultId);
        Assert.Equal("exp-001", churnId);
        Assert.Equal("exp-001", revenueId);
    }

    [Fact]
    public async Task SaveAsync_ValidExperiment_SavesSuccessfully()
    {
        // Arrange
        var experimentId = await _experimentStore.GenerateIdAsync(DefaultModelName, CancellationToken.None);
        var experimentData = CreateExperimentData(experimentId);

        // Act
        await _experimentStore.SaveAsync(DefaultModelName, experimentData, CancellationToken.None);

        // Assert
        Assert.True(_experimentStore.ExperimentExists(DefaultModelName, experimentId));
    }

    [Fact]
    public async Task LoadAsync_ExistingExperiment_ReturnsExperimentData()
    {
        // Arrange
        var experimentId = await _experimentStore.GenerateIdAsync(DefaultModelName, CancellationToken.None);
        var originalData = CreateExperimentData(experimentId);
        await _experimentStore.SaveAsync(DefaultModelName, originalData, CancellationToken.None);

        // Act
        var loadedData = await _experimentStore.LoadAsync(DefaultModelName, experimentId, CancellationToken.None);

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
            () => _experimentStore.LoadAsync(DefaultModelName, "exp-999", CancellationToken.None));
    }

    [Fact]
    public async Task ListAsync_NoExperiments_ReturnsEmptyList()
    {
        // Act
        var experiments = await _experimentStore.ListAsync(DefaultModelName, CancellationToken.None);

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
        var experiments = await _experimentStore.ListAsync(DefaultModelName, CancellationToken.None);
        var experimentsList = experiments.ToList();

        // Assert
        Assert.Equal(3, experimentsList.Count);
        Assert.Contains(experimentsList, e => e.ExperimentId == "exp-001" && e.Status == "Completed");
        Assert.Contains(experimentsList, e => e.ExperimentId == "exp-002" && e.Status == "Completed");
        Assert.Contains(experimentsList, e => e.ExperimentId == "exp-003" && e.Status == "Failed");
    }

    [Fact]
    public async Task ListAsync_AllModels_ReturnsExperimentsFromAllModels()
    {
        // Arrange
        await CreateAndSaveExperimentAsync("exp-001", "Completed", 0.85, "default");
        await CreateAndSaveExperimentAsync("exp-001", "Completed", 0.90, "churn");

        // Act
        var experiments = await _experimentStore.ListAsync(null, CancellationToken.None);
        var experimentsList = experiments.ToList();

        // Assert
        Assert.Equal(2, experimentsList.Count);
        Assert.Contains(experimentsList, e => e.ModelName == "default");
        Assert.Contains(experimentsList, e => e.ModelName == "churn");
    }

    [Fact]
    public async Task ListAsync_WithMetrics_IncludesMetricsInSummary()
    {
        // Arrange
        await CreateAndSaveExperimentAsync("exp-001", "Completed", 0.95);

        // Act
        var experiments = await _experimentStore.ListAsync(DefaultModelName, CancellationToken.None);
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
        var exists = _experimentStore.ExperimentExists(DefaultModelName, experimentId);

        // Assert
        Assert.True(exists);
    }

    [Fact]
    public void ExperimentExists_NonExistingExperiment_ReturnsFalse()
    {
        // Act
        var exists = _experimentStore.ExperimentExists(DefaultModelName, "exp-999");

        // Assert
        Assert.False(exists);
    }

    [Fact]
    public async Task ListAsync_WithResult_IncludesTrainerAndMetricName()
    {
        // Arrange
        var experimentData = new ExperimentData
        {
            ModelName = DefaultModelName,
            ExperimentId = "exp-001",
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = "regression",
            Config = new ExperimentConfig
            {
                DataFile = "test.csv",
                LabelColumn = "label",
                TimeLimitSeconds = 120,
                Metric = "RSquared",
                TestSplit = 0.2
            },
            Result = new ExperimentResult
            {
                BestTrainer = "FastForestRegression",
                TrainingTimeSeconds = 45.3
            },
            Metrics = new Dictionary<string, double> { ["RSquared"] = 0.92 }
        };
        await _experimentStore.SaveAsync(DefaultModelName, experimentData, CancellationToken.None);

        // Act
        var experiments = await _experimentStore.ListAsync(DefaultModelName, CancellationToken.None);
        var experiment = experiments.First();

        // Assert
        Assert.Equal("FastForestRegression", experiment.BestTrainer);
        Assert.Equal("RSquared", experiment.MetricName);
        Assert.Equal(45.3, experiment.TrainingTimeSeconds);
    }

    [Fact]
    public void GetExperimentPath_ReturnsCorrectPath()
    {
        // Arrange
        var experimentId = "exp-001";

        // Act
        var path = _experimentStore.GetExperimentPath(DefaultModelName, experimentId);

        // Assert
        Assert.Contains("models", path);
        Assert.Contains(DefaultModelName, path);
        Assert.Contains("staging", path);
        Assert.Contains(experimentId, path);
    }

    private async Task<string> CreateAndSaveExperimentAsync(string experimentId, string status, double? metric, string? modelName = null)
    {
        var resolvedModelName = modelName ?? DefaultModelName;
        var experimentData = CreateExperimentData(experimentId, status, metric, resolvedModelName);
        await _experimentStore.SaveAsync(resolvedModelName, experimentData, CancellationToken.None);
        return experimentId;
    }

    private ExperimentData CreateExperimentData(string experimentId, string status = "Completed", double? metric = 0.85, string? modelName = null)
    {
        return new ExperimentData
        {
            ModelName = modelName ?? DefaultModelName,
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
