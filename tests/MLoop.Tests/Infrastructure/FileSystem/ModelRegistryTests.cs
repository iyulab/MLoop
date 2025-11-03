using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Infrastructure.FileSystem;

public class ModelRegistryTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;
    private readonly ModelRegistry _modelRegistry;

    public ModelRegistryTests()
    {
        // Create temporary test directory with .mloop
        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testProjectRoot);

        var mloopDir = Path.Combine(_testProjectRoot, ".mloop");
        Directory.CreateDirectory(mloopDir);

        _fileSystem = new FileSystemManager();
        _projectDiscovery = new ProjectDiscovery(_fileSystem);

        // Change to test directory so ProjectDiscovery can find it
        var originalDir = Directory.GetCurrentDirectory();
        Directory.SetCurrentDirectory(_testProjectRoot);

        _experimentStore = new ExperimentStore(_fileSystem, _projectDiscovery);
        _modelRegistry = new ModelRegistry(_fileSystem, _projectDiscovery, _experimentStore);
    }

    public void Dispose()
    {
        // Restore original directory
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
    public async Task ShouldPromoteToProduction_WhenNoProductionModel_ReturnsTrue()
    {
        // Arrange
        var experimentId = await CreateDummyExperimentAsync("exp-001", new Dictionary<string, double>
        {
            ["accuracy"] = 0.85
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteToProductionAsync(
            experimentId,
            "accuracy",
            CancellationToken.None);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public async Task ShouldPromoteToProduction_WhenNewModelBetter_ReturnsTrue()
    {
        // Arrange
        // Create and promote first model
        var exp1 = await CreateDummyExperimentAsync("exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.80
        });
        await _modelRegistry.PromoteAsync(exp1, ModelStage.Production, CancellationToken.None);

        // Create better model
        var exp2 = await CreateDummyExperimentAsync("exp-002", new Dictionary<string, double>
        {
            ["r_squared"] = 0.85
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteToProductionAsync(
            exp2,
            "r_squared",
            CancellationToken.None);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public async Task ShouldPromoteToProduction_WhenNewModelWorse_ReturnsFalse()
    {
        // Arrange
        // Create and promote first model
        var exp1 = await CreateDummyExperimentAsync("exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.85
        });
        await _modelRegistry.PromoteAsync(exp1, ModelStage.Production, CancellationToken.None);

        // Create worse model
        var exp2 = await CreateDummyExperimentAsync("exp-002", new Dictionary<string, double>
        {
            ["r_squared"] = 0.80
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteToProductionAsync(
            exp2,
            "r_squared",
            CancellationToken.None);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public async Task ShouldPromoteToProduction_WithErrorMetric_LowerIsBetter()
    {
        // Arrange
        // Create and promote first model
        var exp1 = await CreateDummyExperimentAsync("exp-001", new Dictionary<string, double>
        {
            ["rmse"] = 50.0
        });
        await _modelRegistry.PromoteAsync(exp1, ModelStage.Production, CancellationToken.None);

        // Create model with lower error (better)
        var exp2 = await CreateDummyExperimentAsync("exp-002", new Dictionary<string, double>
        {
            ["rmse"] = 40.0
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteToProductionAsync(
            exp2,
            "rmse",
            CancellationToken.None);

        // Assert
        Assert.True(result); // Lower error is better
    }

    [Fact]
    public async Task ShouldPromoteToProduction_WithErrorMetric_HigherIsWorse()
    {
        // Arrange
        // Create and promote first model
        var exp1 = await CreateDummyExperimentAsync("exp-001", new Dictionary<string, double>
        {
            ["mae"] = 20.0
        });
        await _modelRegistry.PromoteAsync(exp1, ModelStage.Production, CancellationToken.None);

        // Create model with higher error (worse)
        var exp2 = await CreateDummyExperimentAsync("exp-002", new Dictionary<string, double>
        {
            ["mae"] = 25.0
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteToProductionAsync(
            exp2,
            "mae",
            CancellationToken.None);

        // Assert
        Assert.False(result); // Higher error is worse
    }

    [Fact]
    public async Task AutoPromoteAsync_WhenShouldPromote_PromotesAndReturnsTrue()
    {
        // Arrange
        var experimentId = await CreateDummyExperimentAsync("exp-001", new Dictionary<string, double>
        {
            ["accuracy"] = 0.90
        });

        // Act
        var result = await _modelRegistry.AutoPromoteAsync(
            experimentId,
            "accuracy",
            CancellationToken.None);

        // Assert
        Assert.True(result);

        var productionModel = await _modelRegistry.GetAsync(ModelStage.Production, CancellationToken.None);
        Assert.NotNull(productionModel);
        Assert.Equal(experimentId, productionModel.ExperimentId);
    }

    [Fact]
    public async Task PromoteAsync_ToProduction_CreatesProductionModel()
    {
        // Arrange
        var experimentId = await CreateDummyExperimentAsync("exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.85
        });

        // Act
        await _modelRegistry.PromoteAsync(experimentId, ModelStage.Production, CancellationToken.None);

        // Assert
        var productionModel = await _modelRegistry.GetAsync(ModelStage.Production, CancellationToken.None);
        Assert.NotNull(productionModel);
        Assert.Equal(experimentId, productionModel.ExperimentId);
        Assert.Equal(ModelStage.Production, productionModel.Stage);
    }

    [Fact]
    public async Task PromoteAsync_ToStaging_CreatessStagingModel()
    {
        // Arrange
        var experimentId = await CreateDummyExperimentAsync("exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.85
        });

        // Act
        await _modelRegistry.PromoteAsync(experimentId, ModelStage.Staging, CancellationToken.None);

        // Assert
        var stagingModel = await _modelRegistry.GetAsync(ModelStage.Staging, CancellationToken.None);
        Assert.NotNull(stagingModel);
        Assert.Equal(experimentId, stagingModel.ExperimentId);
        Assert.Equal(ModelStage.Staging, stagingModel.Stage);
    }

    [Fact]
    public async Task PromoteAsync_NonExistingExperiment_ThrowsFileNotFoundException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => _modelRegistry.PromoteAsync("exp-999", ModelStage.Production, CancellationToken.None));
    }

    [Fact]
    public async Task PromoteAsync_ReplacesExistingProductionModel()
    {
        // Arrange
        var exp1 = await CreateDummyExperimentAsync("exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.80
        });
        var exp2 = await CreateDummyExperimentAsync("exp-002", new Dictionary<string, double>
        {
            ["r_squared"] = 0.90
        });

        await _modelRegistry.PromoteAsync(exp1, ModelStage.Production, CancellationToken.None);

        // Act
        await _modelRegistry.PromoteAsync(exp2, ModelStage.Production, CancellationToken.None);

        // Assert
        var productionModel = await _modelRegistry.GetAsync(ModelStage.Production, CancellationToken.None);
        Assert.NotNull(productionModel);
        Assert.Equal(exp2, productionModel.ExperimentId);
    }

    [Fact]
    public async Task ListAsync_NoModels_ReturnsEmptyList()
    {
        // Act
        var models = await _modelRegistry.ListAsync(CancellationToken.None);

        // Assert
        Assert.Empty(models);
    }

    [Fact]
    public async Task ListAsync_WithProductionAndStaging_ReturnsBothModels()
    {
        // Arrange
        var exp1 = await CreateDummyExperimentAsync("exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.85
        });
        var exp2 = await CreateDummyExperimentAsync("exp-002", new Dictionary<string, double>
        {
            ["r_squared"] = 0.90
        });

        await _modelRegistry.PromoteAsync(exp1, ModelStage.Staging, CancellationToken.None);
        await _modelRegistry.PromoteAsync(exp2, ModelStage.Production, CancellationToken.None);

        // Act
        var models = await _modelRegistry.ListAsync(CancellationToken.None);
        var modelsList = models.ToList();

        // Assert
        Assert.Equal(2, modelsList.Count);
        Assert.Contains(modelsList, m => m.Stage == ModelStage.Production && m.ExperimentId == exp2);
        Assert.Contains(modelsList, m => m.Stage == ModelStage.Staging && m.ExperimentId == exp1);
    }

    [Fact]
    public async Task GetAsync_NonExistingStage_ReturnsNull()
    {
        // Act
        var model = await _modelRegistry.GetAsync(ModelStage.Production, CancellationToken.None);

        // Assert
        Assert.Null(model);
    }

    private async Task<string> CreateDummyExperimentAsync(string experimentId, Dictionary<string, double> metrics)
    {
        // Create experiment directory
        var experimentPath = _experimentStore.GetExperimentPath(experimentId);
        Directory.CreateDirectory(experimentPath);

        // Create dummy model file
        var modelPath = Path.Combine(experimentPath, "model.zip");
        await File.WriteAllTextAsync(modelPath, "dummy model content");

        // Create experiment data
        var experimentData = new ExperimentData
        {
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = "regression",
            Config = new ExperimentConfig
            {
                DataFile = "test.csv",
                LabelColumn = "label",
                TimeLimitSeconds = 60,
                Metric = metrics.Keys.First(),
                TestSplit = 0.2
            },
            Metrics = metrics
        };

        await _experimentStore.SaveAsync(experimentData, CancellationToken.None);

        return experimentId;
    }
}
