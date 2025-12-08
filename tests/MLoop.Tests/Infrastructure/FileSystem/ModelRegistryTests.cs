using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Infrastructure.FileSystem;

[Collection("FileSystem")]
public class ModelRegistryTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly IExperimentStore _experimentStore;
    private readonly ModelRegistry _modelRegistry;
    private const string DefaultModelName = "default";

    public ModelRegistryTests()
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
        _modelRegistry = new ModelRegistry(_fileSystem, _projectDiscovery, _experimentStore);
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
    public async Task ShouldPromoteAsync_WhenNoProductionModel_ReturnsTrue()
    {
        // Arrange
        var experimentId = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["accuracy"] = 0.85
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName,
            experimentId,
            "accuracy",
            CancellationToken.None);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public async Task ShouldPromoteAsync_WhenNewModelBetter_ReturnsTrue()
    {
        // Arrange
        // Create and promote first model
        var exp1 = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.80
        });
        await _modelRegistry.PromoteAsync(DefaultModelName, exp1, CancellationToken.None);

        // Create better model
        var exp2 = await CreateDummyExperimentAsync(DefaultModelName, "exp-002", new Dictionary<string, double>
        {
            ["r_squared"] = 0.85
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName,
            exp2,
            "r_squared",
            CancellationToken.None);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public async Task ShouldPromoteAsync_WhenNewModelWorse_ReturnsFalse()
    {
        // Arrange
        // Create and promote first model
        var exp1 = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.85
        });
        await _modelRegistry.PromoteAsync(DefaultModelName, exp1, CancellationToken.None);

        // Create worse model
        var exp2 = await CreateDummyExperimentAsync(DefaultModelName, "exp-002", new Dictionary<string, double>
        {
            ["r_squared"] = 0.80
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName,
            exp2,
            "r_squared",
            CancellationToken.None);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public async Task ShouldPromoteAsync_WithErrorMetric_LowerIsBetter()
    {
        // Arrange
        // Create and promote first model
        var exp1 = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["rmse"] = 50.0
        });
        await _modelRegistry.PromoteAsync(DefaultModelName, exp1, CancellationToken.None);

        // Create model with lower error (better)
        var exp2 = await CreateDummyExperimentAsync(DefaultModelName, "exp-002", new Dictionary<string, double>
        {
            ["rmse"] = 40.0
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName,
            exp2,
            "rmse",
            CancellationToken.None);

        // Assert
        Assert.True(result); // Lower error is better
    }

    [Fact]
    public async Task ShouldPromoteAsync_WithErrorMetric_HigherIsWorse()
    {
        // Arrange
        // Create and promote first model
        var exp1 = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["mae"] = 20.0
        });
        await _modelRegistry.PromoteAsync(DefaultModelName, exp1, CancellationToken.None);

        // Create model with higher error (worse)
        var exp2 = await CreateDummyExperimentAsync(DefaultModelName, "exp-002", new Dictionary<string, double>
        {
            ["mae"] = 25.0
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName,
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
        var experimentId = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["accuracy"] = 0.90
        });

        // Act
        var result = await _modelRegistry.AutoPromoteAsync(
            DefaultModelName,
            experimentId,
            "accuracy",
            CancellationToken.None);

        // Assert
        Assert.True(result);

        var productionModel = await _modelRegistry.GetProductionAsync(DefaultModelName, CancellationToken.None);
        Assert.NotNull(productionModel);
        Assert.Equal(experimentId, productionModel.ExperimentId);
    }

    [Fact]
    public async Task PromoteAsync_ToProduction_CreatesProductionModel()
    {
        // Arrange
        var experimentId = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.85
        });

        // Act
        await _modelRegistry.PromoteAsync(DefaultModelName, experimentId, CancellationToken.None);

        // Assert
        var productionModel = await _modelRegistry.GetProductionAsync(DefaultModelName, CancellationToken.None);
        Assert.NotNull(productionModel);
        Assert.Equal(experimentId, productionModel.ExperimentId);
        Assert.Equal(DefaultModelName, productionModel.ModelName);
    }

    [Fact]
    public async Task PromoteAsync_NonExistingExperiment_ThrowsFileNotFoundException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => _modelRegistry.PromoteAsync(DefaultModelName, "exp-999", CancellationToken.None));
    }

    [Fact]
    public async Task PromoteAsync_ReplacesExistingProductionModel()
    {
        // Arrange
        var exp1 = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.80
        });
        var exp2 = await CreateDummyExperimentAsync(DefaultModelName, "exp-002", new Dictionary<string, double>
        {
            ["r_squared"] = 0.90
        });

        await _modelRegistry.PromoteAsync(DefaultModelName, exp1, CancellationToken.None);

        // Act
        await _modelRegistry.PromoteAsync(DefaultModelName, exp2, CancellationToken.None);

        // Assert
        var productionModel = await _modelRegistry.GetProductionAsync(DefaultModelName, CancellationToken.None);
        Assert.NotNull(productionModel);
        Assert.Equal(exp2, productionModel.ExperimentId);
    }

    [Fact]
    public async Task ListAsync_NoModels_ReturnsEmptyList()
    {
        // Act
        var models = await _modelRegistry.ListAsync(null, CancellationToken.None);

        // Assert
        Assert.Empty(models);
    }

    [Fact]
    public async Task ListAsync_WithMultipleModels_ReturnsAllModels()
    {
        // Arrange
        var exp1 = await CreateDummyExperimentAsync("default", "exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.85
        });
        var exp2 = await CreateDummyExperimentAsync("churn", "exp-001", new Dictionary<string, double>
        {
            ["accuracy"] = 0.90
        });

        await _modelRegistry.PromoteAsync("default", exp1, CancellationToken.None);
        await _modelRegistry.PromoteAsync("churn", exp2, CancellationToken.None);

        // Act
        var models = await _modelRegistry.ListAsync(null, CancellationToken.None);
        var modelsList = models.ToList();

        // Assert
        Assert.Equal(2, modelsList.Count);
        Assert.Contains(modelsList, m => m.ModelName == "default" && m.ExperimentId == exp1);
        Assert.Contains(modelsList, m => m.ModelName == "churn" && m.ExperimentId == exp2);
    }

    [Fact]
    public async Task ListAsync_WithModelFilter_ReturnsOnlyMatchingModel()
    {
        // Arrange
        var exp1 = await CreateDummyExperimentAsync("default", "exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.85
        });
        var exp2 = await CreateDummyExperimentAsync("churn", "exp-001", new Dictionary<string, double>
        {
            ["accuracy"] = 0.90
        });

        await _modelRegistry.PromoteAsync("default", exp1, CancellationToken.None);
        await _modelRegistry.PromoteAsync("churn", exp2, CancellationToken.None);

        // Act
        var models = await _modelRegistry.ListAsync("churn", CancellationToken.None);
        var modelsList = models.ToList();

        // Assert
        Assert.Single(modelsList);
        Assert.Equal("churn", modelsList[0].ModelName);
        Assert.Equal(exp2, modelsList[0].ExperimentId);
    }

    [Fact]
    public async Task GetProductionAsync_NonExistingModel_ReturnsNull()
    {
        // Act
        var model = await _modelRegistry.GetProductionAsync(DefaultModelName, CancellationToken.None);

        // Assert
        Assert.Null(model);
    }

    [Fact]
    public async Task PromoteAsync_DifferentModels_IndependentProductionModels()
    {
        // Arrange
        var defaultExp = await CreateDummyExperimentAsync("default", "exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = 0.85
        });
        var churnExp = await CreateDummyExperimentAsync("churn", "exp-001", new Dictionary<string, double>
        {
            ["accuracy"] = 0.90
        });

        // Act
        await _modelRegistry.PromoteAsync("default", defaultExp, CancellationToken.None);
        await _modelRegistry.PromoteAsync("churn", churnExp, CancellationToken.None);

        // Assert
        var defaultProduction = await _modelRegistry.GetProductionAsync("default", CancellationToken.None);
        var churnProduction = await _modelRegistry.GetProductionAsync("churn", CancellationToken.None);

        Assert.NotNull(defaultProduction);
        Assert.NotNull(churnProduction);
        Assert.Equal("default", defaultProduction.ModelName);
        Assert.Equal("churn", churnProduction.ModelName);
        Assert.Equal(defaultExp, defaultProduction.ExperimentId);
        Assert.Equal(churnExp, churnProduction.ExperimentId);
    }

    [Fact]
    public void GetProductionPath_ReturnsCorrectPath()
    {
        // Act
        var path = _modelRegistry.GetProductionPath(DefaultModelName);

        // Assert
        Assert.Contains("models", path);
        Assert.Contains(DefaultModelName, path);
        Assert.Contains("production", path);
    }

    private async Task<string> CreateDummyExperimentAsync(string modelName, string experimentId, Dictionary<string, double> metrics)
    {
        // Create experiment directory
        var experimentPath = _experimentStore.GetExperimentPath(modelName, experimentId);
        Directory.CreateDirectory(experimentPath);

        // Create dummy model file
        var modelPath = Path.Combine(experimentPath, "model.zip");
        await File.WriteAllTextAsync(modelPath, "dummy model content");

        // Create experiment data
        var experimentData = new ExperimentData
        {
            ModelName = modelName,
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

        await _experimentStore.SaveAsync(modelName, experimentData, CancellationToken.None);

        return experimentId;
    }
}
