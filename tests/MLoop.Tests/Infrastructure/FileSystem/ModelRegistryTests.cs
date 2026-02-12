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
    public async Task ShouldPromoteAsync_NegativeRSquared_ReturnsFalse()
    {
        // Arrange — R² below 0.0 should be rejected
        var experimentId = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = -0.05
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName,
            experimentId,
            "r_squared",
            CancellationToken.None);

        // Assert
        Assert.False(result); // Below minimum threshold (0.0)
    }

    [Fact]
    public async Task ShouldPromoteAsync_AucBelowRandom_ReturnsFalse()
    {
        // Arrange — AUC below 0.5 should be rejected (worse than random)
        var experimentId = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["auc"] = 0.45
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName,
            experimentId,
            "auc",
            CancellationToken.None);

        // Assert
        Assert.False(result); // Below minimum threshold (0.5)
    }

    [Fact]
    public async Task ShouldPromoteAsync_AucAboveRandom_ReturnsTrue()
    {
        // Arrange — AUC above 0.5 should be promoted (no production model)
        var experimentId = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["auc"] = 0.55
        });

        // Act
        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName,
            experimentId,
            "auc",
            CancellationToken.None);

        // Assert
        Assert.True(result); // Above minimum threshold, no production model
    }

    [Theory]
    [InlineData("r_squared", 0.0)]
    [InlineData("auc", 0.5)]
    [InlineData("accuracy", 0.0)]
    [InlineData("f1_score", 0.0)]
    public void GetMinimumMetricThreshold_ReturnsExpectedValues(string metric, double expected)
    {
        var threshold = ModelRegistry.GetMinimumMetricThreshold(metric);
        Assert.NotNull(threshold);
        Assert.Equal(expected, threshold.Value);
    }

    [Theory]
    [InlineData("mae")]
    [InlineData("rmse")]
    [InlineData("mse")]
    public void GetMinimumMetricThreshold_ErrorMetrics_ReturnsNull(string metric)
    {
        var threshold = ModelRegistry.GetMinimumMetricThreshold(metric);
        Assert.Null(threshold);
    }

    [Theory]
    [InlineData("accuracy", 2, 0.5)]     // Binary: 1/2
    [InlineData("accuracy", 3, 0.3333)]  // 3-class: 1/3
    [InlineData("accuracy", 10, 0.1)]    // 10-class: 1/10
    [InlineData("macro_accuracy", 5, 0.2)] // 5-class: 1/5
    public void GetMinimumMetricThreshold_WithClassCount_ReturnsDynamicThreshold(
        string metric, int classCount, double expected)
    {
        var threshold = ModelRegistry.GetMinimumMetricThreshold(metric, classCount);
        Assert.NotNull(threshold);
        Assert.Equal(expected, threshold.Value, 3); // precision to 3 decimal places
    }

    [Fact]
    public void GetMinimumMetricThreshold_AccuracyWithoutClassCount_ReturnsZero()
    {
        var threshold = ModelRegistry.GetMinimumMetricThreshold("accuracy");
        Assert.NotNull(threshold);
        Assert.Equal(0.0, threshold.Value);
    }

    [Fact]
    public void GetMinimumMetricThreshold_AccuracyWithClassCount_HigherThanWithout()
    {
        var withoutClass = ModelRegistry.GetMinimumMetricThreshold("accuracy");
        var withClass = ModelRegistry.GetMinimumMetricThreshold("accuracy", 3);
        Assert.True(withClass > withoutClass);
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

    #region IsClassificationDegenerateModel

    [Fact]
    public void IsClassificationDegenerateModel_BinaryHighAccZeroF1_ReturnsTrue()
    {
        var metrics = new Dictionary<string, double>
        {
            ["accuracy"] = 0.95,
            ["f1_score"] = 0.0
        };

        Assert.True(ModelRegistry.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_MulticlassHighAccZeroF1_ReturnsTrue()
    {
        var metrics = new Dictionary<string, double>
        {
            ["macro_accuracy"] = 0.80,
            ["macro_f1"] = 0.0
        };

        Assert.True(ModelRegistry.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_NormalMetrics_ReturnsFalse()
    {
        var metrics = new Dictionary<string, double>
        {
            ["accuracy"] = 0.85,
            ["f1_score"] = 0.78
        };

        Assert.False(ModelRegistry.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_LowAccZeroF1_ReturnsFalse()
    {
        // Low accuracy + zero F1 = just a bad model, not degenerate
        var metrics = new Dictionary<string, double>
        {
            ["accuracy"] = 0.3,
            ["f1_score"] = 0.0
        };

        Assert.False(ModelRegistry.IsClassificationDegenerateModel(metrics));
    }

    [Fact]
    public void IsClassificationDegenerateModel_RegressionMetrics_ReturnsFalse()
    {
        var metrics = new Dictionary<string, double>
        {
            ["r_squared"] = 0.9,
            ["rmse"] = 0.5
        };

        Assert.False(ModelRegistry.IsClassificationDegenerateModel(metrics));
    }

    #endregion

    #region GetMinimumMetricThreshold

    [Theory]
    [InlineData("r_squared", null, 0.0)]
    [InlineData("auc", null, 0.5)]
    [InlineData("f1_score", null, 0.0)]
    public void GetMinimumMetricThreshold_KnownMetrics_ReturnsThreshold(string metric, int? classCount, double expected)
    {
        var result = ModelRegistry.GetMinimumMetricThreshold(metric, classCount);
        Assert.NotNull(result);
        Assert.Equal(expected, result!.Value, precision: 4);
    }

    [Fact]
    public void GetMinimumMetricThreshold_AccuracyWithClassCount_ReturnsDynamic()
    {
        // 5 classes → threshold = 1/5 = 0.2
        var result = ModelRegistry.GetMinimumMetricThreshold("accuracy", 5);
        Assert.NotNull(result);
        Assert.Equal(0.2, result!.Value, precision: 4);
    }

    [Fact]
    public void GetMinimumMetricThreshold_UnknownMetric_ReturnsNull()
    {
        var result = ModelRegistry.GetMinimumMetricThreshold("custom_metric");
        Assert.Null(result);
    }

    [Fact]
    public void GetMinimumMetricThreshold_ErrorMetric_ReturnsNull()
    {
        var result = ModelRegistry.GetMinimumMetricThreshold("rmse");
        Assert.Null(result);
    }

    #endregion
}
