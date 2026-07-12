using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Models;

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
    public async Task ShouldPromoteAsync_ForecastingAutoMetric_WorseModelRejected()
    {
        // TD-06 convergence: before centralizing on TaskMetadata, ModelRegistry.DefaultMetricForTask
        // returned null for forecasting, so the gate resolved no metricKey and *always* promoted the
        // new model. Now forecasting resolves its canonical "mae" (an error metric), so a worse model
        // (higher mae) is correctly rejected — while the metric is threshold-less, so the gate never
        // falsely blocks (the BUG-46-inverse for the newly-covered tasks).
        var exp1 = await CreateDummyExperimentAsync(DefaultModelName, "exp-001",
            new Dictionary<string, double> { ["mae"] = 10.0 }, task: "forecasting", metricConfig: "auto");
        await _modelRegistry.PromoteAsync(DefaultModelName, exp1, CancellationToken.None);

        var exp2 = await CreateDummyExperimentAsync(DefaultModelName, "exp-002",
            new Dictionary<string, double> { ["mae"] = 15.0 }, task: "forecasting", metricConfig: "auto");

        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName, exp2, "auto", CancellationToken.None);

        Assert.False(result); // worse forecasting model (higher mae) no longer auto-promotes
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
    public async Task ShouldPromoteAsync_UndefinedErrorMetricSentinel_ReturnsFalse()
    {
        // Live repro (8-row regression, --metric rmse): the model scored NaN for every holdout row,
        // MetricSanitizer recorded rmse as the worst-case sentinel — and the gate still promoted it,
        // because error-direction metrics have no floor threshold. The sentinel check must block it.
        var experimentId = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = double.MinValue,
            ["rmse"] = double.MaxValue,
            ["mae"] = double.MaxValue,
            ["mse"] = double.MaxValue
        });

        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName,
            experimentId,
            "rmse",
            CancellationToken.None);

        Assert.False(result); // undefined-metric sentinel — degenerate model must not promote
    }

    [Fact]
    public async Task ShouldPromoteAsync_UndefinedHigherBetterMetricSentinel_ReturnsFalse()
    {
        // Same degenerate signature gated via a higher-is-better metric: r² = MinValue is caught by
        // the sentinel check (it also happens to sit below the 0.0 floor — both agree).
        var experimentId = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["r_squared"] = double.MinValue,
            ["rmse"] = double.MaxValue
        });

        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName,
            experimentId,
            "r_squared",
            CancellationToken.None);

        Assert.False(result);
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

    [Fact]
    public async Task ShouldPromoteAsync_NoProduction_F1Alias_ReturnsTrue()
    {
        // Arrange — user passes "--metric f1" but EvaluationEngine stores the key as "f1_score".
        // A non-degenerate first model must still auto-promote despite the alias mismatch.
        var experimentId = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["accuracy"] = 0.87,
            ["f1_score"] = 0.73,
            ["auc"] = 0.88
        });

        // Act — primary metric is the user-facing alias "f1"
        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName,
            experimentId,
            "f1",
            CancellationToken.None);

        // Assert
        Assert.True(result); // Passes quality gate, no production → promote
    }

    [Fact]
    public async Task ShouldPromoteAsync_F1Alias_ComparesAgainstProduction()
    {
        // Arrange — production has f1_score=0.70, new model f1_score=0.80, primary metric alias "f1".
        var exp1 = await CreateDummyExperimentAsync(DefaultModelName, "exp-001", new Dictionary<string, double>
        {
            ["accuracy"] = 0.80,
            ["f1_score"] = 0.70
        });
        await _modelRegistry.PromoteAsync(DefaultModelName, exp1, CancellationToken.None);

        var exp2 = await CreateDummyExperimentAsync(DefaultModelName, "exp-002", new Dictionary<string, double>
        {
            ["accuracy"] = 0.85,
            ["f1_score"] = 0.80
        });

        // Act
        var betterResult = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName, exp2, "f1", CancellationToken.None);

        // Assert — alias must resolve to f1_score and compare correctly
        Assert.True(betterResult);
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

    [Fact]
    public async Task ShouldPromoteAsync_ImageAutoMetric_NonConverged_ReturnsFalse()
    {
        // BUG-46: a non-converged image-classification model (micro_accuracy below the 1/N
        // random baseline) must be blocked even though the project metric is the deferred
        // "auto" (init leaves image tasks as auto). Binary image task → classCount=2 → 1/2=0.5.
        var experimentId = await CreateDummyExperimentAsync(
            DefaultModelName, "exp-001",
            new Dictionary<string, double>
            {
                ["accuracy"] = 0.5,           // MacroAccuracy
                ["micro_accuracy"] = 0.4286,  // below 1/2 random baseline
                ["log_loss"] = 34.54
            },
            task: "image-classification",
            metricConfig: "auto",
            classCount: 2);

        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName, experimentId, "auto", CancellationToken.None);

        Assert.False(result); // micro_accuracy 0.4286 < 0.5 → quality gate blocks
    }

    [Fact]
    public async Task ShouldPromoteAsync_ImageAutoMetric_Converged_ReturnsTrue()
    {
        // A converged 6-class image model (micro_accuracy 0.90 > 1/6≈0.167) auto-promotes
        // despite the "auto" metric.
        var experimentId = await CreateDummyExperimentAsync(
            DefaultModelName, "exp-001",
            new Dictionary<string, double>
            {
                ["accuracy"] = 0.88,
                ["micro_accuracy"] = 0.90,
                ["log_loss"] = 0.35
            },
            task: "image-classification",
            metricConfig: "auto",
            classCount: 6);

        var result = await _modelRegistry.ShouldPromoteAsync(
            DefaultModelName, experimentId, "auto", CancellationToken.None);

        Assert.True(result); // passes 1/N gate, no production → promote
    }

    private async Task<string> CreateDummyExperimentAsync(
        string modelName,
        string experimentId,
        Dictionary<string, double> metrics,
        string task = "regression",
        string metricConfig = "",
        int? classCount = null)
    {
        // Create experiment directory
        var experimentPath = _experimentStore.GetExperimentPath(modelName, experimentId);
        Directory.CreateDirectory(experimentPath);

        // Create dummy model file
        var modelPath = Path.Combine(experimentPath, "model.zip");
        await File.WriteAllTextAsync(modelPath, "dummy model content");

        // Populate a label schema carrying the class count when requested (mirrors how
        // directory-based tasks feed the quality gate's 1/N threshold).
        InputSchemaInfo? inputSchema = classCount.HasValue
            ? new InputSchemaInfo
            {
                CapturedAt = DateTime.UtcNow,
                Columns = new List<ColumnSchema>
                {
                    new ColumnSchema { Name = "label", DataType = "Categorical", Purpose = "Label", UniqueValueCount = classCount }
                }
            }
            : null;

        // Create experiment data
        var experimentData = new ExperimentData
        {
            ModelName = modelName,
            ExperimentId = experimentId,
            Timestamp = DateTime.UtcNow,
            Status = "Completed",
            Task = task,
            Config = new ExperimentConfig
            {
                DataFile = "test.csv",
                LabelColumn = "label",
                TimeLimitSeconds = 60,
                Metric = string.IsNullOrEmpty(metricConfig) ? metrics.Keys.First() : metricConfig,
                TestSplit = 0.2,
                InputSchema = inputSchema
            },
            Metrics = metrics
        };

        await _experimentStore.SaveAsync(modelName, experimentData, CancellationToken.None);

        return experimentId;
    }
}
