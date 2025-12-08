using MLoop.AIAgent.Agents;
using MLoop.AIAgent.Core;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Tests;

public class MLOpsManagerAgentTests
{
    private readonly MLoopProjectManager _projectManager;

    public MLOpsManagerAgentTests()
    {
        _projectManager = new MLoopProjectManager();
    }

    #region Model Tests

    [Fact]
    public void MLoopProjectConfig_CanBeInitialized()
    {
        // Act
        var config = new MLoopProjectConfig
        {
            ProjectName = "test-project",
            DataPath = "/data/test.csv",
            LabelColumn = "target",
            TaskType = "binary-classification"
        };

        // Assert
        Assert.Equal("test-project", config.ProjectName);
        Assert.Equal("/data/test.csv", config.DataPath);
        Assert.Equal("target", config.LabelColumn);
        Assert.Equal("binary-classification", config.TaskType);
        Assert.Null(config.ProjectDirectory);
    }

    [Fact]
    public void MLoopProjectConfig_WithProjectDirectory_CanBeInitialized()
    {
        // Act
        var config = new MLoopProjectConfig
        {
            ProjectName = "test-project",
            DataPath = "/data/test.csv",
            LabelColumn = "target",
            TaskType = "regression",
            ProjectDirectory = "/projects/test"
        };

        // Assert
        Assert.Equal("/projects/test", config.ProjectDirectory);
    }

    [Fact]
    public void MLoopTrainingConfig_CanBeInitialized()
    {
        // Act
        var config = new MLoopTrainingConfig
        {
            TimeSeconds = 120,
            Metric = "F1Score"
        };

        // Assert
        Assert.Equal(120, config.TimeSeconds);
        Assert.Equal("F1Score", config.Metric);
        Assert.Equal(0.2, config.TestSplit); // Default
    }

    [Fact]
    public void MLoopTrainingConfig_WithAllOptions_CanBeInitialized()
    {
        // Act
        var config = new MLoopTrainingConfig
        {
            TimeSeconds = 300,
            Metric = "AUC",
            TestSplit = 0.3,
            DataPath = "/data/train.csv",
            ExperimentName = "experiment-v1"
        };

        // Assert
        Assert.Equal(300, config.TimeSeconds);
        Assert.Equal("AUC", config.Metric);
        Assert.Equal(0.3, config.TestSplit);
        Assert.Equal("/data/train.csv", config.DataPath);
        Assert.Equal("experiment-v1", config.ExperimentName);
    }

    [Fact]
    public void MLoopExperiment_CanBeInitialized()
    {
        // Arrange
        var timestamp = DateTime.Now;

        // Act
        var experiment = new MLoopExperiment
        {
            Id = "exp-12345678",
            Name = "churn-prediction",
            Timestamp = timestamp,
            Trainer = "LightGbm",
            MetricValue = 0.923,
            MetricName = "F1Score",
            IsProduction = true
        };

        // Assert
        Assert.Equal("exp-12345678", experiment.Id);
        Assert.Equal("churn-prediction", experiment.Name);
        Assert.Equal(timestamp, experiment.Timestamp);
        Assert.Equal("LightGbm", experiment.Trainer);
        Assert.Equal(0.923, experiment.MetricValue);
        Assert.Equal("F1Score", experiment.MetricName);
        Assert.True(experiment.IsProduction);
    }

    [Fact]
    public void MLoopOperationResult_Success_CanBeInitialized()
    {
        // Act
        var result = new MLoopOperationResult
        {
            Success = true,
            ExitCode = 0,
            Output = "Training completed successfully"
        };

        // Assert
        Assert.True(result.Success);
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("Training completed successfully", result.Output);
        Assert.Null(result.Error);
        Assert.Empty(result.Data);
    }

    [Fact]
    public void MLoopOperationResult_WithError_CanBeInitialized()
    {
        // Act
        var result = new MLoopOperationResult
        {
            Success = false,
            ExitCode = 1,
            Output = "",
            Error = "File not found: data.csv"
        };

        // Assert
        Assert.False(result.Success);
        Assert.Equal(1, result.ExitCode);
        Assert.Equal("File not found: data.csv", result.Error);
    }

    [Fact]
    public void MLoopOperationResult_WithData_StoresMetrics()
    {
        // Act
        var result = new MLoopOperationResult
        {
            Success = true,
            ExitCode = 0,
            Output = "Training completed",
            Data = new Dictionary<string, object>
            {
                ["ExperimentId"] = "exp-abc123",
                ["BestTrainer"] = "FastTree",
                ["MetricValue"] = 0.95
            }
        };

        // Assert
        Assert.Equal(3, result.Data.Count);
        Assert.Equal("exp-abc123", result.Data["ExperimentId"]);
        Assert.Equal("FastTree", result.Data["BestTrainer"]);
        Assert.Equal(0.95, result.Data["MetricValue"]);
    }

    #endregion

    #region MLoopProjectManager Tests

    [Fact]
    public void MLoopProjectManager_DefaultConstructor_CreatesInstance()
    {
        // Act
        var manager = new MLoopProjectManager();

        // Assert
        Assert.NotNull(manager);
    }

    [Fact]
    public void MLoopProjectManager_WithCliPath_CreatesInstance()
    {
        // Arrange
        var cliPath = "path/to/MLoop.CLI.csproj";

        // Act
        var manager = new MLoopProjectManager(cliPath);

        // Assert
        Assert.NotNull(manager);
    }

    [Fact]
    public void MLoopProjectManager_WithBothPaths_CreatesInstance()
    {
        // Arrange
        var cliPath = "path/to/MLoop.CLI.csproj";
        var dotnetPath = "/usr/bin/dotnet";

        // Act
        var manager = new MLoopProjectManager(cliPath, dotnetPath);

        // Assert
        Assert.NotNull(manager);
    }

    #endregion

    #region ProjectManager Method Tests (Integration - Skip unless CLI available)

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task InitializeProjectAsync_ValidConfig_ReturnsResult()
    {
        // Arrange
        var tempDataPath = CreateTempCsvFile();
        var config = new MLoopProjectConfig
        {
            ProjectName = "test-project",
            DataPath = tempDataPath,
            LabelColumn = "target",
            TaskType = "binary-classification",
            ProjectDirectory = Path.Combine(Path.GetTempPath(), $"mloop_test_{Guid.NewGuid()}")
        };

        try
        {
            // Act
            var result = await _projectManager.InitializeProjectAsync(config);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.ExitCode >= 0);
        }
        finally
        {
            File.Delete(tempDataPath);
            CleanupTestDirectory(config.ProjectDirectory);
        }
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task TrainModelAsync_ValidConfig_ReturnsResult()
    {
        // Arrange
        var config = new MLoopTrainingConfig
        {
            TimeSeconds = 30,
            Metric = "Accuracy"
        };

        // Act
        var result = await _projectManager.TrainModelAsync(config);

        // Assert
        Assert.NotNull(result);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task EvaluateModelAsync_WithDefaults_ReturnsResult()
    {
        // Act
        var result = await _projectManager.EvaluateModelAsync();

        // Assert
        Assert.NotNull(result);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task ListExperimentsAsync_ReturnsExperimentList()
    {
        // Act
        var experiments = await _projectManager.ListExperimentsAsync();

        // Assert
        Assert.NotNull(experiments);
        Assert.IsType<List<MLoopExperiment>>(experiments);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task PreprocessDataAsync_ValidPath_ReturnsResult()
    {
        // Arrange
        var tempPath = Path.GetTempPath();

        // Act
        var result = await _projectManager.PreprocessDataAsync(tempPath);

        // Assert
        Assert.NotNull(result);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task PredictAsync_ValidPaths_ReturnsResult()
    {
        // Arrange
        var inputPath = CreateTempCsvFile();
        var outputPath = Path.Combine(Path.GetTempPath(), $"predictions_{Guid.NewGuid()}.csv");

        try
        {
            // Act
            var result = await _projectManager.PredictAsync(inputPath, outputPath);

            // Assert
            Assert.NotNull(result);
        }
        finally
        {
            File.Delete(inputPath);
            if (File.Exists(outputPath)) File.Delete(outputPath);
        }
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task PromoteExperimentAsync_ValidId_ReturnsResult()
    {
        // Arrange
        var experimentId = "test-exp-123";

        // Act
        var result = await _projectManager.PromoteExperimentAsync(experimentId);

        // Assert
        Assert.NotNull(result);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task GetDatasetInfoAsync_ValidPath_ReturnsResult()
    {
        // Arrange
        var dataPath = CreateTempCsvFile();

        try
        {
            // Act
            var result = await _projectManager.GetDatasetInfoAsync(dataPath);

            // Assert
            Assert.NotNull(result);
        }
        finally
        {
            File.Delete(dataPath);
        }
    }

    #endregion

    #region Task Type Validation Tests

    [Theory]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    [InlineData("regression")]
    public void MLoopProjectConfig_ValidTaskTypes_CanBeInitialized(string taskType)
    {
        // Act
        var config = new MLoopProjectConfig
        {
            ProjectName = "test",
            DataPath = "data.csv",
            LabelColumn = "label",
            TaskType = taskType
        };

        // Assert
        Assert.Equal(taskType, config.TaskType);
    }

    [Theory]
    [InlineData("Accuracy")]
    [InlineData("F1Score")]
    [InlineData("AUC")]
    [InlineData("RSquared")]
    [InlineData("RMSE")]
    [InlineData("MAE")]
    public void MLoopTrainingConfig_ValidMetrics_CanBeInitialized(string metric)
    {
        // Act
        var config = new MLoopTrainingConfig
        {
            TimeSeconds = 60,
            Metric = metric
        };

        // Assert
        Assert.Equal(metric, config.Metric);
    }

    #endregion

    #region Helper Methods

    private static string CreateTempCsvFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.csv");
        File.WriteAllText(path, @"feature1,feature2,target
1.0,2.0,A
3.0,4.0,B
5.0,6.0,A
7.0,8.0,B
9.0,10.0,A");
        return path;
    }

    private static void CleanupTestDirectory(string? path)
    {
        if (path != null && Directory.Exists(path))
        {
            try
            {
                Directory.Delete(path, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    #endregion
}
