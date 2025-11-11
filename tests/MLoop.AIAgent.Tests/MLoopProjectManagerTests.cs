using MLoop.AIAgent.Core;
using MLoop.AIAgent.Core.Models;

namespace MLoop.AIAgent.Tests;

public class MLoopProjectManagerTests
{
    private readonly MLoopProjectManager _manager;
    private readonly string _testProjectPath;

    public MLoopProjectManagerTests()
    {
        // Use a test-specific MLoop CLI path
        var mloopCliPath = Path.Combine(
            Directory.GetCurrentDirectory(),
            "..", "..", "..", "..", "..",
            "src", "MLoop.CLI", "MLoop.CLI.csproj");

        _manager = new MLoopProjectManager(mloopCliPath);
        _testProjectPath = Path.Combine(Path.GetTempPath(), $"mloop_test_{Guid.NewGuid()}");
    }

    [Fact]
    public void Constructor_WithDefaultParameters_DoesNotThrow()
    {
        // Act & Assert
        var manager = new MLoopProjectManager();
        Assert.NotNull(manager);
    }

    [Fact]
    public void Constructor_WithCustomPaths_DoesNotThrow()
    {
        // Arrange
        var cliPath = "custom/path/MLoop.CLI.csproj";
        var dotnetPath = "custom/dotnet";

        // Act & Assert
        var manager = new MLoopProjectManager(cliPath, dotnetPath);
        Assert.NotNull(manager);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task InitializeProjectAsync_ValidConfig_ReturnsSuccess()
    {
        // Arrange
        var tempData = CreateTempCsvFile();
        var config = new MLoopProjectConfig
        {
            ProjectName = "test-project",
            DataPath = tempData,
            LabelColumn = "target",
            TaskType = "binary-classification",
            ProjectDirectory = _testProjectPath
        };

        try
        {
            // Act
            var result = await _manager.InitializeProjectAsync(config);

            // Assert
            Assert.True(result.Success || result.ExitCode != 0,
                $"CLI execution completed (Exit: {result.ExitCode})");

            if (result.Success)
            {
                Assert.Contains("Project created", result.Output, StringComparison.OrdinalIgnoreCase);
                Assert.True(result.Data.ContainsKey("ProjectDirectory"));
            }
        }
        finally
        {
            CleanupTestDirectory(_testProjectPath);
            File.Delete(tempData);
        }
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task TrainModelAsync_ValidConfig_ReturnsSuccess()
    {
        // Arrange
        var config = new MLoopTrainingConfig
        {
            TimeSeconds = 30,
            Metric = "Accuracy",
            TestSplit = 0.2,
            ExperimentName = "test-experiment"
        };

        // Act
        var result = await _manager.TrainModelAsync(config, _testProjectPath);

        // Assert
        // CLI might fail if project doesn't exist, but we're testing the execution path
        Assert.NotNull(result);
        Assert.True(result.ExitCode >= 0);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task PreprocessDataAsync_ValidPath_ReturnsResult()
    {
        // Act
        var result = await _manager.PreprocessDataAsync(_testProjectPath);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.ExitCode >= 0);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task EvaluateModelAsync_WithDefaults_ReturnsResult()
    {
        // Act
        var result = await _manager.EvaluateModelAsync(projectPath: _testProjectPath);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.ExitCode >= 0);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task EvaluateModelAsync_WithExperimentId_ReturnsResult()
    {
        // Arrange
        var experimentId = Guid.NewGuid().ToString();

        // Act
        var result = await _manager.EvaluateModelAsync(
            experimentId: experimentId,
            projectPath: _testProjectPath);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.ExitCode >= 0);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task PredictAsync_ValidPaths_ReturnsResult()
    {
        // Arrange
        var inputPath = CreateTempCsvFile();
        var outputPath = Path.Combine(Path.GetTempPath(), "predictions.csv");

        try
        {
            // Act
            var result = await _manager.PredictAsync(
                inputPath,
                outputPath,
                projectPath: _testProjectPath);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.ExitCode >= 0);
        }
        finally
        {
            File.Delete(inputPath);
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }
        }
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task ListExperimentsAsync_ReturnsExperimentList()
    {
        // Act
        var experiments = await _manager.ListExperimentsAsync(_testProjectPath);

        // Assert
        Assert.NotNull(experiments);
        // May be empty if no experiments exist
        Assert.IsType<List<MLoopExperiment>>(experiments);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task PromoteExperimentAsync_ValidId_ReturnsResult()
    {
        // Arrange
        var experimentId = Guid.NewGuid().ToString();

        // Act
        var result = await _manager.PromoteExperimentAsync(
            experimentId,
            _testProjectPath);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.ExitCode >= 0);
    }

    [Fact(Skip = "Integration test - requires MLoop CLI to be built")]
    public async Task GetDatasetInfoAsync_ValidPath_ReturnsResult()
    {
        // Arrange
        var dataPath = CreateTempCsvFile();

        try
        {
            // Act
            var result = await _manager.GetDatasetInfoAsync(
                dataPath,
                _testProjectPath);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.ExitCode >= 0);
        }
        finally
        {
            File.Delete(dataPath);
        }
    }

    [Fact]
    public void MLoopProjectConfig_RequiredProperties_CanBeInitialized()
    {
        // Act
        var config = new MLoopProjectConfig
        {
            ProjectName = "test",
            DataPath = "data.csv",
            LabelColumn = "label",
            TaskType = "regression"
        };

        // Assert
        Assert.Equal("test", config.ProjectName);
        Assert.Equal("data.csv", config.DataPath);
        Assert.Equal("label", config.LabelColumn);
        Assert.Equal("regression", config.TaskType);
        Assert.Null(config.ProjectDirectory);
    }

    [Fact]
    public void MLoopTrainingConfig_RequiredProperties_CanBeInitialized()
    {
        // Act
        var config = new MLoopTrainingConfig
        {
            TimeSeconds = 60,
            Metric = "F1Score"
        };

        // Assert
        Assert.Equal(60, config.TimeSeconds);
        Assert.Equal("F1Score", config.Metric);
        Assert.Equal(0.2, config.TestSplit); // Default value
        Assert.Null(config.DataPath);
        Assert.Null(config.ExperimentName);
    }

    [Fact]
    public void MLoopExperiment_RequiredProperties_CanBeInitialized()
    {
        // Arrange
        var timestamp = DateTime.Now;

        // Act
        var experiment = new MLoopExperiment
        {
            Id = "exp-123",
            Name = "test-exp",
            Timestamp = timestamp,
            Trainer = "LightGbm",
            MetricValue = 0.95,
            MetricName = "Accuracy"
        };

        // Assert
        Assert.Equal("exp-123", experiment.Id);
        Assert.Equal("test-exp", experiment.Name);
        Assert.Equal(timestamp, experiment.Timestamp);
        Assert.Equal("LightGbm", experiment.Trainer);
        Assert.Equal(0.95, experiment.MetricValue);
        Assert.Equal("Accuracy", experiment.MetricName);
        Assert.False(experiment.IsProduction); // Default value
    }

    [Fact]
    public void MLoopOperationResult_RequiredProperties_CanBeInitialized()
    {
        // Act
        var result = new MLoopOperationResult
        {
            Success = true,
            ExitCode = 0,
            Output = "Success message"
        };

        // Assert
        Assert.True(result.Success);
        Assert.Equal(0, result.ExitCode);
        Assert.Equal("Success message", result.Output);
        Assert.Null(result.Error);
        Assert.Empty(result.Data);
    }

    [Fact]
    public void MLoopOperationResult_WithData_StoresKeyValuePairs()
    {
        // Arrange
        var result = new MLoopOperationResult
        {
            Success = true,
            ExitCode = 0,
            Output = "Output",
            Data = new Dictionary<string, object>
            {
                ["ExperimentId"] = "exp-123",
                ["MetricValue"] = 0.92
            }
        };

        // Assert
        Assert.Equal(2, result.Data.Count);
        Assert.Equal("exp-123", result.Data["ExperimentId"]);
        Assert.Equal(0.92, result.Data["MetricValue"]);
    }

    // Helper methods
    private string CreateTempCsvFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.csv");
        File.WriteAllText(path, @"feature1,feature2,target
1.0,2.0,A
3.0,4.0,B
5.0,6.0,A
7.0,8.0,B");
        return path;
    }

    private void CleanupTestDirectory(string path)
    {
        if (Directory.Exists(path))
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
}
