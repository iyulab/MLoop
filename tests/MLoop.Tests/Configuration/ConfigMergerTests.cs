using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Configuration;

public class ConfigMergerTests
{
    private readonly ConfigMerger _merger = new();

    #region Merge Tests

    [Fact]
    public void Merge_AllNull_ReturnsDefaults()
    {
        var result = _merger.Merge(null, null, null);

        Assert.NotNull(result);
        Assert.Empty(result.Models);
    }

    [Fact]
    public void Merge_ProjectConfigOnly_AppliesProjectConfig()
    {
        var projectConfig = new MLoopConfig
        {
            Project = "my-project",
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "regression", Label = "Price" }
            }
        };

        var result = _merger.Merge(projectConfig: projectConfig);

        Assert.Equal("my-project", result.Project);
        Assert.Single(result.Models);
        Assert.Equal("regression", result.Models["default"].Task);
    }

    [Fact]
    public void Merge_UserOverridesProject()
    {
        var projectConfig = new MLoopConfig
        {
            Project = "project-name",
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "regression", Label = "Price" }
            }
        };
        var userConfig = new MLoopConfig
        {
            Project = "user-name",
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "binary-classification", Label = "Target" }
            }
        };

        var result = _merger.Merge(userConfig: userConfig, projectConfig: projectConfig);

        Assert.Equal("user-name", result.Project);
        Assert.Equal("binary-classification", result.Models["default"].Task);
        Assert.Equal("Target", result.Models["default"].Label);
    }

    [Fact]
    public void Merge_CliOverridesAll()
    {
        var projectConfig = new MLoopConfig { Project = "project" };
        var userConfig = new MLoopConfig { Project = "user" };
        var cliConfig = new MLoopConfig { Project = "cli" };

        var result = _merger.Merge(cliConfig, userConfig, projectConfig);

        Assert.Equal("cli", result.Project);
    }

    [Fact]
    public void Merge_ModelsFromMultipleSources_Combined()
    {
        var projectConfig = new MLoopConfig
        {
            Models = new Dictionary<string, ModelDefinition>
            {
                ["model-a"] = new() { Task = "regression", Label = "A" }
            }
        };
        var userConfig = new MLoopConfig
        {
            Models = new Dictionary<string, ModelDefinition>
            {
                ["model-b"] = new() { Task = "binary-classification", Label = "B" }
            }
        };

        var result = _merger.Merge(userConfig: userConfig, projectConfig: projectConfig);

        Assert.Equal(2, result.Models.Count);
        Assert.True(result.Models.ContainsKey("model-a"));
        Assert.True(result.Models.ContainsKey("model-b"));
    }

    [Fact]
    public void Merge_DataSettings_Merged()
    {
        var projectConfig = new MLoopConfig
        {
            Data = new DataSettings { Train = "project-train.csv" }
        };
        var userConfig = new MLoopConfig
        {
            Data = new DataSettings { Train = "user-train.csv", Test = "test.csv" }
        };

        var result = _merger.Merge(userConfig: userConfig, projectConfig: projectConfig);

        Assert.Equal("user-train.csv", result.Data?.Train);
        Assert.Equal("test.csv", result.Data?.Test);
    }

    #endregion

    #region GetEffectiveModelDefinition Tests

    [Fact]
    public void GetEffectiveModelDefinition_FromConfig_ReturnsDefinition()
    {
        var config = new MLoopConfig
        {
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "regression", Label = "Price" }
            }
        };

        var result = _merger.GetEffectiveModelDefinition(config, "default");

        Assert.Equal("regression", result.Task);
        Assert.Equal("Price", result.Label);
    }

    [Fact]
    public void GetEffectiveModelDefinition_CliOverridesConfig()
    {
        var config = new MLoopConfig
        {
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "regression", Label = "Price" }
            }
        };

        var result = _merger.GetEffectiveModelDefinition(
            config, "default",
            cliLabel: "Target",
            cliTask: "binary-classification");

        Assert.Equal("binary-classification", result.Task);
        Assert.Equal("Target", result.Label);
    }

    [Fact]
    public void GetEffectiveModelDefinition_MissingTask_Throws()
    {
        var config = new MLoopConfig
        {
            Models = new Dictionary<string, ModelDefinition>()
        };

        Assert.Throws<InvalidOperationException>(() =>
            _merger.GetEffectiveModelDefinition(config, "missing", cliLabel: "Label"));
    }

    [Fact]
    public void GetEffectiveModelDefinition_MissingLabel_Throws()
    {
        var config = new MLoopConfig
        {
            Models = new Dictionary<string, ModelDefinition>()
        };

        Assert.Throws<InvalidOperationException>(() =>
            _merger.GetEffectiveModelDefinition(config, "missing", cliTask: "regression"));
    }

    [Fact]
    public void GetEffectiveModelDefinition_DefaultTrainingSettings()
    {
        var config = new MLoopConfig
        {
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "regression", Label = "Price" }
            }
        };

        var result = _merger.GetEffectiveModelDefinition(config, "default");

        Assert.Equal(ConfigDefaults.DefaultTimeLimitSeconds, result.Training?.TimeLimitSeconds);
        Assert.Equal(ConfigDefaults.DefaultMetric, result.Training?.Metric);
        Assert.Equal(ConfigDefaults.DefaultTestSplit, result.Training?.TestSplit);
    }

    #endregion

    #region MergeTrainingSettings Tests

    [Fact]
    public void MergeTrainingSettings_AllNull_ReturnsDefaults()
    {
        var result = _merger.MergeTrainingSettings(null, null, null);

        Assert.Equal(ConfigDefaults.DefaultTimeLimitSeconds, result.TimeLimitSeconds);
        Assert.Equal(ConfigDefaults.DefaultMetric, result.Metric);
        Assert.Equal(ConfigDefaults.DefaultTestSplit, result.TestSplit);
    }

    [Fact]
    public void MergeTrainingSettings_ModelOverridesDefaults()
    {
        var defaults = new TrainingSettings
        {
            TimeLimitSeconds = 300,
            Metric = "auto",
            TestSplit = 0.2
        };
        var modelSettings = new TrainingSettings
        {
            TimeLimitSeconds = 600,
            Metric = "accuracy"
        };

        var result = _merger.MergeTrainingSettings(defaults, modelSettings, null);

        Assert.Equal(600, result.TimeLimitSeconds);
        Assert.Equal("accuracy", result.Metric);
        Assert.Equal(0.2, result.TestSplit); // Defaults preserved for unset
    }

    [Fact]
    public void MergeTrainingSettings_CliOverridesAll()
    {
        var defaults = new TrainingSettings { TimeLimitSeconds = 300 };
        var modelSettings = new TrainingSettings { TimeLimitSeconds = 600 };
        var cliSettings = new TrainingSettings { TimeLimitSeconds = 120 };

        var result = _merger.MergeTrainingSettings(defaults, modelSettings, cliSettings);

        Assert.Equal(120, result.TimeLimitSeconds);
    }

    #endregion
}
