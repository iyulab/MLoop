using MLoop.CLI.Infrastructure.Configuration;
using MLoop.Core.Preprocessing;

namespace MLoop.Tests.Configuration;

public class ConfigValidatorTests
{
    [Fact]
    public void Validate_ValidConfig_NoErrors()
    {
        var config = new MLoopConfig
        {
            Project = "test-project",
            Models = new()
            {
                ["default"] = new ModelDefinition
                {
                    Task = "regression",
                    Label = "target"
                }
            }
        };

        var result = ConfigValidator.Validate(config);

        Assert.Empty(result.Errors);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void Validate_MissingProject_ReturnsWarning()
    {
        var config = new MLoopConfig
        {
            Models = new()
            {
                ["default"] = new ModelDefinition { Task = "regression", Label = "target" }
            }
        };

        var result = ConfigValidator.Validate(config);

        Assert.Empty(result.Errors);
        Assert.Single(result.Warnings);
        Assert.Contains("Project name", result.Warnings[0]);
    }

    [Fact]
    public void Validate_NoModels_ReturnsError()
    {
        var config = new MLoopConfig { Project = "test" };

        var result = ConfigValidator.Validate(config);

        Assert.Single(result.Errors);
        Assert.Contains("No models defined", result.Errors[0]);
    }

    [Fact]
    public void Validate_InvalidTaskType_ReturnsError()
    {
        var config = new MLoopConfig
        {
            Project = "test",
            Models = new()
            {
                ["default"] = new ModelDefinition { Task = "invalid-task", Label = "target" }
            }
        };

        var result = ConfigValidator.Validate(config);

        Assert.Single(result.Errors);
        Assert.Contains("Invalid task type", result.Errors[0]);
    }

    [Fact]
    public void Validate_MissingLabel_ReturnsError()
    {
        var config = new MLoopConfig
        {
            Project = "test",
            Models = new()
            {
                ["default"] = new ModelDefinition { Task = "regression", Label = "" }
            }
        };

        var result = ConfigValidator.Validate(config);

        Assert.Single(result.Errors);
        Assert.Contains("Label column is required", result.Errors[0]);
    }

    // PrepStep validation tests

    [Fact]
    public void ValidatePrepSteps_ValidSteps_NoErrors()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "fill-missing", Columns = ["col1"], Method = "mean" },
            new() { Type = "normalize", Column = "col2", Method = "min-max" },
            new() { Type = "drop-duplicates" }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Empty(errors);
    }

    [Fact]
    public void ValidatePrepSteps_UnknownType_ReturnsError()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "nonexistent-type" }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Single(errors);
        Assert.Contains("Unknown prep step type", errors[0]);
    }

    [Fact]
    public void ValidatePrepSteps_EmptyType_ReturnsError()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "" }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Single(errors);
        Assert.Contains("Step type is required", errors[0]);
    }

    [Fact]
    public void ValidatePrepSteps_FillMissingWithoutColumns_ReturnsError()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "fill-missing", Method = "mean" }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Single(errors);
        Assert.Contains("requires 'columns' or 'column'", errors[0]);
    }

    [Fact]
    public void ValidatePrepSteps_RenameColumnsWithoutMapping_ReturnsError()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "rename-columns" }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Single(errors);
        Assert.Contains("requires 'mapping'", errors[0]);
    }

    [Fact]
    public void ValidatePrepSteps_ParseDatetimeWithoutFormat_ReturnsError()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "parse-datetime", Column = "date_col" }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Single(errors);
        Assert.Contains("requires 'format'", errors[0]);
    }

    [Fact]
    public void ValidatePrepSteps_FilterRowsMissingAllParams_ReturnsMultipleErrors()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "filter-rows" }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Equal(3, errors.Count);
        Assert.Contains(errors, e => e.Contains("requires 'column'"));
        Assert.Contains(errors, e => e.Contains("requires 'operator'"));
        Assert.Contains(errors, e => e.Contains("requires 'value'"));
    }

    [Fact]
    public void ValidatePrepSteps_RollingWithoutWindowSize_ReturnsError()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "rolling", Columns = ["value"] }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Single(errors);
        Assert.Contains("requires 'window_size' > 0", errors[0]);
    }

    [Fact]
    public void ValidatePrepSteps_ResampleWithoutTimeColumn_ReturnsError()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "resample", Columns = ["value"], Window = "1H" }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Single(errors);
        Assert.Contains("requires 'time_column'", errors[0]);
    }

    [Fact]
    public void ValidatePrepSteps_AddColumnWithoutParams_ReturnsErrors()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "add-column" }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Equal(2, errors.Count);
        Assert.Contains(errors, e => e.Contains("requires 'column'"));
        Assert.Contains(errors, e => e.Contains("requires 'value' or 'expression'"));
    }

    [Fact]
    public void ValidatePrepSteps_UnderscoreVariant_Accepted()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "fill_missing", Columns = ["col1"] },
            new() { Type = "remove_columns", Columns = ["col2"] },
            new() { Type = "parse_datetime", Column = "date", Format = "yyyy-MM-dd" }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Empty(errors);
    }

    // CSV label validation tests

    [Fact]
    public void ValidateLabelInCsv_LabelExists_NoErrors()
    {
        var headers = new[] { "id", "feature1", "target" };
        var models = new Dictionary<string, ModelDefinition>
        {
            ["default"] = new() { Task = "regression", Label = "target" }
        };

        var errors = ConfigValidator.ValidateLabelInCsv(headers, models);

        Assert.Empty(errors);
    }

    [Fact]
    public void ValidateLabelInCsv_LabelMissing_ReturnsError()
    {
        var headers = new[] { "id", "feature1", "feature2" };
        var models = new Dictionary<string, ModelDefinition>
        {
            ["default"] = new() { Task = "regression", Label = "target" }
        };

        var errors = ConfigValidator.ValidateLabelInCsv(headers, models);

        Assert.Single(errors);
        Assert.Contains("'target' not found in CSV", errors[0]);
        Assert.Contains("id, feature1, feature2", errors[0]);
    }

    [Fact]
    public void ValidateLabelInCsv_CaseInsensitive_NoErrors()
    {
        var headers = new[] { "ID", "Feature1", "Target" };
        var models = new Dictionary<string, ModelDefinition>
        {
            ["default"] = new() { Task = "regression", Label = "target" }
        };

        var errors = ConfigValidator.ValidateLabelInCsv(headers, models);

        Assert.Empty(errors);
    }

    [Fact]
    public void ValidateLabelInCsv_MultipleModels_ValidatesAll()
    {
        var headers = new[] { "id", "score", "label" };
        var models = new Dictionary<string, ModelDefinition>
        {
            ["model-a"] = new() { Task = "regression", Label = "score" },
            ["model-b"] = new() { Task = "binary-classification", Label = "missing_col" }
        };

        var errors = ConfigValidator.ValidateLabelInCsv(headers, models);

        Assert.Single(errors);
        Assert.Contains("model-b", errors[0]);
        Assert.Contains("missing_col", errors[0]);
    }
}
