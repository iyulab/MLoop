using MLoop.CLI.Infrastructure.Configuration;
using MLoop.Core.Preprocessing;

namespace MLoop.Tests.Configuration;

// ConfigValidator's parallel Validate(MLoopConfig)/ValidateLabelInCsv config validator was a
// production-orphan (only these tests called it) that had drifted from the live ValidateCommand;
// it was removed in Cycle 51 and its unique required-field coverage (forecasting/ranking/
// recommendation) ported into ValidateCommand (see ValidateCommandTests). Only ValidatePrepSteps
// remains live (used by PrepRunCommand), so only its tests remain here.
public class ConfigValidatorTests
{
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

    [Fact]
    public void ValidatePrepSteps_SampleStep_ValidRandomConfig_NoErrors()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "sample", Method = "random", Count = 1000 }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Empty(errors);
    }

    [Fact]
    public void ValidatePrepSteps_SampleStep_ValidStratifiedConfig_NoErrors()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "sample", Method = "stratified", Count = 1000, Column = "label" }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Empty(errors);
    }

    [Fact]
    public void ValidatePrepSteps_SampleStep_MissingCount_ReturnsError()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "sample", Method = "random", Count = 0 }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Single(errors);
        Assert.Contains("count", errors[0], StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ValidatePrepSteps_SampleStep_StratifiedWithoutColumn_ReturnsError()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "sample", Method = "stratified", Count = 1000 }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Single(errors);
        Assert.Contains("column", errors[0], StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ValidatePrepSteps_SampleStep_InvalidMethod_ReturnsError()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "sample", Method = "invalid", Count = 1000 }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Single(errors);
        Assert.Contains("method", errors[0], StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ValidatePrepSteps_DataSamplingAlias_ValidConfig_NoErrors()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "data-sampling", Method = "random", Count = 500 }
        };

        var errors = ConfigValidator.ValidatePrepSteps(steps);

        Assert.Empty(errors);
    }
}
