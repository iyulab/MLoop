using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.Core.Preprocessing;

namespace MLoop.Tests.Commands;

public class ValidateCommandTests
{
    #region IsValidModelName

    [Theory]
    [InlineData("default", true)]
    [InlineData("my-model", true)]
    [InlineData("my_model", true)]
    [InlineData("_private", true)]
    [InlineData("a", true)]
    [InlineData("model123", true)]
    [InlineData("123model", false)]    // starts with digit
    [InlineData("-model", false)]       // starts with hyphen
    [InlineData("my model", false)]     // contains space
    [InlineData("my.model", false)]     // contains dot
    [InlineData("", false)]             // empty
    [InlineData("  ", false)]           // whitespace
    public void IsValidModelName_ValidatesCorrectly(string name, bool expected)
    {
        Assert.Equal(expected, ValidateCommand.IsValidModelName(name));
    }

    #endregion

    #region ValidateModel — Task Type

    [Theory]
    [InlineData("regression")]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    [InlineData("anomaly-detection")]
    [InlineData("clustering")]
    [InlineData("ranking")]
    [InlineData("forecasting")]
    [InlineData("time-series-anomaly")]
    [InlineData("recommendation")]
    [InlineData("image-classification")]
    [InlineData("object-detection")]
    [InlineData("text-classification")]
    [InlineData("sentence-similarity")]
    [InlineData("ner")]
    [InlineData("question-answering")]
    public void ValidateModel_AcceptsAllValidTaskTypes(string taskType)
    {
        var model = new ModelDefinition { Task = taskType, Label = "Y" };
        var (errors, _) = RunValidateModel("test", model);

        Assert.DoesNotContain(errors, e => e.Path.Contains(".task"));
    }

    [Fact]
    public void ValidateModel_RejectsInvalidTaskType()
    {
        var model = new ModelDefinition { Task = "unknown-task", Label = "Y" };
        var (errors, _) = RunValidateModel("test", model);

        Assert.Contains(errors, e => e.Path == "models.test.task" && e.Message.Contains("Invalid task type"));
    }

    [Fact]
    public void ValidateModel_RequiresTaskType()
    {
        var model = new ModelDefinition { Task = "", Label = "Y" };
        var (errors, _) = RunValidateModel("test", model);

        Assert.Contains(errors, e => e.Path == "models.test.task" && e.Message.Contains("required"));
    }

    #endregion

    #region ValidateModel — Label

    [Fact]
    public void ValidateModel_RequiresLabel()
    {
        var model = new ModelDefinition { Task = "regression", Label = "" };
        var (errors, _) = RunValidateModel("test", model);

        Assert.Contains(errors, e => e.Path == "models.test.label" && e.Message.Contains("required"));
    }

    [Fact]
    public void ValidateModel_AcceptsValidLabel()
    {
        var model = new ModelDefinition { Task = "regression", Label = "target" };
        var (errors, _) = RunValidateModel("test", model);

        Assert.DoesNotContain(errors, e => e.Path.Contains(".label"));
    }

    #endregion

    #region ValidateModel — Column Overrides

    [Theory]
    [InlineData("text")]
    [InlineData("categorical")]
    [InlineData("numeric")]
    [InlineData("ignore")]
    public void ValidateModel_AcceptsValidColumnTypes(string columnType)
    {
        var model = new ModelDefinition
        {
            Task = "regression",
            Label = "Y",
            Columns = new() { { "col1", new ColumnOverride { Type = columnType } } }
        };
        var (errors, _) = RunValidateModel("test", model);

        Assert.DoesNotContain(errors, e => e.Path.Contains("columns"));
    }

    [Fact]
    public void ValidateModel_RejectsInvalidColumnType()
    {
        var model = new ModelDefinition
        {
            Task = "regression",
            Label = "Y",
            Columns = new() { { "col1", new ColumnOverride { Type = "invalid" } } }
        };
        var (errors, _) = RunValidateModel("test", model);

        Assert.Contains(errors, e => e.Path.Contains("col1") && e.Message.Contains("Invalid column type"));
    }

    [Fact]
    public void ValidateModel_WarnsWhenLabelColumnOverridden()
    {
        var model = new ModelDefinition
        {
            Task = "regression",
            Label = "Y",
            Columns = new() { { "Y", new ColumnOverride { Type = "numeric" } } }
        };
        var (_, warnings) = RunValidateModel("test", model);

        Assert.Contains(warnings, w => w.Path.Contains("columns.Y") && w.Message.Contains("label column"));
    }

    #endregion

    #region ValidateTrainingSettings

    [Fact]
    public void ValidateTrainingSettings_RejectsZeroTimeLimit()
    {
        var training = new TrainingSettings { TimeLimitSeconds = 0 };
        var (errors, _) = RunValidateTraining(training);

        Assert.Contains(errors, e => e.Message.Contains("greater than 0"));
    }

    [Fact]
    public void ValidateTrainingSettings_RejectsNegativeTimeLimit()
    {
        var training = new TrainingSettings { TimeLimitSeconds = -1 };
        var (errors, _) = RunValidateTraining(training);

        Assert.Contains(errors, e => e.Message.Contains("greater than 0"));
    }

    [Fact]
    public void ValidateTrainingSettings_WarnsOnShortTimeLimit()
    {
        var training = new TrainingSettings { TimeLimitSeconds = 10 };
        var (errors, warnings) = RunValidateTraining(training);

        Assert.Empty(errors);
        Assert.Contains(warnings, w => w.Message.Contains("less than 30 seconds"));
    }

    [Fact]
    public void ValidateTrainingSettings_WarnsOnLongTimeLimit()
    {
        var training = new TrainingSettings { TimeLimitSeconds = 7200 };
        var (errors, warnings) = RunValidateTraining(training);

        Assert.Empty(errors);
        Assert.Contains(warnings, w => w.Message.Contains("over 1 hour"));
    }

    [Fact]
    public void ValidateTrainingSettings_AcceptsReasonableTimeLimit()
    {
        var training = new TrainingSettings { TimeLimitSeconds = 300 };
        var (errors, warnings) = RunValidateTraining(training);

        Assert.Empty(errors);
        Assert.Empty(warnings);
    }

    [Fact]
    public void ValidateTrainingSettings_RejectsTestSplitOutOfRange()
    {
        var training = new TrainingSettings { TestSplit = 0 };
        var (errors, _) = RunValidateTraining(training);
        Assert.Contains(errors, e => e.Message.Contains("between 0 and 1"));

        (errors, _) = RunValidateTraining(new TrainingSettings { TestSplit = 1.0 });
        Assert.Contains(errors, e => e.Message.Contains("between 0 and 1"));

        (errors, _) = RunValidateTraining(new TrainingSettings { TestSplit = -0.1 });
        Assert.Contains(errors, e => e.Message.Contains("between 0 and 1"));
    }

    [Fact]
    public void ValidateTrainingSettings_WarnsOnExtremeTestSplit()
    {
        var (_, warnings) = RunValidateTraining(new TrainingSettings { TestSplit = 0.05 });
        Assert.Contains(warnings, w => w.Message.Contains("less than 0.1"));

        (_, warnings) = RunValidateTraining(new TrainingSettings { TestSplit = 0.6 });
        Assert.Contains(warnings, w => w.Message.Contains("greater than 0.5"));
    }

    [Fact]
    public void ValidateTrainingSettings_AcceptsReasonableTestSplit()
    {
        var (errors, warnings) = RunValidateTraining(new TrainingSettings { TestSplit = 0.2 });
        Assert.Empty(errors);
        Assert.Empty(warnings);
    }

    [Fact]
    public void ValidateTrainingSettings_WarnsOnUnknownMetric()
    {
        var (_, warnings) = RunValidateTraining(new TrainingSettings { Metric = "nonexistent" });
        Assert.Contains(warnings, w => w.Message.Contains("Unknown metric"));
    }

    [Theory]
    [InlineData("auto")]
    [InlineData("accuracy")]
    [InlineData("auc")]
    [InlineData("r2")]
    [InlineData("rmse")]
    public void ValidateTrainingSettings_AcceptsKnownMetrics(string metric)
    {
        var (_, warnings) = RunValidateTraining(new TrainingSettings { Metric = metric });
        Assert.DoesNotContain(warnings, w => w.Message.Contains("Unknown metric"));
    }

    #endregion

    #region ValidatePrepSteps

    [Fact]
    public void ValidatePrepSteps_RequiresStepType()
    {
        var steps = new List<PrepStep> { new() { Type = "" } };
        var (errors, _) = RunValidatePrepSteps(steps);

        Assert.Contains(errors, e => e.Message.Contains("type is required"));
    }

    [Fact]
    public void ValidatePrepSteps_RejectsUnknownStepType()
    {
        var steps = new List<PrepStep> { new() { Type = "unknown-step" } };
        var (errors, _) = RunValidatePrepSteps(steps);

        Assert.Contains(errors, e => e.Message.Contains("Unknown prep step type"));
    }

    [Fact]
    public void ValidatePrepSteps_FillMissing_RequiresColumns()
    {
        var steps = new List<PrepStep> { new() { Type = "fill-missing" } };
        var (errors, _) = RunValidatePrepSteps(steps);

        Assert.Contains(errors, e => e.Message.Contains("'columns' or 'column'"));
    }

    [Fact]
    public void ValidatePrepSteps_FillMissing_AcceptsColumnOrColumns()
    {
        // Single column
        var steps1 = new List<PrepStep> { new() { Type = "fill-missing", Column = "col1" } };
        var (errors1, _) = RunValidatePrepSteps(steps1);
        Assert.DoesNotContain(errors1, e => e.Message.Contains("'columns' or 'column'"));

        // Multiple columns
        var steps2 = new List<PrepStep> { new() { Type = "fill-missing", Columns = new() { "a", "b" } } };
        var (errors2, _) = RunValidatePrepSteps(steps2);
        Assert.DoesNotContain(errors2, e => e.Message.Contains("'columns' or 'column'"));
    }

    [Fact]
    public void ValidatePrepSteps_RenameColumns_RequiresMapping()
    {
        var steps = new List<PrepStep> { new() { Type = "rename-columns" } };
        var (errors, _) = RunValidatePrepSteps(steps);

        Assert.Contains(errors, e => e.Message.Contains("'mapping'"));
    }

    [Fact]
    public void ValidatePrepSteps_ParseDatetime_RequiresColumnAndFormat()
    {
        var steps = new List<PrepStep> { new() { Type = "parse-datetime" } };
        var (errors, _) = RunValidatePrepSteps(steps);

        Assert.Contains(errors, e => e.Message.Contains("'column'"));
        Assert.Contains(errors, e => e.Message.Contains("'format'"));
    }

    [Fact]
    public void ValidatePrepSteps_FilterRows_RequiresAllParams()
    {
        var steps = new List<PrepStep> { new() { Type = "filter-rows" } };
        var (errors, _) = RunValidatePrepSteps(steps);

        Assert.Contains(errors, e => e.Message.Contains("'column'"));
        Assert.Contains(errors, e => e.Message.Contains("'operator'"));
        Assert.Contains(errors, e => e.Message.Contains("'value'"));
    }

    [Fact]
    public void ValidatePrepSteps_Rolling_RequiresWindowAndColumns()
    {
        var steps = new List<PrepStep> { new() { Type = "rolling", WindowSize = 0 } };
        var (errors, _) = RunValidatePrepSteps(steps);

        Assert.Contains(errors, e => e.Message.Contains("'window_size' > 0"));
        Assert.Contains(errors, e => e.Message.Contains("'columns'"));
    }

    [Fact]
    public void ValidatePrepSteps_Resample_RequiresTimeColumnAndWindow()
    {
        var steps = new List<PrepStep> { new() { Type = "resample" } };
        var (errors, _) = RunValidatePrepSteps(steps);

        Assert.Contains(errors, e => e.Message.Contains("'time_column'"));
        Assert.Contains(errors, e => e.Message.Contains("'window'"));
        Assert.Contains(errors, e => e.Message.Contains("'columns'"));
    }

    [Fact]
    public void ValidatePrepSteps_AcceptsUnderscoreVariant()
    {
        var steps = new List<PrepStep>
        {
            new() { Type = "fill_missing", Columns = new() { "a" } }
        };
        var (errors, _) = RunValidatePrepSteps(steps);

        Assert.DoesNotContain(errors, e => e.Message.Contains("Unknown"));
    }

    #endregion

    #region ValidateModel — Model Name

    [Fact]
    public void ValidateModel_RejectsInvalidModelName()
    {
        var model = new ModelDefinition { Task = "regression", Label = "Y" };
        var (errors, _) = RunValidateModel("123invalid", model);

        Assert.Contains(errors, e => e.Message.Contains("Model name must be"));
    }

    [Fact]
    public void ValidateModel_RejectsEmptyModelName()
    {
        var model = new ModelDefinition { Task = "regression", Label = "Y" };
        var (errors, _) = RunValidateModel("", model);

        Assert.Contains(errors, e => e.Message.Contains("empty"));
    }

    #endregion

    #region Helpers

    private static (List<ValidateCommand.ValidationError> errors, List<ValidateCommand.ValidationWarning> warnings)
        RunValidateModel(string name, ModelDefinition model)
    {
        var errors = new List<ValidateCommand.ValidationError>();
        var warnings = new List<ValidateCommand.ValidationWarning>();
        ValidateCommand.ValidateModel(name, model, errors, warnings);
        return (errors, warnings);
    }

    private static (List<ValidateCommand.ValidationError> errors, List<ValidateCommand.ValidationWarning> warnings)
        RunValidateTraining(TrainingSettings training)
    {
        var errors = new List<ValidateCommand.ValidationError>();
        var warnings = new List<ValidateCommand.ValidationWarning>();
        ValidateCommand.ValidateTrainingSettings("test", training, errors, warnings);
        return (errors, warnings);
    }

    private static (List<ValidateCommand.ValidationError> errors, List<ValidateCommand.ValidationWarning> warnings)
        RunValidatePrepSteps(List<PrepStep> steps)
    {
        var errors = new List<ValidateCommand.ValidationError>();
        var warnings = new List<ValidateCommand.ValidationWarning>();
        ValidateCommand.ValidatePrepSteps("test.prep", steps, errors, warnings);
        return (errors, warnings);
    }

    #endregion
}
