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

    [Theory]
    [InlineData("anomaly-detection")]
    [InlineData("clustering")]
    [InlineData("time-series-anomaly")]
    public void ValidateModel_DoesNotRequireLabelForUnsupervisedTasks(string task)
    {
        // F-19: ConfigMerger/InitCommand/TrainCommand/CsvDataLoader all treat these as label-optional
        // (a dummy label is loaded), so requiring one in validate contradicted what `train` accepts.
        var model = new ModelDefinition { Task = task, Label = "" };
        var (errors, _) = RunValidateModel("test", model);

        Assert.DoesNotContain(errors, e => e.Path.Contains(".label"));
    }

    // F-20: task-specific required fields ported from the removed parallel ConfigValidator. These
    // are mandatory at train time, so validate must catch them (validate↔train parity).

    [Fact]
    public void ValidateModel_ForecastingWithoutHorizon_ReturnsError()
    {
        var model = new ModelDefinition { Task = "forecasting", Label = "value" };
        var (errors, _) = RunValidateModel("test", model);

        Assert.Contains(errors, e => e.Path == "models.test.horizon" && e.Message.Contains("Horizon"));
    }

    [Fact]
    public void ValidateModel_ForecastingWithHorizon_NoHorizonError()
    {
        var model = new ModelDefinition { Task = "forecasting", Label = "value", Horizon = 10 };
        var (errors, _) = RunValidateModel("test", model);

        Assert.DoesNotContain(errors, e => e.Path.Contains(".horizon"));
    }

    [Fact]
    public void ValidateModel_RankingWithoutGroupColumn_ReturnsError()
    {
        var model = new ModelDefinition { Task = "ranking", Label = "relevance" };
        var (errors, _) = RunValidateModel("test", model);

        Assert.Contains(errors, e => e.Path == "models.test.group_column" && e.Message.Contains("Group column"));
    }

    [Fact]
    public void ValidateModel_RankingWithGroupColumn_NoGroupError()
    {
        var model = new ModelDefinition { Task = "ranking", Label = "relevance", GroupColumn = "query_id" };
        var (errors, _) = RunValidateModel("test", model);

        Assert.DoesNotContain(errors, e => e.Path.Contains(".group_column"));
    }

    [Fact]
    public void ValidateModel_RecommendationWithoutColumns_ReturnsBothErrors()
    {
        var model = new ModelDefinition { Task = "recommendation", Label = "rating" };
        var (errors, _) = RunValidateModel("test", model);

        Assert.Contains(errors, e => e.Path == "models.test.user_column" && e.Message.Contains("User column"));
        Assert.Contains(errors, e => e.Path == "models.test.item_column" && e.Message.Contains("Item column"));
    }

    [Fact]
    public void ValidateModel_RecommendationWithColumns_NoColumnErrors()
    {
        var model = new ModelDefinition
        {
            Task = "recommendation", Label = "rating", UserColumn = "user_id", ItemColumn = "item_id"
        };
        var (errors, _) = RunValidateModel("test", model);

        Assert.DoesNotContain(errors, e => e.Path.Contains(".user_column") || e.Path.Contains(".item_column"));
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

    [Theory]
    [InlineData("forecasting")]
    [InlineData("time-series-anomaly")]
    public void ValidateTrainingSettings_WarnsTestSplitIgnoredForTimeSeries(string task)
    {
        // Time-series tasks feed the full series and hold out the last `horizon` rows internally
        // (AutoMLRunner.IsTimeSeriesTask), so test_split is silently inert. Validate must surface it.
        var (errors, warnings) = RunValidateTraining(new TrainingSettings { TestSplit = 0.2 }, task);
        Assert.Empty(errors);
        Assert.Contains(warnings, w => w.Message.Contains("ignored for forecasting"));
        // The heuristic range warnings must not also fire — the value is moot for these tasks.
        Assert.DoesNotContain(warnings, w => w.Message.Contains("less than 0.1"));
        Assert.DoesNotContain(warnings, w => w.Message.Contains("greater than 0.5"));
    }

    [Fact]
    public void ValidateTrainingSettings_StillRejectsOutOfRangeTestSplitForTimeSeries()
    {
        // An out-of-range value is a real config typo even when the task ignores the split.
        var (errors, _) = RunValidateTraining(new TrainingSettings { TestSplit = 1.5 }, "forecasting");
        Assert.Contains(errors, e => e.Message.Contains("between 0 and 1"));
    }

    [Fact]
    public void ValidateTrainingSettings_NoTimeSeriesNoteForTabularTasks()
    {
        // Regression (default) keeps the unperturbed behavior: a reasonable split is clean.
        var (errors, warnings) = RunValidateTraining(new TrainingSettings { TestSplit = 0.2 });
        Assert.Empty(errors);
        Assert.DoesNotContain(warnings, w => w.Message.Contains("ignored for forecasting"));
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
    // F-17: canonical task-specific metrics that `mloop init` writes and AutoML/promotion use
    // (ModelRegistry.DefaultMetricForTask) must validate cleanly — they were flagged "Unknown".
    [InlineData("macro_accuracy")]   // multiclass-classification default
    [InlineData("micro_accuracy")]   // image/text-classification default
    [InlineData("r_squared")]        // regression default (canonical)
    [InlineData("log_loss")]
    [InlineData("f1_score")]
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
        RunValidateTraining(TrainingSettings training, string task = "regression")
    {
        var errors = new List<ValidateCommand.ValidationError>();
        var warnings = new List<ValidateCommand.ValidationWarning>();
        // Default to a non-time-series task so existing test_split assertions stay unperturbed; the
        // time-series no-op note only fires when callers pass forecasting / time-series-anomaly.
        ValidateCommand.ValidateTrainingSettings("test", training, task, errors, warnings);
        return (errors, warnings);
    }

    private static (List<ValidateCommand.ValidationError> errors, List<ValidateCommand.ValidationWarning> warnings)
        RunValidatePrepSteps(List<PrepStep> steps)
    {
        var errors = new List<ValidateCommand.ValidationError>();
        var warnings = new List<ValidateCommand.ValidationWarning>();
        // These tests assert step-type validation (task-independent); use a preFeaturizer-supporting
        // task so leakage warnings don't perturb the error/warning counts under test.
        ValidateCommand.ValidatePrepSteps("test.prep", steps, "regression", errors, warnings);
        return (errors, warnings);
    }

    #endregion
}

public class ValidateCommandPolicyTests
{
    [Fact]
    public void PrepPolicyNotes_NormalizeOnSupportedTask_AddsInformationalNote()
    {
        var prep = new List<PrepStep> { new() { Type = "normalize", Method = "z-score", Columns = ["a"] } };
        var notes = ValidateCommand.PrepPolicyNotes(prep, "regression");
        Assert.Single(notes);
        Assert.Contains("normalize", notes[0]);
        Assert.Contains("AutoML", notes[0]); // 자동선택 맥락 명시
    }

    [Fact]
    public void PrepPolicyNotes_CsvStageStep_NoNote()
    {
        var prep = new List<PrepStep> { new() { Type = "remove-columns", Columns = ["a"] } };
        var notes = ValidateCommand.PrepPolicyNotes(prep, "regression");
        Assert.Empty(notes); // csv-stage 변환엔 모델클래스 권고 불필요
    }

    [Fact]
    public void PrepPolicyNotes_NormalizeOnUnsupportedTask_NoNote()
    {
        var prep = new List<PrepStep> { new() { Type = "normalize", Method = "z-score", Columns = ["a"] } };
        var notes = ValidateCommand.PrepPolicyNotes(prep, "clustering");
        Assert.Empty(notes); // leakage already flagged by InspectPrepLeakage; no trainer-context note for CSV-baked tasks
    }
}
