using MLoop.CLI.Commands;
using MLoop.Extensibility.Hooks;

namespace MLoop.Tests.Commands;

public class NewHookCommandTests
{
    #region ToPascalCase

    [Theory]
    [InlineData("data-validation", "DataValidation")]
    [InlineData("mlflow-logging", "MlflowLogging")]
    [InlineData("my_hook", "MyHook")]
    [InlineData("simple", "Simple")]
    [InlineData("pre-train-check", "PreTrainCheck")]
    [InlineData("ALL-CAPS", "AllCaps")]
    public void ToPascalCase_ConvertsCorrectly(string input, string expected)
    {
        Assert.Equal(expected, NewHookCommand.ToPascalCase(input));
    }

    #endregion

    #region ToTitleCase

    [Theory]
    [InlineData("data-validation", "Data Validation")]
    [InlineData("mlflow_logging", "Mlflow Logging")]
    [InlineData("simple", "Simple")]
    [InlineData("pre-train-check", "Pre Train Check")]
    public void ToTitleCase_ConvertsCorrectly(string input, string expected)
    {
        Assert.Equal(expected, NewHookCommand.ToTitleCase(input));
    }

    #endregion

    #region ValidateAndParseHookType

    [Theory]
    [InlineData("pre-train", HookType.PreTrain)]
    [InlineData("post-train", HookType.PostTrain)]
    [InlineData("pre-predict", HookType.PrePredict)]
    [InlineData("post-evaluate", HookType.PostEvaluate)]
    public void ValidateAndParseHookType_ValidTypes_ReturnsEnum(string input, HookType expected)
    {
        Assert.Equal(expected, NewHookCommand.ValidateAndParseHookType(input));
    }

    [Theory]
    [InlineData("PRE-TRAIN", HookType.PreTrain)]
    [InlineData("Post-Train", HookType.PostTrain)]
    public void ValidateAndParseHookType_CaseInsensitive(string input, HookType expected)
    {
        Assert.Equal(expected, NewHookCommand.ValidateAndParseHookType(input));
    }

    [Theory]
    [InlineData("invalid")]
    [InlineData("pre-deploy")]
    [InlineData("")]
    public void ValidateAndParseHookType_InvalidTypes_ReturnsNull(string input)
    {
        Assert.Null(NewHookCommand.ValidateAndParseHookType(input));
    }

    #endregion

    #region SelectTemplate

    [Fact]
    public void SelectTemplate_ExplicitTemplate_ReturnsAsIs()
    {
        Assert.Equal("performance-gate", NewHookCommand.SelectTemplate("performance-gate", HookType.PreTrain));
    }

    [Fact]
    public void SelectTemplate_NullTemplate_DefaultsForPreTrain()
    {
        Assert.Equal("validation", NewHookCommand.SelectTemplate(null, HookType.PreTrain));
    }

    [Fact]
    public void SelectTemplate_NullTemplate_DefaultsForPostTrain()
    {
        Assert.Equal("logging", NewHookCommand.SelectTemplate(null, HookType.PostTrain));
    }

    [Fact]
    public void SelectTemplate_NullTemplate_DefaultsForPrePredict()
    {
        Assert.Equal("validation", NewHookCommand.SelectTemplate(null, HookType.PrePredict));
    }

    [Fact]
    public void SelectTemplate_NullTemplate_DefaultsForPostEvaluate()
    {
        Assert.Equal("logging", NewHookCommand.SelectTemplate(null, HookType.PostEvaluate));
    }

    [Fact]
    public void SelectTemplate_EmptyTemplate_DefaultsForHookType()
    {
        Assert.Equal("validation", NewHookCommand.SelectTemplate("", HookType.PreTrain));
    }

    #endregion

    #region GenerateHookContent

    [Theory]
    [InlineData("basic")]
    [InlineData("validation")]
    [InlineData("logging")]
    [InlineData("performance-gate")]
    [InlineData("deploy")]
    public void GenerateHookContent_AllTemplates_ContainClassName(string template)
    {
        var result = NewHookCommand.GenerateHookContent("my-hook", HookType.PreTrain, template);

        Assert.Contains("class MyHookHook", result);
        Assert.Contains("IMLoopHook", result);
    }

    [Theory]
    [InlineData("basic")]
    [InlineData("validation")]
    [InlineData("logging")]
    [InlineData("performance-gate")]
    [InlineData("deploy")]
    public void GenerateHookContent_AllTemplates_ContainHookName(string template)
    {
        var result = NewHookCommand.GenerateHookContent("data-check", HookType.PreTrain, template);

        Assert.Contains("Data Check", result);
    }

    [Fact]
    public void GenerateHookContent_BasicTemplate_ContainsHookResult()
    {
        var result = NewHookCommand.GenerateHookContent("test", HookType.PreTrain, "basic");

        Assert.Contains("HookResult.Continue()", result);
        Assert.Contains("HookResult.Abort(", result);
    }

    [Fact]
    public void GenerateHookContent_ValidationTemplate_ContainsDataViewCheck()
    {
        var result = NewHookCommand.GenerateHookContent("test", HookType.PreTrain, "validation");

        Assert.Contains("DataView", result);
        Assert.Contains("MIN_ROWS", result);
    }

    [Fact]
    public void GenerateHookContent_LoggingTemplate_ContainsMetrics()
    {
        var result = NewHookCommand.GenerateHookContent("test", HookType.PostTrain, "logging");

        Assert.Contains("Metrics", result);
        Assert.Contains("metricName", result);
    }

    [Fact]
    public void GenerateHookContent_PerformanceGateTemplate_ContainsThreshold()
    {
        var result = NewHookCommand.GenerateHookContent("test", HookType.PostEvaluate, "performance-gate");

        Assert.Contains("MIN_ACCURACY", result);
        Assert.Contains("BinaryClassificationMetrics", result);
    }

    [Fact]
    public void GenerateHookContent_DeployTemplate_ContainsDeployLogic()
    {
        var result = NewHookCommand.GenerateHookContent("test", HookType.PostTrain, "deploy");

        Assert.Contains("DEPLOY_THRESHOLD", result);
        Assert.Contains("DeploymentTriggered", result);
    }

    [Fact]
    public void GenerateHookContent_UnknownTemplate_FallsBackToBasic()
    {
        var result = NewHookCommand.GenerateHookContent("test", HookType.PreTrain, "unknown-template");

        Assert.Contains("HookResult.Continue()", result);
        Assert.Contains("TODO: Implement your hook logic here", result);
    }

    #endregion
}
