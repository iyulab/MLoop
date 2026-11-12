using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Data;
using MLoop.Core.Training;

namespace MLoop.Tests.Configuration;

/// <summary>
/// A task that trains on a directory of images has its loader make the label column; nothing for the user to name. And the options
/// that shape a table's rows do nothing to such a task, which is a reason to refuse them, not to let them be believed.
/// </summary>
public class DirectoryTaskLabelTests
{
    [Theory]
    [InlineData("object-detection")]
    [InlineData("image-classification")]
    public void ADirectoryTask_NeedsNoLabelNamed_AndGetsTheLoadersOwn(string task)
    {
        var definition = new ConfigMerger().GetEffectiveModelDefinition(
            new MLoopConfig { Models = new Dictionary<string, ModelDefinition> { ["default"] = new() { Task = task, Label = null! } } }, "default");

        Assert.Equal(CocoDataLoader.DefaultLabelColumn, definition.Label);
        Assert.False(TrainingRunner.LabelColumnMustBeNamed(task));
        Assert.Null(TrainingInput.DefinitionRefusal(definition));
    }

    [Theory]
    [InlineData("binary-classification")]
    [InlineData("regression")]
    [InlineData("multiclass-classification")]
    public void ATableTask_StillMustNameItsLabel(string task)
    {
        var config = new MLoopConfig { Models = new Dictionary<string, ModelDefinition> { ["default"] = new() { Task = task, Label = null! } } };

        var ex = Assert.Throws<InvalidOperationException>(() => new ConfigMerger().GetEffectiveModelDefinition(config, "default"));

        Assert.Contains("Label not specified", ex.Message);
        Assert.True(TrainingRunner.LabelColumnMustBeNamed(task));
    }

    // A client that writes «none» as an empty string (`"labelColumn": ""`) named a column «» and the run failed with
    // `Could not find input column ''`. A blank label or task is no choice: mloop.yaml, then the task, decides.
    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void ABlankLabelOrTask_IsNoChoice_SoTheDefinitionAndTheTaskDecide(string blank)
    {
        var yaml = new MLoopConfig { Models = new Dictionary<string, ModelDefinition> { ["default"] = new() { Task = "image-classification", Label = null! } } };

        var definition = new ConfigMerger().GetEffectiveModelDefinition(yaml, "default", cliLabel: blank, cliTask: blank);

        Assert.Equal("image-classification", definition.Task);
        Assert.Equal(CocoDataLoader.DefaultLabelColumn, definition.Label);
    }

    [Fact]
    public void ABlankLabel_ForATableTask_IsStillRefusedAsMissing()
    {
        var yaml = new MLoopConfig { Models = new Dictionary<string, ModelDefinition> { ["default"] = new() { Task = "regression", Label = null! } } };

        var ex = Assert.Throws<InvalidOperationException>(() => new ConfigMerger().GetEffectiveModelDefinition(yaml, "default", cliLabel: ""));

        Assert.Contains("Label not specified", ex.Message);
    }

    [Fact]
    public void ALabelTheUserNamedForADirectoryTask_IsKept()
    {
        var definition = new ConfigMerger().GetEffectiveModelDefinition(
            new MLoopConfig { Models = new Dictionary<string, ModelDefinition> { ["default"] = new() { Task = "object-detection", Label = "Boxes" } } }, "default");

        Assert.Equal("Boxes", definition.Label);
    }

    [Fact]
    public void TableOptions_AreRefusedForADirectoryTask_AndNamed()
    {
        var message = TrainingInput.TableOnlyOptionsRefusal("object-detection", ["--max-rows", "--seed"]);

        Assert.NotNull(message);
        Assert.Contains("--max-rows", message);
        Assert.Contains("--seed", message);
        Assert.Contains("object-detection", message);
    }

    [Theory]
    [InlineData("binary-classification")]
    [InlineData("regression")]
    public void TableOptions_AreFineForATableTask(string task)
        => Assert.Null(TrainingInput.TableOnlyOptionsRefusal(task, ["--max-rows", "--seed"]));

    [Fact]
    public void NoTableOptions_IsNoRefusalEvenForADirectoryTask()
        => Assert.Null(TrainingInput.TableOnlyOptionsRefusal("image-classification", []));
}
