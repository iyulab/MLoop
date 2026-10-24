using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Configuration;

/// <summary>
/// <c>train</c>'s label sync edits <c>mloop.yaml</c> on its lines: the comments <c>init</c> writes and
/// the user's key order survive (a re-serialized file kept neither), and a layout the edit does not know
/// is refused rather than guessed at.
/// </summary>
public class MLoopYamlEditTests
{
    // The shape `mloop init` writes.
    private const string Init = """
        # MLoop Project Configuration
        # Multi-model MLOps with minimal cost

        project: p

        # Model definitions
        models:
          default:
            task: binary-classification
            label: Label
            description: Default model for binary-classification
            training:
              # time_limit_seconds: 300   # optional — omit to let mloop auto-estimate
              metric: accuracy
              test_split: 0.2

        # Add more models as needed:
        # models:
        #   churn:
        #     label: Churned

        # Default data paths
        data:
          train: datasets/train.csv
        """;

    [Fact]
    public void ALabelChange_EditsThatLineOnly()
    {
        var edited = MLoopYamlEdit.SetModelScalars(Init, "default", [("label", "y")]);

        Assert.Equal(Init.Replace("    label: Label", "    label: y"), edited);
    }

    [Fact]
    public void AMissingKey_IsAddedToTheModelsBlock_AndATrailingCommentIsKept()
    {
        var withComment = Init.Replace("    task: binary-classification", "    task: binary-classification   # set by init");

        var edited = MLoopYamlEdit.SetModelScalars(withComment, "default", [("task", "regression"), ("group_column", "store")])!;

        Assert.Contains("    task: regression   # set by init", edited);
        Assert.Contains("  default:\n    group_column: store\n", edited);
        Assert.Contains("# time_limit_seconds: 300", edited);
    }

    [Fact]
    public void AValueThatIsNotAPlainScalar_IsQuoted()
    {
        var edited = MLoopYamlEdit.SetModelScalars(Init, "default", [("label", "Churn: yes")])!;

        Assert.Contains("    label: \"Churn: yes\"", edited);
    }

    [Fact]
    public void ANewModel_IsAppendedToTheModelsMapping_BeforeTheNextSection()
    {
        var edited = MLoopYamlEdit.AddModel(Init, "churn:\n  task: binary-classification\n  label: Churned\n")!;

        Assert.Contains("      test_split: 0.2\n  churn:\n    task: binary-classification\n    label: Churned\n", edited);
        Assert.Contains("# Default data paths", edited);
    }

    [Fact]
    public void WindowsLineEndings_AreKept()
    {
        var edited = MLoopYamlEdit.SetModelScalars(Init.Replace("\n", "\r\n"), "default", [("label", "y")])!;

        Assert.Contains("    label: y\r\n", edited);
        Assert.DoesNotContain("\r\r", edited);
    }

    [Theory]
    [InlineData("project: p\nmodels: {default: {label: Label}}\n")]                  // flow style
    [InlineData("project: p\nmodels:\n  default:\n    label:\n      - a\n")]           // not a scalar
    [InlineData("project_name: p\ntask: regression\nlabel_column: y\n")]               // the old single-model format
    [InlineData("project: p\nmodels:\n  other:\n    label: x\n")]                      // no such model
    public void ALayoutTheEditDoesNotKnow_IsRefused(string yaml) =>
        Assert.Null(MLoopYamlEdit.SetModelScalars(yaml, "default", [("label", "y")]));
}
