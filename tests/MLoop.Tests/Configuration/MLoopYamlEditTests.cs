using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Configuration;

/// <summary>
/// <c>train</c>'s label sync edits <c>mloop.yaml</c> on its lines: the comments <c>init</c> writes and
/// the user's key order survive (a re-serialized file kept neither), and a layout the edit does not know
/// is refused rather than guessed at.
/// </summary>
public class MLoopYamlEditTests
{
    // The shape `mloop init` writes — with LF endings whatever the checkout gave this file (a Windows
    // clone converts the literal to CRLF, and every expectation below is written in LF).
    private static readonly string Init = """
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
        """.ReplaceLineEndings("\n");

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

    [Fact]
    public void AMissingBlock_IsAddedAtTheEndOfTheModel_AndEveryCommentIsKept()
    {
        var edited = MLoopYamlEdit.SetModelBlock(Init, "default", "prep", "prep:\n- type: fill-missing\n  method: mean\n")!;

        Assert.Contains("      test_split: 0.2\n    prep:\n    - type: fill-missing\n      method: mean\n\n# Add more models as needed:", edited);
        Assert.Equal(Init.Split('#').Length, edited.Split('#').Length);
    }

    [Fact]
    public void ABlockWrittenAsASequenceAtTheKeysIndentation_IsReplacedWhole()
    {
        var withPrep = MLoopYamlEdit.SetModelBlock(Init, "default", "prep", "prep:\n- type: fill-missing\n  method: mean\n- type: scale\n")!;

        var edited = MLoopYamlEdit.SetModelBlock(withPrep, "default", "prep", "prep:\n- type: scale\n")!;

        Assert.Contains("    prep:\n    - type: scale\n\n# Add more models", edited);
        Assert.DoesNotContain("fill-missing", edited);
    }

    [Fact]
    public void ANullBlock_RemovesTheKey_AndLeavesTheRest()
    {
        var withColumns = MLoopYamlEdit.SetModelBlock(Init, "default", "columns", "columns:\n  id:\n    type: ignore\n")!;

        var edited = MLoopYamlEdit.SetModelBlock(withColumns, "default", "columns", null)!;

        Assert.Equal(Init, edited);
    }

    [Fact]
    public void ABlockInTheMiddleOfTheModel_KeepsTheKeysAfterIt()
    {
        var yaml = "models:\n  default:\n    columns:\n      id:\n        type: ignore\n    # the label\n    label: y\n";

        var edited = MLoopYamlEdit.SetModelBlock(yaml, "default", "columns", "columns:\n  ts:\n    type: ignore\n")!;

        Assert.Equal("models:\n  default:\n    columns:\n      ts:\n        type: ignore\n    # the label\n    label: y\n", edited);
    }

    [Fact]
    public void MixedLineEndings_AreReadAsLines_AndWrittenWithTheFilesOwnEnding()
    {
        // A CRLF file with a line appended by a tool that writes LF.
        var mixed = Init.Replace("\n", "\r\n") + "\nextra: 1\n";

        var edited = MLoopYamlEdit.SetModelScalars(mixed, "default", [("label", "y")])!;

        Assert.Contains("    label: y\r\n", edited);
        Assert.Contains("\r\nextra: 1\r\n", edited);
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
