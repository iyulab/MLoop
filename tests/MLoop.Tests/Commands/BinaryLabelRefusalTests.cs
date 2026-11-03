using System.Globalization;
using System.Text;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Commands;

/// <summary>
/// A binary-classification task given a label of more than two values is refused before the search, naming the label, how many
/// values it holds and what to do. Before, every trainer failed its first trials with "Schema mismatch for label column:
/// expected Boolean" and the run said so only after the first round - on a table of millions of rows, minutes later.
/// </summary>
public class BinaryLabelRefusalTests
{
    [Fact]
    public void ThreeValues_AreRefused_WithTheLabelTheCountAndSomeOfTheValues()
    {
        var refusal = TrainingInput.BinaryLabelRefusal("binary-classification", "grade", ["a", "b", "c"]);

        Assert.NotNull(refusal);
        Assert.Contains("'grade'", refusal, StringComparison.Ordinal);
        Assert.Contains("3 different values", refusal, StringComparison.Ordinal);
        Assert.Contains("'a', 'b', 'c'", refusal, StringComparison.Ordinal);
        Assert.DoesNotContain("…", refusal, StringComparison.Ordinal);
        Assert.Contains("…", TrainingInput.BinaryLabelRefusal("binary-classification", "g", ["a", "b", "c", "d"])!, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("yes", "no")]
    [InlineData("Yes", "yes", "No")]                      // the label is read ignoring case
    [InlineData("예", "아니오")]
    [InlineData("한", "한", "한")]                          // one value, written in two compositions below
    public void TwoClasses_AreNotRefused(params string[] values) =>
        Assert.Null(TrainingInput.BinaryLabelRefusal("binary-classification", "y", values));

    [Fact]
    public void TheSameKoreanWord_InTwoCompositions_IsOneValue()
    {
        var composed = "한글";
        var decomposed = composed.Normalize(NormalizationForm.FormD);
        Assert.NotEqual(composed, decomposed);

        Assert.Null(TrainingInput.BinaryLabelRefusal("binary-classification", "y", [composed, decomposed, "아니오"]));
    }

    [Theory]
    [InlineData("multiclass-classification")]
    [InlineData("regression")]
    [InlineData("text-classification")]
    [InlineData(null)]
    public void ATaskThatIsNotBinary_IsNeverRefused(string? task) =>
        Assert.Null(TrainingInput.BinaryLabelRefusal(task, "y", ["a", "b", "c", "d"]));

    [Fact]
    public void TheTaskIsRead_AsThePipelineNamesIt()
    {
        // any spelling TaskTypes.Canonical accepts is the same task
        Assert.NotNull(TrainingInput.BinaryLabelRefusal("Binary_Classification", "y", ["a", "b", "c"]));
    }

    [Fact]
    public async Task TheTrainCommand_RefusesBeforeAnyTrial()
    {
        var workspace = Path.Combine(Path.GetTempPath(), "mloop-binary-label-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(workspace);
        var original = Directory.GetCurrentDirectory();
        try
        {
            Directory.SetCurrentDirectory(workspace);
            var init = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
            Assert.True(init.ExitCode == 0, init.Stdout + init.Stderr);
            Directory.SetCurrentDirectory(Path.Combine(workspace, "p"));

            var csv = new StringBuilder("a,b,y\n");
            var rng = new Random(4);
            for (var i = 0; i < 120; i++)
                csv.Append(CultureInfo.InvariantCulture, $"{rng.NextDouble():F3},{rng.NextDouble():F3},{"abc"[i % 3]}\n");
            await File.WriteAllTextAsync("datasets/train.csv", csv.ToString());

            var (exit, stdout, stderr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "10", "--max-trials", "3");

            Assert.Equal(1, exit);
            var all = string.Join(' ', (stdout + " " + stderr).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
            Assert.Contains("3 different values", all, StringComparison.Ordinal);
            Assert.Contains("multiclass-classification", all, StringComparison.Ordinal);
            Assert.DoesNotContain("Schema mismatch", all, StringComparison.Ordinal);
            Assert.False(Directory.Exists(Path.Combine("models", "default", "staging", "exp-001")), "no experiment was started");
        }
        finally
        {
            Directory.SetCurrentDirectory(original);
            try { Directory.Delete(workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
        }
    }
}
