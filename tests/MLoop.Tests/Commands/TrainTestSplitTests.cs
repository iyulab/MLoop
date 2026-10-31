using System.Globalization;
using System.Text;
using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>train --test-split</c> / <c>training.test_split</c> is the share of the rows held out to measure the model on. A share outside
/// (0, 1) holds nothing out - or everything - and with none held out the scores a run reports are those of rows the model was
/// trained on: refused with the reason, as <c>mloop validate</c> already says of the file's value.
/// </summary>
[Collection("FileSystem")]
public sealed class TrainTestSplitTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-testsplit-" + Guid.NewGuid().ToString("N"));

    public TrainTestSplitTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static ModelDefinition Definition(string task, double? testSplit)
        => new() { Task = task, Label = "y", Training = new TrainingSettings { TestSplit = testSplit } };

    [Theory]
    [InlineData(0.05)]
    [InlineData(0.2)]
    [InlineData(0.5)]
    [InlineData(0.95)]
    public void AShareBetweenZeroAndOne_HasNoConflict(double share)
        => Assert.Null(ModelTrainingConfig.TestSplitConflict(Definition("binary-classification", share)));

    [Fact]
    public void NoShareGiven_LeavesTheDefault()
        => Assert.Null(ModelTrainingConfig.TestSplitConflict(Definition("regression", null)));

    [Theory]
    [InlineData(0)]
    [InlineData(-0.2)]
    [InlineData(1)]
    [InlineData(1.5)]
    public void AShareOutsideZeroAndOne_IsRefusedWithTheReason(double share)
    {
        var message = ModelTrainingConfig.TestSplitConflict(Definition("multiclass-classification", share));

        Assert.NotNull(message);
        Assert.Contains("between 0 and 1", message, StringComparison.Ordinal);
        Assert.Contains("held out", message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("forecasting")]
    [InlineData("clustering")]
    public void ATaskThatHoldsNothingOut_IsLeftAlone(string task)
        => Assert.Null(ModelTrainingConfig.TestSplitConflict(Definition(task, 0)));

    [Theory]
    [InlineData("0")]
    [InlineData("-0.2")]
    [InlineData("1")]
    public async Task TheCommandLineShare_IsRefusedBeforeAnythingIsTrained(string share)
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var rng = new Random(1);
        var text = new StringBuilder("x,y\n");
        for (var i = 0; i < 80; i++)
        {
            var x = rng.NextDouble();
            text.Append(string.Create(CultureInfo.InvariantCulture, $"{x:F4},{(x > 0.5 ? "yes" : "no")}\n"));
        }
        await File.WriteAllTextAsync("datasets/train.csv", text.ToString());

        var run = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--max-trials", "2", $"--test-split={share}");

        Assert.NotEqual(0, run.ExitCode);
        Assert.Contains("between 0 and 1", run.Stdout + run.Stderr, StringComparison.Ordinal);
        Assert.False(Directory.Exists(Path.Combine("models", "default", "staging", "exp-001")));
    }
}
