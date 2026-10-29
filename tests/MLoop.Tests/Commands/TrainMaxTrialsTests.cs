using System.Globalization;
using System.Text;
using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.Core.Storage;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>train --max-trials</c> / <c>training.max_trials</c> — a cap on the trainer search's work beside the cap on its
/// clock. The search's seed is fixed, so a trial count makes a run the same on a fast machine and a slow one; a
/// limit on a task that runs no search, or below one trial, is refused instead of silently meaning nothing.
/// </summary>
[Collection("FileSystem")]
public sealed class TrainMaxTrialsTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-maxtrials-" + Guid.NewGuid().ToString("N"));

    public TrainMaxTrialsTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static ModelDefinition Definition(string task, int? maxTrials)
        => new() { Task = task, Label = "y", Training = new TrainingSettings { MaxTrials = maxTrials } };

    [Fact]
    public void ALimitOnATaskThatSearches_HasNoConflict_AndReachesTheTrainingConfig()
    {
        var definition = Definition("binary-classification", 7);

        Assert.Null(ModelTrainingConfig.MaxTrialsConflict(definition));
        Assert.Equal(7, ModelTrainingConfig.From(definition, "default", "train.csv", 60, 0.2).MaxTrials);
    }

    [Fact]
    public void NoLimit_LeavesTheTimeBudgetAlone()
    {
        var definition = Definition("regression", null);

        Assert.Null(ModelTrainingConfig.MaxTrialsConflict(definition));
        Assert.Null(ModelTrainingConfig.From(definition, "default", "train.csv", 60, 0.2).MaxTrials);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void ALimitBelowOneTrial_IsRefused(int maxTrials)
        => Assert.Contains("not a number of trials", ModelTrainingConfig.MaxTrialsConflict(Definition("regression", maxTrials)));

    [Theory]
    [InlineData("clustering")]
    [InlineData("anomaly-detection")]
    [InlineData("recommendation")]
    public void ALimitOnATaskThatRunsNoSearch_IsRefusedWithTheReason(string task)
    {
        var message = ModelTrainingConfig.MaxTrialsConflict(Definition(task, 5));

        Assert.NotNull(message);
        Assert.Contains("no trials to count", message);
    }

    [Fact]
    public void TheCommandLineLimit_TakesThePlaceOfTheFilesLimit()
    {
        var merger = new ConfigMerger();
        var fromFile = new TrainingSettings { MaxTrials = 3 };

        Assert.Equal(5, merger.MergeTrainingSettings(null, fromFile, new TrainingSettings { MaxTrials = 5 }).MaxTrials);
        Assert.Equal(3, merger.MergeTrainingSettings(null, fromFile, new TrainingSettings()).MaxTrials);
        Assert.Null(merger.MergeTrainingSettings(null, new TrainingSettings(), null).MaxTrials);
    }

    [Fact]
    public void TheYamlKey_IsOneThatIsRead()
        => Assert.Empty(ConfigKeyAudit.Find("training:\n  max_trials: 5\n  time_limit_seconds: 60\n", typeof(ModelDefinition)));

    [Fact]
    public void Validate_RefusesALimitBelowOneTrialAndOnATaskThatRunsNoSearch()
    {
        var errors = new List<ValidateCommand.ValidationError>();
        var warnings = new List<ValidateCommand.ValidationWarning>();

        ValidateCommand.ValidateTrainingSettings("models.a.training", new TrainingSettings { MaxTrials = 0 }, "regression", errors, warnings);
        ValidateCommand.ValidateTrainingSettings("models.b.training", new TrainingSettings { MaxTrials = 5 }, "clustering", errors, warnings);
        ValidateCommand.ValidateTrainingSettings("models.c.training", new TrainingSettings { MaxTrials = 5 }, "regression", errors, warnings);

        Assert.Equal(["models.a.training.max_trials", "models.b.training.max_trials"], errors.Select(e => e.Path));
    }

    [Fact]
    public async Task Train_WithAMaxTrials_StopsTheSearchAfterThatManyTrials()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));

        var csv = new StringBuilder("x,z,y\n");
        var rng = new Random(3);
        for (var i = 0; i < 200; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(x + rng.NextDouble() * 0.4 > 0.7 ? "true" : "false")}\n");
        }
        await File.WriteAllTextAsync("datasets/train.csv", csv.ToString());

        var (exit, output, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "y", "--max-trials", "3", "--time", "30", "--no-promote");

        Assert.True(exit == 0, output);
        var trialsFile = Assert.Single(Directory.GetFiles("models", ExperimentLayout.TrialsFileName, SearchOption.AllDirectories));
        var trials = (await File.ReadAllLinesAsync(trialsFile)).Count(l => !string.IsNullOrWhiteSpace(l));
        Assert.InRange(trials, 1, 3);
    }

    [Fact]
    public async Task Train_WithAMaxTrialsAndNoTime_SkipsTheTimeEstimate_ThatWouldRunTheSameSearchTwice()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Table());

        var (exit, output, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "y", "--max-trials", "3", "--no-promote", "--json");

        Assert.True(exit == 0, output);
        Assert.DoesNotContain("probe", output, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Train_WithAutoTimeAndAMaxTrials_IsRefused()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Table());

        var (exit, _, stderr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--auto-time", "--max-trials", "3");

        Assert.Equal(1, exit);
        Assert.Contains("--auto-time cannot be combined with a trial limit", stderr);
    }

    private static string Table()
    {
        var csv = new StringBuilder("x,z,y\n");
        var rng = new Random(3);
        for (var i = 0; i < 200; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(x + rng.NextDouble() * 0.4 > 0.7 ? "true" : "false")}\n");
        }
        return csv.ToString();
    }

    [Fact]
    public async Task Train_WithAMaxTrialsOnATaskThatRunsNoSearch_FailsBeforeTraining()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "clustering");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", "a,b\n1,2\n2,3\n3,4\n");

        var (exit, _, stderr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--max-trials", "3", "--time", "5");

        Assert.Equal(1, exit);
        Assert.Contains("no trials to count", stderr);
    }
}
