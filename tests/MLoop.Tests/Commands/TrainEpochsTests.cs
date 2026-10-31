using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>train --epochs</c> / <c>training.epochs</c> — the one dial on a deep-learning fit, which runs a fixed number of
/// epochs and takes no time budget. A count on a task that does not train by epochs, or below one epoch, is refused
/// instead of silently meaning nothing.
/// </summary>
[Collection("FileSystem")]
public sealed class TrainEpochsTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-epochs-" + Guid.NewGuid().ToString("N"));

    public TrainEpochsTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static ModelDefinition Definition(string task, int? epochs)
        => new() { Task = task, Label = "y", Training = new TrainingSettings { Epochs = epochs } };

    [Theory]
    [InlineData("text-classification")]
    [InlineData("sentence-similarity")]
    [InlineData("ner")]
    [InlineData("question-answering")]
    [InlineData("object-detection")]
    public void ACountOnATaskThatTrainsByEpochs_HasNoConflict_AndReachesTheTrainingConfig(string task)
    {
        var definition = Definition(task, 4);

        Assert.Null(ModelTrainingConfig.EpochsConflict(definition));
        Assert.Equal(4, ModelTrainingConfig.From(definition, "default", "train.csv", 60, 0.2).Epochs);
    }

    [Fact]
    public void NoCount_LeavesTheDefault()
    {
        var definition = Definition("object-detection", null);

        Assert.Null(ModelTrainingConfig.EpochsConflict(definition));
        Assert.Null(ModelTrainingConfig.From(definition, "default", "train.csv", 60, 0.2).Epochs);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-2)]
    public void ACountBelowOneEpoch_IsRefused(int epochs)
        => Assert.Contains("not a number of epochs", ModelTrainingConfig.EpochsConflict(Definition("object-detection", epochs)));

    [Theory]
    [InlineData("binary-classification", "does not train by epochs")]
    [InlineData("regression", "does not train by epochs")]
    [InlineData("clustering", "does not train by epochs")]
    [InlineData("image-classification", "stops early on its own")]
    public void ACountOnATaskThatDoesNotCountEpochs_IsRefusedWithTheReason(string task, string reason)
    {
        var message = ModelTrainingConfig.EpochsConflict(Definition(task, 5));

        Assert.NotNull(message);
        Assert.Contains(reason, message);
        Assert.Contains("no epoch count to set", message);
    }

    [Fact]
    public void TheCommandLineCount_TakesThePlaceOfTheFilesCount()
    {
        var merger = new ConfigMerger();
        var fromFile = new TrainingSettings { Epochs = 3 };

        Assert.Equal(5, merger.MergeTrainingSettings(null, fromFile, new TrainingSettings { Epochs = 5 }).Epochs);
        Assert.Equal(3, merger.MergeTrainingSettings(null, fromFile, new TrainingSettings()).Epochs);
        Assert.Null(merger.MergeTrainingSettings(null, new TrainingSettings(), null).Epochs);
    }

    [Fact]
    public void TheYamlKey_IsOneThatIsRead()
        => Assert.Empty(ConfigKeyAudit.Find("training:\n  epochs: 5\n", typeof(ModelDefinition)));

    [Fact]
    public void Validate_RefusesACountBelowOneEpochAndOnATaskThatDoesNotCountEpochs()
    {
        var errors = new List<ValidateCommand.ValidationError>();
        var warnings = new List<ValidateCommand.ValidationWarning>();

        ValidateCommand.ValidateTrainingSettings("models.a.training", new TrainingSettings { Epochs = 0 }, "object-detection", errors, warnings);
        ValidateCommand.ValidateTrainingSettings("models.b.training", new TrainingSettings { Epochs = 5 }, "regression", errors, warnings);
        ValidateCommand.ValidateTrainingSettings("models.c.training", new TrainingSettings { Epochs = 5 }, "ner", errors, warnings);

        Assert.Equal(["models.a.training.epochs", "models.b.training.epochs"], errors.Select(e => e.Path));
    }

    [Fact]
    public async Task Train_WithEpochsOnATaskThatDoesNotCountEpochs_FailsBeforeTraining()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", "x,y\n1,true\n2,false\n3,true\n");

        var (exit, _, stderr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--epochs", "3", "--time", "5");   // no search: refused before it trains

        Assert.Equal(1, exit);
        Assert.Contains("no epoch count to set", stderr);
    }
}
