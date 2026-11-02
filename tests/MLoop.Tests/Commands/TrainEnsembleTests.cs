using System.Globalization;
using System.Text;
using System.Text.Json;
using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>train --ensemble</c> / <c>ensemble: true</c> — answer with a blend of the search's best trials. The setting reaches the training
/// configuration and is recorded with the experiment; a task the blend is not made for, and a model that is also asked to be portable,
/// are refused instead of silently meaning nothing.
/// </summary>
[Collection("FileSystem")]
public sealed class TrainEnsembleTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-ensemble-" + Guid.NewGuid().ToString("N"));

    public TrainEnsembleTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static ModelDefinition Definition(string task, bool? ensemble, bool? portable = null)
        => new() { Task = task, Label = "y", Ensemble = ensemble, Portable = portable };

    [Theory]
    [InlineData("binary-classification")]
    [InlineData("regression")]
    public void ABlendOnATaskItIsMadeFor_HasNoConflict_AndReachesTheTrainingConfig(string task)
    {
        var definition = Definition(task, true);

        Assert.Null(ModelTrainingConfig.EnsembleConflict(definition));
        Assert.True(ModelTrainingConfig.From(definition, "default", "train.csv", 60, 0.2).Ensemble);
    }

    [Fact]
    public void WithoutTheSetting_TheRunIsNotABlend()
    {
        Assert.Null(ModelTrainingConfig.EnsembleConflict(Definition("clustering", null)));
        Assert.Null(ModelTrainingConfig.EnsembleConflict(Definition("clustering", false)));
        Assert.False(ModelTrainingConfig.From(Definition("binary-classification", null), "default", "train.csv", 60, 0.2).Ensemble);
    }

    [Theory]
    [InlineData("multiclass-classification")]
    [InlineData("clustering")]
    [InlineData("anomaly-detection")]
    [InlineData("forecasting")]
    public void ABlendOnATaskItIsNotMadeFor_IsRefusedWithTheReason(string task)
    {
        var message = ModelTrainingConfig.EnsembleConflict(Definition(task, true));

        Assert.NotNull(message);
        Assert.Contains("binary-classification or regression", message);
    }

    [Fact]
    public void ABlendThatIsAlsoPortable_IsRefused()
        => Assert.Contains("ONNX", ModelTrainingConfig.EnsembleConflict(Definition("regression", true, portable: true)));

    [Fact]
    public void TheYamlKey_IsOneThatIsRead()
        => Assert.Empty(ConfigKeyAudit.Find("ensemble: true\n", typeof(ModelDefinition)));

    [Fact]
    public void Validate_RefusesABlendOnATaskItIsNotMadeFor()
    {
        var errors = new List<ValidateCommand.ValidationError>();
        var warnings = new List<ValidateCommand.ValidationWarning>();

        ValidateCommand.ValidateModel("a", Definition("clustering", true), errors, warnings);
        ValidateCommand.ValidateModel("b", Definition("binary-classification", true), errors, warnings);

        Assert.Contains(errors, e => e.Path == "models.a.ensemble");
        Assert.DoesNotContain(errors, e => e.Path == "models.b.ensemble");
    }

    private static string Table(int rows = 500)
    {
        var csv = new StringBuilder("x1,x2,x3,y\n");
        var rng = new Random(3);
        for (var i = 0; i < rows; i++)
        {
            double X() => rng.NextDouble() * 2 - 1;
            var (x1, x2, x3) = (X(), X(), X());
            var logit = 1.5 * x1 - x2 + 1.2 * x3 * x1 + (rng.NextDouble() - 0.5) * 2;
            csv.Append(CultureInfo.InvariantCulture, $"{x1:F4},{x2:F4},{x3:F4},{(logit > 0 ? "yes" : "no")}\n");
        }
        return csv.ToString();
    }

    [Fact]
    public async Task Train_WithEnsemble_RecordsTheRequestAndWhatItCameTo()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Table());

        var (exit, output, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "y", "--max-trials", "10", "--time", "60", "--no-promote", "--ensemble");

        Assert.True(exit == 0, output);
        var experiment = Path.Combine("models", "default", "staging", "exp-001");
        Assert.Contains("\"ensemble\": true", await File.ReadAllTextAsync(Path.Combine(experiment, "config.json")));
        using var metadata = JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(experiment, "metadata.json")));
        // A blend or the reason there is none: the request always comes to one of the two.
        var outcome = metadata.RootElement.GetProperty("result").GetProperty("ensemble");
        Assert.True(outcome.TryGetProperty("members", out _) || outcome.TryGetProperty("skippedBecause", out _));
    }

    [Theory]
    [InlineData("--portable")]
    public async Task Train_WithEnsembleAndAnOptionItCannotBeWith_IsRefusedBeforeTraining(string other)
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Table(80));

        var (exit, output, error) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "y", "--time", "5", "--no-promote", "--ensemble", other);   // no search: refused before any trial

        Assert.Equal(1, exit);
        Assert.Contains("ensemble", output + error);
        Assert.False(Directory.Exists(Path.Combine("models", "default", "staging", "exp-001")));
    }

    [Fact]
    public async Task Train_WithEnsembleOnATaskItIsNotMadeFor_IsRefusedBeforeTraining()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "c", "--task", "clustering");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "c"));
        await File.WriteAllTextAsync("datasets/train.csv", "a,b\n1,2\n3,4\n5,6\n");

        var (exit, output, error) = await CliRunner.RunAsync("train", "datasets/train.csv", "--ensemble", "--time", "5");   // no search: clustering

        Assert.Equal(1, exit);
        Assert.Contains("ensemble", output + error);
        Assert.False(Directory.Exists(Path.Combine("models", "default", "staging", "exp-001")));
    }
}
