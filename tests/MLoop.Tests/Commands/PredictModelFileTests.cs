using System.Globalization;
using System.Text;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop predict --model &lt;file&gt;</c>. A model file carries no record of the columns it was trained on or of its label's values;
/// that is in its experiment record. The file in the <c>production</c> slot, and a model kept with a copy of its experiment folder
/// somewhere else, were scored as the data reads: the labels came back as <c>True</c>/<c>False</c> where the model was trained on
/// <c>yes</c>/<c>no</c>, and the score column had another name. The answer must be the one the project's own prediction gives.
/// </summary>
[Collection("FileSystem")]
public sealed class PredictModelFileTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-predict-model-" + Guid.NewGuid().ToString("N"));

    public PredictModelFileTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static string Rows(int rows, bool labelled)
    {
        var csv = new StringBuilder(labelled ? "x,kind,y\n" : "x,kind\n");
        var rng = new Random(7);
        string[] kinds = ["a", "b", "c"];
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble();
            var kind = kinds[rng.Next(kinds.Length)];
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{kind}");
            if (labelled) csv.Append(rng.NextDouble() < (x > 0.5 ? 0.9 : 0.1) ? ",yes" : ",no");
            csv.Append('\n');
        }
        return csv.ToString();
    }

    private async Task TrainAsync()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Rows(200, labelled: true));
        await File.WriteAllTextAsync("datasets/predict.csv", Rows(40, labelled: false));
        var (trained, trainOut, trainErr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(trained == 0, trainOut + trainErr);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001")).ExitCode);
    }

    private static string Said((int ExitCode, string Stdout, string Stderr) run) =>
        System.Text.RegularExpressions.Regex.Replace(run.Stdout + run.Stderr, @"\s+", " ");

    [Fact]
    public async Task TheFileInTheProductionSlot_AnswersAsTheProjectDoes()
    {
        await TrainAsync();
        Assert.Equal(0, (await CliRunner.RunAsync("predict", "datasets/predict.csv", "--output", "plain.csv")).ExitCode);

        var run = await CliRunner.RunAsync("predict", "datasets/predict.csv", "--model", Path.Combine("models", "default", "production", "model.zip"), "--output", "slot.csv");

        Assert.True(run.ExitCode == 0, run.Stdout + run.Stderr);
        Assert.DoesNotContain("is not in an experiment folder", Said(run), StringComparison.Ordinal);
        var plain = File.ReadAllLines("plain.csv");
        Assert.Contains("PredictedLabel", plain[0], StringComparison.Ordinal);
        Assert.Equal(plain, File.ReadAllLines("slot.csv"));
        Assert.DoesNotContain(",True,", string.Join('\n', File.ReadAllLines("slot.csv").Skip(1)), StringComparison.Ordinal);
    }

    [Fact]
    public async Task AModelCopiedWithItsExperimentFolder_AnswersAsTheProjectDoes_AndTakesNoExperimentId()
    {
        await TrainAsync();
        Assert.Equal(0, (await CliRunner.RunAsync("predict", "datasets/predict.csv", "--output", "plain.csv")).ExitCode);
        var copy = Path.Combine(_workspace, "kept", "my-model");
        Directory.CreateDirectory(copy);
        foreach (var file in Directory.GetFiles(Path.Combine("models", "default", "staging", "exp-001")))
            File.Copy(file, Path.Combine(copy, Path.GetFileName(file)));

        var run = await CliRunner.RunAsync("predict", "datasets/predict.csv", "--model", Path.Combine(copy, "model.zip"), "--output", "copy.csv", "--log");

        Assert.True(run.ExitCode == 0, run.Stdout + run.Stderr);
        Assert.Contains("Training schema:", Said(run), StringComparison.Ordinal);
        Assert.Equal(File.ReadAllLines("plain.csv"), File.ReadAllLines("copy.csv"));
        // its predictions are logged with no experiment id: the copy is not an experiment of this project
        Assert.DoesNotContain("exp-001", string.Concat(Directory.EnumerateFiles(Path.Combine(".mloop", "logs"), "*.jsonl", SearchOption.AllDirectories).Select(File.ReadAllText)), StringComparison.Ordinal);
    }

    [Fact]
    public async Task AModelFileWithNoRecord_StillSaysTheTrainingSchemaIsUnknown()
    {
        await TrainAsync();
        var alone = Path.Combine(_workspace, "alone.zip");
        File.Copy(Path.Combine("models", "default", "staging", "exp-001", "model.zip"), alone);

        var run = await CliRunner.RunAsync("predict", "datasets/predict.csv", "--model", alone, "--output", "alone.csv");

        Assert.Contains("no experiment record is beside it", Said(run), StringComparison.Ordinal);
    }
}
