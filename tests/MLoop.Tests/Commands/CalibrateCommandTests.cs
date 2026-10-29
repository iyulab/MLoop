using System.Globalization;
using System.Text;
using System.Text.Json;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop calibrate</c> on a real binary model: once fit, evaluate reports the ECE it recorded on the
/// same rows, predict answers with the calibrated probability, promote carries it, and it is refused
/// while a decision threshold stands.
/// </summary>
[Collection("FileSystem")]
public sealed class CalibrateCommandTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-calibrate-" + Guid.NewGuid().ToString("N"));

    public CalibrateCommandTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static string Noisy(int rows, int seed)
    {
        var csv = new StringBuilder("x,z,y\n");
        var rng = new Random(seed);
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble();
            var y = rng.NextDouble() < x;
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{(y ? "yes" : "no")}\n");
        }
        return csv.ToString();
    }

    private async Task TrainAsync()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Noisy(300, 3));
        await File.WriteAllTextAsync("datasets/recheck.csv", Noisy(300, 9));
        var (exit, output, _) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(exit == 0, output);
    }

    private static async Task<JsonElement> JsonAsync(params string[] args)
    {
        var (exit, stdout, stderr) = await CliRunner.RunAsync([.. args, "--json"]);
        Assert.True(exit == 0, stdout + stderr);
        return JsonDocument.Parse(stdout).RootElement.Clone();
    }

    [Fact]
    public async Task ATrainedBinaryModel_ArrivesCalibratedOnItsOutOfFoldAnswers_AndClearGoesBackToTheTrainers()
    {
        await TrainAsync();

        var shown = (await JsonAsync("calibrate")).GetProperty("calibration");
        Assert.Equal("platt", shown.GetProperty("method").GetString());
        Assert.Equal(MLoop.Core.Training.TrainingRunner.OutOfFoldSource, shown.GetProperty("fitOn").GetString());
        Assert.True(File.Exists("models/default/staging/exp-001/calibration.json"));
        Assert.True(File.Exists("models/default/production/calibration.json"));
        var calibrated = (await JsonAsync("evaluate", "exp-001", "datasets/recheck.csv")).GetProperty("testMetrics");

        Assert.Equal("cleared", (await JsonAsync("calibrate", "exp-001", "--clear")).GetProperty("status").GetString());
        Assert.False(File.Exists("models/default/staging/exp-001/calibration.json"));
        var own = (await JsonAsync("evaluate", "exp-001", "datasets/recheck.csv")).GetProperty("testMetrics");
        // Same model, same rows, same ranking: only how sure it says it is moved.
        Assert.Equal(own.GetProperty("auc").GetDouble(), calibrated.GetProperty("auc").GetDouble(), 9);
        Assert.NotEqual(own.GetProperty("ece").GetDouble(), calibrated.GetProperty("ece").GetDouble());
    }

    [Fact]
    public async Task AFitCalibration_IsWhatEvaluateAndPredictAnswerWith_UntilCleared()
    {
        await TrainAsync();
        // From the trainer's own probabilities: the calibration training fitted is cleared first.
        await JsonAsync("calibrate", "--clear");
        var before = (await JsonAsync("evaluate", "exp-001", "datasets/recheck.csv")).GetProperty("testMetrics");

        var fit = await JsonAsync("calibrate", "--data", "datasets/recheck.csv", "--method", "isotonic");

        Assert.Equal("fit", fit.GetProperty("status").GetString());
        Assert.True(fit.GetProperty("inProduction").GetBoolean());
        var calibration = fit.GetProperty("calibration");
        Assert.True(File.Exists("models/default/production/calibration.json"));
        Assert.True(calibration.GetProperty("eceAfter").GetDouble() <= calibration.GetProperty("eceBefore").GetDouble());

        // evaluate reads the calibrated probabilities: on the rows it was fit on, the ECE it recorded.
        var after = (await JsonAsync("evaluate", "exp-001", "datasets/recheck.csv")).GetProperty("testMetrics");
        Assert.Equal(calibration.GetProperty("eceAfter").GetDouble(), after.GetProperty("ece").GetDouble(), 6);
        Assert.Equal(before.GetProperty("ece").GetDouble(), calibration.GetProperty("eceBefore").GetDouble(), 6);

        // predict answers with them, and its label agrees with its probability.
        foreach (var row in (await JsonAsync("predict", "datasets/recheck.csv")).GetProperty("predictions").EnumerateArray())
        {
            var yes = row.GetProperty("probabilities").GetProperty("yes").GetDouble();
            Assert.Equal(yes >= 0.5 ? "yes" : "no", row.GetProperty("predictedLabel").GetString());
        }

        Assert.Equal("cleared", (await JsonAsync("calibrate", "--clear")).GetProperty("status").GetString());
        Assert.False(File.Exists("models/default/production/calibration.json"));
        Assert.Equal(before.GetProperty("ece").GetDouble(),
            (await JsonAsync("evaluate", "exp-001", "datasets/recheck.csv")).GetProperty("testMetrics").GetProperty("ece").GetDouble(), 9);
    }

    [Fact]
    public async Task ACalibration_IsRefusedWhileAThresholdStands_AndAThresholdAfterItReadsTheCalibratedProbability()
    {
        await TrainAsync();
        await JsonAsync("threshold", "--precision", "0.7", "--data", "datasets/recheck.csv");

        var (exit, stdout, _) = await CliRunner.RunAsync("calibrate", "--data", "datasets/recheck.csv", "--json");
        Assert.Equal(1, exit);
        Assert.Contains("threshold", JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString());

        await JsonAsync("threshold", "--clear");
        await JsonAsync("calibrate", "--data", "datasets/recheck.csv");
        var chosen = (await JsonAsync("threshold", "--precision", "0.7", "--data", "datasets/recheck.csv")).GetProperty("threshold");

        var evaluated = (await JsonAsync("evaluate", "exp-001", "datasets/recheck.csv")).GetProperty("testMetrics");
        Assert.Equal(chosen.GetProperty("precision").GetDouble(), evaluated.GetProperty("precision").GetDouble(), 9);
    }

    [Fact]
    public async Task APromotedModelWithoutACalibration_DropsTheOneProductionHad()
    {
        await TrainAsync();
        await JsonAsync("calibrate", "--data", "datasets/recheck.csv");
        Assert.Equal(0, (await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10", "--no-promote")).ExitCode);
        await JsonAsync("calibrate", "exp-002", "--clear");   // training calibrated it; this one is to have none

        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-002", "--json", "--force")).ExitCode);

        Assert.False(File.Exists("models/default/production/calibration.json"));
        Assert.True(File.Exists("models/default/staging/exp-001/calibration.json"));
    }

    private static string NoisyMulticlass(int rows, int seed)
    {
        var csv = new StringBuilder("x,z,y\n");
        var rng = new Random(seed);
        var names = new[] { "low", "mid", "high" };
        for (var i = 0; i < rows; i++)
        {
            var c = rng.Next(3);
            csv.Append(CultureInfo.InvariantCulture, $"{c + rng.NextDouble() * 1.6:F4},{rng.NextDouble():F4},{names[c]}\n");
        }
        return csv.ToString();
    }

    /// <summary>
    /// A multiclass model is calibrated by temperature: evaluate reports the ECE it recorded, predict
    /// keeps its answers and its class names, and platt is refused for it.
    /// </summary>
    [Fact]
    public async Task AMulticlassModel_IsCalibratedByTemperature_AndKeepsItsAnswersAndClassNames()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "m", "--task", "multiclass-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "m"));
        await File.WriteAllTextAsync("datasets/train.csv", NoisyMulticlass(300, 3));
        await File.WriteAllTextAsync("datasets/recheck.csv", NoisyMulticlass(300, 9));
        var (exit, output, _) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(exit == 0, output);
        var before = (await JsonAsync("predict", "datasets/recheck.csv")).GetProperty("predictions").EnumerateArray()
            .Select(r => r.GetProperty("predictedLabel").GetString()).ToList();

        var calibration = (await JsonAsync("calibrate", "--data", "datasets/recheck.csv")).GetProperty("calibration");

        Assert.Equal("temperature", calibration.GetProperty("method").GetString());
        var evaluated = (await JsonAsync("evaluate", "exp-001", "datasets/recheck.csv")).GetProperty("testMetrics");
        Assert.Equal(calibration.GetProperty("eceAfter").GetDouble(), evaluated.GetProperty("ece").GetDouble(), 6);

        var after = (await JsonAsync("predict", "datasets/recheck.csv")).GetProperty("predictions").EnumerateArray().ToList();
        Assert.Equal(before, after.Select(r => r.GetProperty("predictedLabel").GetString()));
        Assert.Equal(["high", "low", "mid"], after[0].GetProperty("probabilities").EnumerateObject().Select(p => p.Name).Order());

        var (refused, stdout, _) = await CliRunner.RunAsync("calibrate", "--data", "datasets/recheck.csv", "--method", "platt", "--json");
        Assert.Equal(1, refused);
        Assert.Contains("temperature", JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString());
    }
}
