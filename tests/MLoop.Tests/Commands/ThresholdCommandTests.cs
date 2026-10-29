using System.Globalization;
using System.Text;
using System.Text.Json;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop threshold</c> on a real binary model: once chosen, predict and evaluate answer at it, the
/// production copy follows, and <c>--clear</c> returns the trainer's own.
/// </summary>
[Collection("FileSystem")]
public sealed class ThresholdCommandTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-threshold-" + Guid.NewGuid().ToString("N"));

    public ThresholdCommandTests()
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
            var y = rng.NextDouble() < x;   // the larger x, the likelier yes — a ranking, not a rule
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
        var (exit, output, error) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        // stderr carries the reason a run fails; without it a failure here showed only the progress lines.
        Assert.True(exit == 0, output + Environment.NewLine + error);
    }

    private static async Task<JsonElement> JsonAsync(params string[] args)
    {
        var (exit, stdout, stderr) = await CliRunner.RunAsync([.. args, "--json"]);
        Assert.True(exit == 0, stdout + stderr);
        return JsonDocument.Parse(stdout).RootElement.Clone();
    }

    [Fact]
    public async Task AChosenThreshold_IsWhatPredictAndEvaluateAnswerAt_UntilCleared()
    {
        await TrainAsync();
        var before = (await JsonAsync("evaluate", "exp-001", "datasets/recheck.csv")).GetProperty("testMetrics");

        var chosen = await JsonAsync("threshold", "--precision", "0.7", "--data", "datasets/recheck.csv");

        Assert.Equal("chosen", chosen.GetProperty("status").GetString());
        Assert.True(chosen.GetProperty("inProduction").GetBoolean());
        var threshold = chosen.GetProperty("threshold");
        Assert.True(threshold.GetProperty("precision").GetDouble() >= 0.7);
        Assert.True(File.Exists("models/default/production/decision-threshold.json"));

        // evaluate answers at it: the precision it was chosen for, on the rows it was chosen on.
        var evaluated = await JsonAsync("evaluate", "exp-001", "datasets/recheck.csv");
        var after = evaluated.GetProperty("testMetrics");
        Assert.Equal(threshold.GetProperty("precision").GetDouble(), after.GetProperty("precision").GetDouble(), 9);
        Assert.Equal(threshold.GetProperty("recall").GetDouble(), after.GetProperty("recall").GetDouble(), 9);
        Assert.Equal(before.GetProperty("auc").GetDouble(), after.GetProperty("auc").GetDouble(), 9);   // ranking unchanged

        // predict answers at it: as many positives as the confusion matrix's positive column.
        var counts = evaluated.GetProperty("confusion").GetProperty("counts");
        var answeredYes = counts[0][1].GetDouble() + counts[1][1].GetDouble();
        var predicted = await JsonAsync("predict", "datasets/recheck.csv");
        Assert.Equal(answeredYes, predicted.GetProperty("predictions").EnumerateArray().Count(p => p.GetProperty("predictedLabel").GetString() == "yes"));

        var cleared = await JsonAsync("threshold", "--clear");
        Assert.Equal("cleared", cleared.GetProperty("status").GetString());
        Assert.False(File.Exists("models/default/production/decision-threshold.json"));
        Assert.Equal(before.GetProperty("precision").GetDouble(),
            (await JsonAsync("evaluate", "exp-001", "datasets/recheck.csv")).GetProperty("testMetrics").GetProperty("precision").GetDouble(), 9);
    }

    /// <summary>
    /// The precision asked for is the negative class's — the side whose wrong answers are costly
    /// when «no» means «safe, skip the check». evaluate then reports that precision at the threshold.
    /// </summary>
    [Fact]
    public async Task AThresholdForTheNegativeClass_ReachesItsPrecision_AndEvaluateReportsIt()
    {
        await TrainAsync();

        var chosen = await JsonAsync("threshold", "--precision", "0.8", "--precision-of", "NO", "--data", "datasets/recheck.csv");

        var threshold = chosen.GetProperty("threshold");
        Assert.Equal("no", threshold.GetProperty("precisionOf").GetString());   // recorded as the label wrote it
        Assert.True(threshold.GetProperty("precision").GetDouble() >= 0.8);

        var after = (await JsonAsync("evaluate", "exp-001", "datasets/recheck.csv")).GetProperty("testMetrics");
        Assert.Equal(threshold.GetProperty("precision").GetDouble(), after.GetProperty("negative_precision").GetDouble(), 9);
        Assert.Equal(threshold.GetProperty("recall").GetDouble(), after.GetProperty("negative_recall").GetDouble(), 9);
    }

    [Fact]
    public async Task APromotedModelWithoutAThreshold_DropsTheOneProductionHad()
    {
        await TrainAsync();
        await JsonAsync("threshold", "--precision", "0.7", "--data", "datasets/recheck.csv");
        Assert.Equal(0, (await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10", "--no-promote")).ExitCode);

        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-002", "--json", "--force")).ExitCode);

        Assert.False(File.Exists("models/default/production/decision-threshold.json"));
        Assert.True(File.Exists("models/default/staging/exp-001/decision-threshold.json"));   // it stays with its model
    }

    [Theory]
    [InlineData("--precision", "0.9")]                     // no --data
    [InlineData("--precision", "1.5", "--data", "x.csv")]  // not a share
    [InlineData("--clear", "--precision", "0.9")]          // both
    [InlineData("--precision-of", "no")]                    // a class without a precision
    [InlineData("--precision", "0.9", "--precision-of", "maybe", "--data", "datasets/recheck.csv")]  // not a class
    [InlineData("--guarantee", "0.05")]                                                              // a guarantee of no precision
    [InlineData("--guarantee", "1.5", "--precision", "0.9", "--data", "datasets/recheck.csv")]       // not a risk
    [InlineData("--guarantee", "0", "--precision", "0.9", "--data", "datasets/recheck.csv")]         // not a risk
    [InlineData("--guarantee", "0.05", "--precision", "1", "--data", "datasets/recheck.csv")]        // no file proves a precision of 1
    public async Task AnAskThatCannotBeAnswered_IsRefused(params string[] args)
    {
        await TrainAsync();

        var (exit, stdout, _) = await CliRunner.RunAsync(["threshold", .. args, "--json"]);

        Assert.Equal(1, exit);
        Assert.False(string.IsNullOrWhiteSpace(JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString()));
    }

    /// <summary>
    /// <c>--guarantee</c> is an added mode: the usual choice carries no guarantee block, the certified one carries it, its
    /// lower bound is never below the target, and it never answers more rows than the usual choice would.
    /// </summary>
    [Fact]
    public async Task AGuaranteedThreshold_CarriesItsCertificate_AndIsNoLooserThanTheUsualOne()
    {
        await TrainAsync();

        var usual = (await JsonAsync("threshold", "--precision", "0.6", "--data", "datasets/recheck.csv")).GetProperty("threshold");
        Assert.Equal(JsonValueKind.Null, usual.GetProperty("guarantee").ValueKind);   // the default is unchanged

        var certified = (await JsonAsync("threshold", "--precision", "0.6", "--guarantee", "0.2", "--data", "datasets/recheck.csv")).GetProperty("threshold");

        var guarantee = certified.GetProperty("guarantee");
        Assert.Equal(0.2, guarantee.GetProperty("delta").GetDouble(), 9);
        Assert.True(guarantee.GetProperty("lowerBound").GetDouble() >= 0.6);
        Assert.True(certified.GetProperty("threshold").GetDouble() >= usual.GetProperty("threshold").GetDouble());
        Assert.True(certified.GetProperty("recall").GetDouble() <= usual.GetProperty("recall").GetDouble());
        Assert.Contains("guarantee", await File.ReadAllTextAsync("models/default/production/decision-threshold.json"));
    }

    /// <summary>When no threshold on the file can be certified the command says so and writes nothing — the model keeps what it had.</summary>
    [Fact]
    public async Task AGuaranteeTheFileCannotCarry_IsRefused_AndWritesNothing()
    {
        await TrainAsync();

        // A precision of 0.99 at risk 0.01 takes 459 rows all right; the file has 300.
        var (exit, stdout, _) = await CliRunner.RunAsync("threshold", "--precision", "0.99", "--guarantee", "0.01", "--data", "datasets/recheck.csv", "--json");

        Assert.Equal(1, exit);
        Assert.Contains("certified", JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString());
        Assert.False(File.Exists("models/default/production/decision-threshold.json"));
    }
}
