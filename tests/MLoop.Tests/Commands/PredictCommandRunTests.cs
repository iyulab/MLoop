using System.Text.Json;
using System.Text.RegularExpressions;
using MLoop.CLI.Commands;
using static MLoop.Core.Prediction.CategoricalMapper;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop predict</c> as the command answers it: the places it refuses (outside a project, no model, a model file that is not
/// there, an option that does not fit the model), what <c>--unknown-strategy</c> does to a category training never saw, and what the
/// output holds with <c>--include-features</c>, <c>--log</c> and <c>--json</c>. The command's own body ran in few tests; the pieces
/// behind it did, which says nothing about whether the command handed them the right thing.
/// An unknown <c>--unknown-strategy</c> used to run as <c>auto</c>: a typo by someone who wanted the run to stop on a new
/// category predicted past it, reading the value as missing. It is refused now, before the model is loaded.
/// </summary>
[Collection("FileSystem")]
public sealed class PredictCommandRunTests : IClassFixture<PredictCommandRunTests.TrainedProject>, IDisposable
{
    /// <summary>One small model, trained once for the class: three colours, a number, and a label that depends on both.</summary>
    public sealed class TrainedProject : IAsyncLifetime
    {
        private readonly string _base = Path.Combine(Path.GetTempPath(), "mloop-predict-" + Guid.NewGuid().ToString("N"));
        public string Root => Path.Combine(_base, "p");

        public async ValueTask InitializeAsync()
        {
            var original = Directory.GetCurrentDirectory();
            try
            {
                Directory.CreateDirectory(_base);
                Directory.SetCurrentDirectory(_base);
                var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
                Assert.True(initExit == 0, initOutput);
                Directory.SetCurrentDirectory(Root);
                var random = new Random(4);
                var rows = new List<string> { "x,color,y" };
                for (var i = 0; i < 300; i++)
                {
                    var color = new[] { "red", "green", "blue" }[random.Next(3)];
                    var x = random.NextDouble();
                    rows.Add(FormattableString.Invariant($"{x:0.000},{color},{((x > 0.5) != (color == "red") ? "yes" : "no")}"));
                }
                await File.WriteAllLinesAsync("datasets/train.csv", rows);
                var (exit, output, error) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "3");
                Assert.True(exit == 0, output + error);
            }
            finally
            {
                Directory.SetCurrentDirectory(original);
            }
        }

        public ValueTask DisposeAsync()
        {
            try { Directory.Delete(_base, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
            return ValueTask.CompletedTask;
        }
    }

    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _root;

    public PredictCommandRunTests(TrainedProject project)
    {
        _root = project.Root;
        Directory.SetCurrentDirectory(_root);
        // Two rows training knew, one colour it never saw.
        File.WriteAllLines("datasets/predict.csv", ["x,color", "0.9,red", "0.1,green", "0.7,purple"]);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
    }

    // The console colours a count and its noun apart; a test reads the words.
    private static string Plain(string text) => Regex.Replace(text, "\\x1B\\[[0-9;]*m", "");

    private static async Task<(int ExitCode, string Stdout, string Stderr)> PredictAsync(params string[] args)
    {
        var (exit, stdout, stderr) = await CliRunner.RunAsync(["predict", .. args]);
        return (exit, Plain(stdout), Plain(stderr));
    }

    private static string[] DataLines(string path) => [.. File.ReadAllLines(path).Skip(1).Where(l => l.Length > 0)];

    [Theory]
    [InlineData("auto")]
    [InlineData("use-most-frequent")]
    [InlineData("use-missing")]
    [InlineData("USE-MISSING")]   // any case, as it was
    public async Task AColourTrainingNeverSaw_IsPredictedPast_ByEveryStrategyThatDoesNotStop(string strategy)
    {
        var (exit, stdout, stderr) = await PredictAsync("datasets/predict.csv", "--unknown-strategy", strategy, "--output", $"predictions/{strategy}.csv");

        Assert.True(exit == 0, stdout + stderr);
        Assert.Equal(3, DataLines($"predictions/{strategy}.csv").Length);
    }

    [Theory]
    [InlineData("error")]
    [InlineData("ERROR")]
    public async Task TheErrorStrategy_StopsOnTheNewColour_NamingTheColumnAndTheValue(string strategy)
    {
        var (exit, stdout, stderr) = await PredictAsync("datasets/predict.csv", "--unknown-strategy", strategy, "--output", "predictions/stopped.csv");

        Assert.Equal(1, exit);
        Assert.Contains("Column 'color'", stdout + stderr);
        Assert.Contains("Unknown values: purple", stdout + stderr);
        Assert.False(File.Exists("predictions/stopped.csv"));
    }

    [Theory]
    [InlineData("errror")]
    [InlineData("most-frequent")]
    [InlineData("auto,error")]
    public async Task AStrategyNameNoneOfTheFourIs_IsRefusedBeforeTheModelIsLoaded(string strategy)
    {
        var (exit, stdout, stderr) = await PredictAsync("datasets/predict.csv", "--unknown-strategy", strategy);

        Assert.Equal(1, exit);
        Assert.Contains($"Unknown strategy '{strategy}' for --unknown-strategy", stdout + stderr);
        Assert.Contains("auto, error, use-most-frequent, use-missing", stdout + stderr);
        Assert.DoesNotContain("Using production model", stdout + stderr);
    }

    [Fact]
    public void EveryNameTheRefusalOffers_IsOneTheOptionMeans_AndTheFourAreAllOfThem()
    {
        Assert.Equal(["auto", "error", "use-most-frequent", "use-missing"], UnknownValueStrategies.Names);
        Assert.Equal(UnknownValueStrategy.Auto, UnknownValueStrategies.Parse("auto"));
        Assert.Equal(UnknownValueStrategy.Error, UnknownValueStrategies.Parse(" Error "));
        Assert.Equal(UnknownValueStrategy.UseMostFrequent, UnknownValueStrategies.Parse("use-most-frequent"));
        Assert.Equal(UnknownValueStrategy.UseMissing, UnknownValueStrategies.Parse("USE-MISSING"));
        Assert.All(UnknownValueStrategies.Names, name => Assert.NotNull(UnknownValueStrategies.Parse(name)));
        Assert.Null(UnknownValueStrategies.Parse("errror"));
        Assert.Null(UnknownValueStrategies.Refusal("auto"));
        Assert.Null(UnknownValueStrategies.Refusal(""));
        Assert.NotNull(UnknownValueStrategies.Refusal("errror"));
    }

    [Fact]
    public async Task ByDefault_TheInputColumnsComeBackBesideTheAnswers_AndNotWhenTurnedOff()
    {
        var withFeatures = await PredictAsync("datasets/predict.csv", "--unknown-strategy", "auto", "--output", "predictions/with.csv");
        var without = await PredictAsync("datasets/predict.csv", "--unknown-strategy", "auto", "--include-features", "false", "--output", "predictions/without.csv");

        Assert.Equal(0, withFeatures.ExitCode);
        Assert.Equal(0, without.ExitCode);
        var withHeader = File.ReadLines("predictions/with.csv").First().TrimStart('﻿');
        var withoutHeader = File.ReadLines("predictions/without.csv").First().TrimStart('﻿');
        Assert.StartsWith("x,color,PredictedLabel", withHeader);
        Assert.StartsWith("PredictedLabel", withoutHeader);
        Assert.DoesNotContain("color", withoutHeader);
        Assert.Equal(3, DataLines("predictions/with.csv").Length);
        Assert.Equal(3, DataLines("predictions/without.csv").Length);
    }

    [Fact]
    public async Task WithLog_EachPredictionIsKeptUnderTheProjectsLogs()
    {
        var (exit, stdout, stderr) = await PredictAsync("datasets/predict.csv", "--log", "--output", "predictions/logged.csv");

        Assert.True(exit == 0, stdout + stderr);
        var logs = Directory.GetFiles(Path.Combine(".mloop", "logs", "default"), "*.jsonl");
        Assert.NotEmpty(logs);
        Assert.True(logs.Sum(file => File.ReadAllLines(file).Count(l => l.Length > 0)) >= 3);
    }

    [Fact]
    public async Task Json_AnswersWithOneDocument_AndRefusesAFileBesideIt()
    {
        var (exit, stdout, _) = await PredictAsync("datasets/predict.csv", "--json");

        Assert.Equal(0, exit);
        using (var document = JsonDocument.Parse(stdout))
        {
            Assert.Equal("default", document.RootElement.GetProperty("model").GetString());
            Assert.Equal(3, document.RootElement.GetProperty("count").GetInt32());
            Assert.Equal(3, document.RootElement.GetProperty("predictions").GetArrayLength());
        }

        var (refusedExit, refusedOut, _) = await PredictAsync("datasets/predict.csv", "--json", "--output", "predictions/never.csv");

        Assert.Equal(1, refusedExit);
        using var error = JsonDocument.Parse(refusedOut);
        Assert.Contains("--output and --json cannot be used together", error.RootElement.GetProperty("error").GetString());
        Assert.False(File.Exists("predictions/never.csv"));
    }

    [Fact]
    public async Task AScoreThreshold_IsRefusedForAModelThatIsNotADetector()
    {
        var (exit, stdout, stderr) = await PredictAsync("datasets/predict.csv", "--score-threshold", "0.5");

        Assert.Equal(1, exit);
        Assert.Contains("--score-threshold applies to object detection; this model's task is binary-classification.", stdout + stderr);
    }

    [Fact]
    public async Task ADataFileThatIsNotThere_IsNamedWithWhereARelativePathResolves()
    {
        var (exit, stdout, stderr) = await PredictAsync("nofile.csv");

        Assert.Equal(1, exit);
        Assert.Contains("Data file not found", stdout + stderr);
        Assert.Contains("resolves against the project root", stdout + stderr);
    }

    [Fact]
    public async Task AModelFileThatIsNotThere_IsSaidToBeMissing()
    {
        var (exit, stdout, stderr) = await PredictAsync("datasets/predict.csv", "--model", "nope.zip");

        Assert.Equal(1, exit);
        Assert.Contains("Model file not found", stdout + stderr);
    }

    [Fact]
    public async Task OutsideAProject_TheCommandSaysSo_AndPointsToInit()
    {
        var elsewhere = Path.Combine(Path.GetTempPath(), "mloop-predict-outside-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(elsewhere);
        try
        {
            Directory.SetCurrentDirectory(elsewhere);

            var (exit, stdout, stderr) = await PredictAsync("data.csv");

            Assert.Equal(1, exit);
            Assert.Contains("Not inside a MLoop project", stdout + stderr);
            Assert.Contains("mloop init", stdout + stderr);
        }
        finally
        {
            Directory.SetCurrentDirectory(_root);
            try { Directory.Delete(elsewhere, recursive: true); } catch { /* not a test failure */ }
        }
    }

    [Fact]
    public async Task AProjectWithNoModelYet_SaysToTrainOne()
    {
        var empty = Path.Combine(Path.GetTempPath(), "mloop-predict-empty-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(empty);
        try
        {
            Directory.SetCurrentDirectory(empty);
            var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "q", "--task", "binary-classification");
            Assert.True(initExit == 0, initOutput);
            Directory.SetCurrentDirectory(Path.Combine(empty, "q"));
            File.WriteAllLines("datasets/predict.csv", ["x,color", "1,red"]);

            var (exit, stdout, stderr) = await PredictAsync("datasets/predict.csv");

            Assert.Equal(1, exit);
            Assert.Contains("No production model found for 'default'", stdout + stderr);
            Assert.Contains("mloop train", stdout + stderr);
        }
        finally
        {
            Directory.SetCurrentDirectory(_root);
            try { Directory.Delete(empty, recursive: true); } catch { /* not a test failure */ }
        }
    }
}
