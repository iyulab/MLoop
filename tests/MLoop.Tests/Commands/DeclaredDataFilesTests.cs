using System.Globalization;
using System.Text;

namespace MLoop.Tests.Commands;

/// <summary>
/// The files <c>mloop evaluate</c> and <c>mloop predict</c> read when none is named: the ones <c>data.test</c> / <c>data.predict</c>
/// of <c>mloop.yaml</c> name, else <c>datasets/test.csv</c> / <c>predict.csv</c> - whether or not the project still has
/// <c>datasets/train.csv</c>. <c>mloop init</c> writes <c>data.test</c> as a "default data path", and the commands read the
/// conventional file under the line <c>Auto-detected</c> whatever the yaml said; a project kept for scoring and predicting,
/// its training data elsewhere, was told the conventional file was missing while it sat there.
/// </summary>
[Collection("FileSystem")]
public sealed class DeclaredDataFilesTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-declared-data-" + Guid.NewGuid().ToString("N"));

    public DeclaredDataFilesTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static string Rows(int rows, bool labelled, int seed = 7)
    {
        var csv = new StringBuilder(labelled ? "x,z,y\n" : "x,z\n");
        var rng = new Random(seed);
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4}");
            if (labelled) csv.Append(rng.NextDouble() < (x > 0.5 ? 0.9 : 0.1) ? ",yes" : ",no");
            csv.Append('\n');
        }
        return csv.ToString();
    }

    /// <summary>A project with a promoted model; inside it, with no test, predict or train file but the training one.</summary>
    private async Task ProjectWithAModelAsync()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Rows(200, labelled: true));
        var (trained, trainOut, trainErr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(trained == 0, trainOut + trainErr);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001")).ExitCode);
    }

    /// <summary>Sets <c>data.&lt;key&gt;</c> in the project's mloop.yaml - in place when <c>mloop init</c> wrote the key, else as a new line.</summary>
    private static void Declare(string key, string path)
    {
        var lines = File.ReadAllLines("mloop.yaml").ToList();
        var dataAt = lines.FindIndex(l => l == "data:");
        Assert.True(dataAt >= 0, "the project file has no data: block to declare a path in");
        var keyAt = lines.FindIndex(dataAt + 1, l => l.StartsWith($"  {key}:", StringComparison.Ordinal));
        if (keyAt >= 0)
            lines[keyAt] = $"  {key}: {path}";
        else
            lines.Insert(dataAt + 1, $"  {key}: {path}");
        File.WriteAllLines("mloop.yaml", lines);
    }

    [Fact]
    public async Task TheFilesMLoopYamlNames_AreWhatEvaluateAndPredictRead_WhenNoneIsGiven()
    {
        await ProjectWithAModelAsync();
        await File.WriteAllTextAsync("hold.csv", Rows(60, labelled: true, seed: 11));
        await File.WriteAllTextAsync("pp.csv", Rows(5, labelled: false, seed: 13));
        // The conventional files exist too, and are not what was declared: a wrong pick would still run, and say so below.
        await File.WriteAllTextAsync("datasets/test.csv", Rows(60, labelled: true, seed: 17));
        await File.WriteAllTextAsync("datasets/predict.csv", Rows(3, labelled: false, seed: 19));
        Declare("test", "hold.csv");
        Declare("predict", "pp.csv");

        var (evalExit, evalOut, evalErr) = await CliRunner.RunAsync("evaluate");
        Assert.True(evalExit == 0, evalOut + evalErr);
        Assert.Contains("hold.csv", evalOut, StringComparison.Ordinal);
        Assert.DoesNotContain(Path.Combine("datasets", "test.csv"), evalOut, StringComparison.Ordinal);

        var (predictExit, predictOut, predictErr) = await CliRunner.RunAsync("predict", "--output", "out.csv");
        Assert.True(predictExit == 0, predictOut + predictErr);
        Assert.Contains("pp.csv", predictOut, StringComparison.Ordinal);
        Assert.Equal(5 + 1, File.ReadAllLines("out.csv").Count(l => l.Length > 0));
    }

    [Fact]
    public async Task EvaluateAndPredict_FindTheirConventionalFile_WithoutATrainCsv()
    {
        await ProjectWithAModelAsync();
        await File.WriteAllTextAsync("datasets/test.csv", Rows(60, labelled: true, seed: 11));
        await File.WriteAllTextAsync("datasets/predict.csv", Rows(4, labelled: false, seed: 13));
        File.Delete("datasets/train.csv");

        var (evalExit, evalOut, evalErr) = await CliRunner.RunAsync("evaluate");
        Assert.True(evalExit == 0, evalOut + evalErr);

        var (predictExit, predictOut, predictErr) = await CliRunner.RunAsync("predict", "--output", "out.csv");
        Assert.True(predictExit == 0, predictOut + predictErr);
        Assert.Equal(4 + 1, File.ReadAllLines("out.csv").Count(l => l.Length > 0));
    }

    [Fact]
    public async Task ADeclaredFileThatDoesNotExist_IsNamedByItsKey_WhenNothingStandsInForIt()
    {
        await ProjectWithAModelAsync();
        Declare("test", "nope.csv");
        Declare("predict", "nowhere.csv");

        var (evalExit, evalOut, evalErr) = await CliRunner.RunAsync("evaluate");
        Assert.NotEqual(0, evalExit);
        Assert.Contains("data.test", evalOut + evalErr, StringComparison.Ordinal);
        Assert.Contains("nope.csv", evalOut + evalErr, StringComparison.Ordinal);

        var (predictExit, predictOut, predictErr) = await CliRunner.RunAsync("predict");
        Assert.NotEqual(0, predictExit);
        Assert.Contains("data.predict", predictOut + predictErr, StringComparison.Ordinal);
        Assert.Contains("nowhere.csv", predictOut + predictErr, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Train_NamesTheDeclaredTrainFileThatIsMissing_NotOnlyTheConventionalOne()
    {
        // evaluate and predict said what mloop.yaml declared; train alone named datasets/train.csv — a file the user never asked for.
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        Declare("train", "fitt.csv");

        var (exit, stdout, stderr) = await CliRunner.RunAsync("train");

        Assert.NotEqual(0, exit);
        var said = stdout + stderr;
        Assert.Contains("data.train", said, StringComparison.Ordinal);
        Assert.Contains("fitt.csv", said, StringComparison.Ordinal);
        Assert.Contains("datasets/train.csv", said, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Analyze_NamesTheDeclaredTrainFileThatIsMissing_AsTrainDoes()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        Declare("train", "fitt.csv");

        var (exit, stdout, stderr) = await CliRunner.RunAsync("analyze", "profile");

        Assert.NotEqual(0, exit);
        var said = stdout + stderr;
        Assert.Contains("data.train", said, StringComparison.Ordinal);
        Assert.Contains("fitt.csv", said, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Train_WithNothingDeclared_StillSaysTheConventionalSentence()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var lines = File.ReadAllLines("mloop.yaml").Where(l => !l.StartsWith("  train:", StringComparison.Ordinal)).ToArray();
        File.WriteAllLines("mloop.yaml", lines);

        var (exit, stdout, stderr) = await CliRunner.RunAsync("train");

        Assert.NotEqual(0, exit);
        Assert.Contains("No data file specified and datasets/train.csv not found.", stdout + stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("data.train", stdout + stderr, StringComparison.Ordinal);
    }
}
