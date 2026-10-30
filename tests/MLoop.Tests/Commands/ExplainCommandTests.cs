using System.Globalization;
using System.Text;
using System.Text.Json;
using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop explain</c> on real models: the column that carries the label falls the score most when shuffled, a column of noise falls
/// it by about nothing, the same seed gives the same numbers, and what cannot be explained is refused in words.
/// </summary>
[Collection("FileSystem")]
public sealed class ExplainCommandTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-explain-" + Guid.NewGuid().ToString("N"));

    public ExplainCommandTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    // x decides the answer (with some noise), z and w are noise.
    private static string Binary(int rows, int seed)
    {
        var csv = new StringBuilder("x,z,w,y\n");
        var rng = new Random(seed);
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble();
            var yes = rng.NextDouble() < (x > 0.5 ? 0.92 : 0.08);
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{rng.NextDouble():F4},{(yes ? "yes" : "no")}\n");
        }
        return csv.ToString();
    }

    private static string Regression(int rows, int seed)
    {
        var csv = new StringBuilder("x,z,w,y\n");
        var rng = new Random(seed);
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble() * 10;
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4},{rng.NextDouble():F4},{(3 * x + rng.NextDouble()):F4}\n");
        }
        return csv.ToString();
    }

    private async Task TrainAsync(string task, string trainData, string labelled)
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", task)).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", trainData);
        await File.WriteAllTextAsync("datasets/recheck.csv", labelled);
        var (exit, output, error) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(exit == 0, output + Environment.NewLine + error);
    }

    private static async Task<JsonElement> JsonAsync(params string[] args)
    {
        var (exit, stdout, stderr) = await CliRunner.RunAsync([.. args, "--json"]);
        Assert.True(exit == 0, stdout + stderr);
        return JsonDocument.Parse(stdout).RootElement.Clone();
    }

    [Fact]
    public async Task AClassifier_RelyesOnTheColumnThatDecidesTheAnswer_NotOnNoise()
    {
        await TrainAsync("binary-classification", Binary(400, 3), Binary(400, 9));

        var explanation = await JsonAsync("explain", "exp-001", "--data", "datasets/recheck.csv");

        Assert.True(explanation.GetProperty("higherIsBetter").GetBoolean());
        var features = explanation.GetProperty("features").EnumerateArray().ToList();
        Assert.Equal("x", features[0].GetProperty("column").GetString());
        var x = features[0].GetProperty("importance").GetDouble();
        var noise = features.Where(f => f.GetProperty("column").GetString() != "x").Select(f => f.GetProperty("importance").GetDouble()).ToList();
        Assert.True(x > 0.2, $"shuffling the deciding column should cost a lot of {explanation.GetProperty("metric").GetString()}, cost {x}");
        Assert.All(noise, n => Assert.True(Math.Abs(n) < x / 4, $"a noise column fell the score by {n} against {x}"));
        Assert.Equal(400, explanation.GetProperty("rows").GetInt32());
    }

    // x decides the answer; n01..n(columns-1) are noise.
    private static string Wide(int rows, int seed, int columns)
    {
        var header = "x," + string.Join(",", Enumerable.Range(1, columns - 1).Select(j => $"n{j:D2}")) + ",y\n";
        var csv = new StringBuilder(header);
        var rng = new Random(seed);
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble();
            var yes = rng.NextDouble() < (x > 0.5 ? 0.92 : 0.08);
            csv.Append(x.ToString("F4", CultureInfo.InvariantCulture));
            for (var j = 1; j < columns; j++)
                csv.Append(',').Append(rng.NextDouble().ToString("F4", CultureInfo.InvariantCulture));
            csv.Append(',').Append(yes ? "yes" : "no").Append('\n');
        }
        return csv.ToString();
    }

    [Fact]
    public async Task AWideTable_ShowsTheColumnsTheModelReliesOnMost_AndSaysHowManyItLeftOut()
    {
        const int columns = 30;
        await TrainAsync("binary-classification", Wide(400, 3, columns), Wide(400, 9, columns));

        var (exit, table, _) = await CliRunner.RunAsync("explain", "exp-001", "--data", "datasets/recheck.csv", "--repeats", "1");
        var all = await JsonAsync("explain", "exp-001", "--data", "datasets/recheck.csv", "--repeats", "1");

        Assert.Equal(0, exit);
        Assert.Equal(columns, all.GetProperty("features").GetArrayLength());   // --json carries every column
        var shown = Enumerable.Range(1, columns - 1).Count(j => table.Contains($"n{j:D2}", StringComparison.Ordinal));
        Assert.Equal(ExplainCommand.DisplayedColumns - 1, shown);               // x is the one more of the twenty
        Assert.Contains(" x ", table);
        Assert.Contains($"and {columns - ExplainCommand.DisplayedColumns} more columns", table);
        Assert.Contains("--json lists every column", table);
    }

    [Fact]
    public void ALongJsonRun_SaysHowFarItIsATenthAtATime_AShortOneStaysSilent()
    {
        var reported = 0;
        var notes = new List<string>();
        for (var done = 1; done <= 3000; done++)
            if (ExplainCommand.ProgressNote(done, 3000, ref reported) is { } note)
                notes.Add(note);

        Assert.Equal(10, notes.Count);
        Assert.Equal("explain: 10% (300 of 3000 scorings)", notes[0]);
        Assert.Equal("explain: 100% (3000 of 3000 scorings)", notes[^1]);

        var quiet = 0;
        for (var done = 1; done < ExplainCommand.NarratedFrom; done++)
            Assert.Null(ExplainCommand.ProgressNote(done, ExplainCommand.NarratedFrom - 1, ref quiet));
    }

    [Fact]
    public async Task ARegressor_IsExplainedInTheMetricEvaluateReports()
    {
        await TrainAsync("regression", Regression(400, 3), Regression(400, 9));

        var explanation = await JsonAsync("explain", "--data", "datasets/recheck.csv", "--repeats", "2");

        // The metric is the one `mloop evaluate` puts first for a regressor, read in its own direction (higher R² is better).
        Assert.Equal("r_squared", explanation.GetProperty("metric").GetString());
        Assert.True(explanation.GetProperty("higherIsBetter").GetBoolean());
        var features = explanation.GetProperty("features").EnumerateArray().ToList();
        Assert.Equal("x", features[0].GetProperty("column").GetString());
        Assert.True(features[0].GetProperty("importance").GetDouble() > 0.5);   // R² falls from near 1 when x is shuffled
        Assert.Equal(2, explanation.GetProperty("repeats").GetInt32());
    }

    [Fact]
    public async Task TheSameSeedGivesTheSameNumbers_AndASampleIsWhatWasAsked()
    {
        await TrainAsync("binary-classification", Binary(400, 3), Binary(400, 9));

        var first = await JsonAsync("explain", "exp-001", "--data", "datasets/recheck.csv", "--rows", "150", "--seed", "5");
        var second = await JsonAsync("explain", "exp-001", "--data", "datasets/recheck.csv", "--rows", "150", "--seed", "5");

        Assert.Equal(150, first.GetProperty("rows").GetInt32());
        Assert.Equal(first.GetProperty("features").GetRawText(), second.GetProperty("features").GetRawText());
    }

    [Theory]
    [InlineData("--repeats", "0")]
    [InlineData("--rows", "1")]
    public async Task AnAskThatCannotBeAnswered_IsRefusedInWords(string option, string value)
    {
        await TrainAsync("binary-classification", Binary(200, 3), Binary(200, 9));

        var (exit, stdout, _) = await CliRunner.RunAsync(["explain", "exp-001", "--data", "datasets/recheck.csv", option, value, "--json"]);

        Assert.Equal(1, exit);
        Assert.False(string.IsNullOrWhiteSpace(JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString()));
    }

    [Fact]
    public async Task AFileWithoutTheLabel_IsRefused_NotScoredAsIfItHadOne()
    {
        await TrainAsync("binary-classification", Binary(200, 3), Binary(200, 9));
        await File.WriteAllTextAsync("datasets/unlabeled.csv", "x,z,w\n0.1,0.2,0.3\n0.9,0.8,0.7\n0.5,0.5,0.5\n");

        var (exit, stdout, _) = await CliRunner.RunAsync(["explain", "exp-001", "--data", "datasets/unlabeled.csv", "--json"]);

        Assert.Equal(1, exit);
        Assert.Contains("label", JsonDocument.Parse(stdout).RootElement.GetProperty("error").GetString(), StringComparison.OrdinalIgnoreCase);
    }
}
