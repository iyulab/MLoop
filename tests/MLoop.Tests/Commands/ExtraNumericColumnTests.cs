using System.Globalization;
using System.Text;

namespace MLoop.Tests.Commands;

/// <summary>
/// A file given to <c>predict</c> or <c>evaluate</c> that has a numeric column the model was not trained on - a weight, an id. The
/// column loader folds the numeric columns of a file into one vector; a model trained on a single numeric feature reads a column named
/// after it, and found none once a second numeric column made the vector (<c>Could not find input column 'x'</c>). The answer must be
/// the one the file without that column gives.
/// </summary>
[Collection("FileSystem")]
public sealed class ExtraNumericColumnTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-extra-numeric-" + Guid.NewGuid().ToString("N"));

    public ExtraNumericColumnTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static (string[] Plain, string[] WithWeight) Tables(int rows, bool labelled, int seed)
    {
        var rng = new Random(seed);
        string[] kinds = ["a", "b", "c"];
        var plain = new List<string> { labelled ? "x,kind,y" : "x,kind" };
        var weighted = new List<string> { labelled ? "x,kind,y,w" : "x,kind,w" };
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble();
            var kind = kinds[rng.Next(kinds.Length)];
            var head = string.Create(CultureInfo.InvariantCulture, $"{x:F4},{kind}");
            var label = labelled ? (rng.NextDouble() < (x > 0.5 ? 0.9 : 0.1) ? ",yes" : ",no") : "";
            plain.Add(head + label);
            weighted.Add(head + label + (kind == "c" ? ",3" : ",1"));
        }
        return ([.. plain], [.. weighted]);
    }

    [Fact]
    public async Task ANumericColumnTheModelDoesNotRead_ChangesNeitherThePredictionsNorTheEvaluation()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", string.Join('\n', Tables(200, labelled: true, seed: 1).Plain) + "\n");
        var (trained, trainOut, trainErr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(trained == 0, trainOut + trainErr);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001")).ExitCode);

        var test = Tables(80, labelled: true, seed: 2);
        await File.WriteAllTextAsync("datasets/test-plain.csv", string.Join('\n', test.Plain) + "\n");
        await File.WriteAllTextAsync("datasets/test-weight.csv", string.Join('\n', test.WithWeight) + "\n");
        var fresh = Tables(30, labelled: false, seed: 3);
        await File.WriteAllTextAsync("datasets/fresh-plain.csv", string.Join('\n', fresh.Plain) + "\n");
        await File.WriteAllTextAsync("datasets/fresh-weight.csv", string.Join('\n', fresh.WithWeight) + "\n");

        var plainEval = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/test-plain.csv", "--json");
        var weightEval = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/test-weight.csv", "--json");
        Assert.True(weightEval.ExitCode == 0, weightEval.Stdout + weightEval.Stderr);
        using (var plain = System.Text.Json.JsonDocument.Parse(plainEval.Stdout))
        using (var weight = System.Text.Json.JsonDocument.Parse(weightEval.Stdout))
        {
            Assert.Equal(plain.RootElement.GetProperty("testMetrics").ToString(), weight.RootElement.GetProperty("testMetrics").ToString());
            Assert.Equal(plain.RootElement.GetProperty("confusion").ToString(), weight.RootElement.GetProperty("confusion").ToString());
        }

        Assert.Equal(0, (await CliRunner.RunAsync("predict", "datasets/fresh-plain.csv", "--output", "plain-out.csv")).ExitCode);
        var predicted = await CliRunner.RunAsync("predict", "datasets/fresh-weight.csv", "--output", "weight-out.csv");
        Assert.True(predicted.ExitCode == 0, predicted.Stdout + predicted.Stderr);
        // the predictions are the same rows' predictions; the weight column rides along in the copy of the input
        static string[] Answers(string path, int firstAnswerColumn) =>
            [.. File.ReadAllLines(path).Select(l => string.Join(',', l.Split(',').Skip(firstAnswerColumn)))];
        Assert.Equal(Answers("plain-out.csv", 2), Answers("weight-out.csv", 3));
    }

    /// <summary>The model that reads a column by its own name is the one trained on a lone numeric feature - nothing else in the file.</summary>
    [Theory]
    [InlineData("multiclass-classification")]
    [InlineData("regression")]
    public async Task ALoneNumericFeature_IsNotChangedByAWeightColumnTheModelDoesNotRead(string task)
    {
        var regression = task == "regression";
        static (string[] Plain, string[] WithWeight) Series(int rows, bool labelled, bool regression, int seed)
        {
            var rng = new Random(seed);
            var plain = new List<string> { labelled ? "x,y" : "x" };
            var weighted = new List<string> { labelled ? "x,w,y" : "x,w" };
            for (var i = 0; i < rows; i++)
            {
                var x = rng.NextDouble() * 2 - 1;
                var head = string.Create(CultureInfo.InvariantCulture, $"{x:F4}");
                var shifted = x + (rng.NextDouble() - 0.5) * 0.3;
                var label = !labelled ? ""
                    : regression ? string.Create(CultureInfo.InvariantCulture, $",{3 * x + (rng.NextDouble() - 0.5):F4}")
                    : shifted < -0.3 ? ",lo" : shifted > 0.3 ? ",hi" : ",mid";
                plain.Add(head + label);
                weighted.Add(string.Create(CultureInfo.InvariantCulture, $"{head},{1 + rng.Next(4)}") + label);
            }
            return ([.. plain], [.. weighted]);
        }

        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", task)).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", string.Join('\n', Series(240, labelled: true, regression, seed: 1).Plain) + "\n");
        var (trained, trainOut, trainErr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(trained == 0, trainOut + trainErr);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001")).ExitCode);

        var test = Series(90, labelled: true, regression, seed: 2);
        await File.WriteAllTextAsync("datasets/test-plain.csv", string.Join('\n', test.Plain) + "\n");
        await File.WriteAllTextAsync("datasets/test-weight.csv", string.Join('\n', test.WithWeight) + "\n");
        var fresh = Series(30, labelled: false, regression, seed: 3);
        await File.WriteAllTextAsync("datasets/fresh-plain.csv", string.Join('\n', fresh.Plain) + "\n");
        await File.WriteAllTextAsync("datasets/fresh-weight.csv", string.Join('\n', fresh.WithWeight) + "\n");

        var plainEval = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/test-plain.csv", "--json");
        var weightEval = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/test-weight.csv", "--json");
        Assert.True(weightEval.ExitCode == 0, weightEval.Stdout + weightEval.Stderr);
        using (var plain = System.Text.Json.JsonDocument.Parse(plainEval.Stdout))
        using (var weight = System.Text.Json.JsonDocument.Parse(weightEval.Stdout))
        {
            Assert.Equal(plain.RootElement.GetProperty("testMetrics").ToString(), weight.RootElement.GetProperty("testMetrics").ToString());
            Assert.Equal(plain.RootElement.GetProperty("confusion").ToString(), weight.RootElement.GetProperty("confusion").ToString());
        }

        Assert.Equal(0, (await CliRunner.RunAsync("predict", "datasets/fresh-plain.csv", "--output", "plain-out.csv")).ExitCode);
        var predicted = await CliRunner.RunAsync("predict", "datasets/fresh-weight.csv", "--output", "weight-out.csv");
        Assert.True(predicted.ExitCode == 0, predicted.Stdout + predicted.Stderr);
        static string[] Answers(string path, int firstAnswerColumn) =>
            [.. File.ReadAllLines(path).Select(l => string.Join(',', l.Split(',').Skip(firstAnswerColumn)))];
        Assert.Equal(Answers("plain-out.csv", 1), Answers("weight-out.csv", 2));
    }
}
