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

    /// <summary>
    /// A weight column that is one of the model's own inputs: trained without <c>--weight</c> on a file that has a
    /// <c>w</c> column, <c>w</c> is a feature. Counting the evaluation rows by <c>w</c> must not take it out of the
    /// vector the model reads (<c>Feature vector dimension mismatch</c>), and a paired comparison of that model with one
    /// trained <i>with</i> <c>--weight w</c> (where <c>w</c> is not a feature) must score each on its own inputs.
    /// </summary>
    [Fact]
    public async Task AWeightColumnTheModelReadsAsAFeature_StillWeighsTheEvaluationRows()
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "regression")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));

        static string Table(int rows, int seed)
        {
            var rng = new Random(seed);
            var lines = new List<string> { "x1,x2,w,y" };
            for (var i = 0; i < rows; i++)
            {
                double x1 = rng.NextDouble() * 4 - 2, x2 = rng.NextDouble() * 4 - 2;
                var w = i % 3 == 0 ? 3 : 1;
                var y = 2 * x1 + x2 + (rng.NextDouble() - 0.5) * (w == 3 ? 3 : 0.4);
                lines.Add(string.Create(CultureInfo.InvariantCulture, $"{x1:F4},{x2:F4},{w},{y:F4}"));
            }
            return string.Join('\n', lines) + "\n";
        }

        await File.WriteAllTextAsync("datasets/train.csv", Table(240, seed: 1));
        await File.WriteAllTextAsync("datasets/test.csv", Table(90, seed: 2));
        // exp-001: w is a feature. exp-002: w weights the training rows and is not an input.
        var first = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "30", "--max-trials", "4");
        Assert.True(first.ExitCode == 0, first.Stdout + first.Stderr);
        var second = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--weight", "w", "--time", "30", "--max-trials", "4", "--no-promote");
        Assert.True(second.ExitCode == 0, second.Stdout + second.Stderr);

        var weighted = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/test.csv", "--weight", "w", "--json");
        Assert.True(weighted.ExitCode == 0, weighted.Stdout + weighted.Stderr);
        double reported;
        using (var doc = System.Text.Json.JsonDocument.Parse(weighted.Stdout))
            reported = doc.RootElement.GetProperty("testMetrics").GetProperty("mse").GetDouble();

        // The same number worked out from the model's own answers, row by row.
        var predicted = await CliRunner.RunAsync("predict", "datasets/test.csv", "--experiment", "exp-001", "--json");
        Assert.True(predicted.ExitCode == 0, predicted.Stdout + predicted.Stderr);
        double sumWeightedSquares = 0, sumWeights = 0;
        var rows = File.ReadAllLines("datasets/test.csv").Skip(1).Select(l => l.Split(',')).ToArray();
        using (var doc = System.Text.Json.JsonDocument.Parse(predicted.Stdout))
        {
            var answers = doc.RootElement.GetProperty("predictions").EnumerateArray().ToArray();
            Assert.Equal(rows.Length, answers.Length);
            for (var i = 0; i < rows.Length; i++)
            {
                var w = double.Parse(rows[i][2], CultureInfo.InvariantCulture);
                var error = double.Parse(rows[i][3], CultureInfo.InvariantCulture) - answers[i].GetProperty("score").GetDouble();
                sumWeightedSquares += w * error * error;
                sumWeights += w;
            }
        }
        Assert.Equal(sumWeightedSquares / sumWeights, reported, 4);

        var against = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/test.csv", "--against", "exp-002", "--json");
        Assert.True(against.ExitCode == 0, against.Stdout + against.Stderr);
        using var withBaseline = System.Text.Json.JsonDocument.Parse(against.Stdout);
        Assert.Equal("exp-002", withBaseline.RootElement.GetProperty("comparison").GetProperty("against").GetString());
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
