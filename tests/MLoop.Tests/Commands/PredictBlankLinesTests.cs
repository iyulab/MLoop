using System.Globalization;
using System.Text;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop predict</c> on a file that has a blank line. A prediction file has no label column, so predict writes a copy with an
/// empty one in front of every line - a blank line became <c>,</c>, a line of the wrong width among a few dozen, and a file of
/// fewer than a hundred rows failed with "unable to split the file into multiple columns" (80 rows and a trailing blank line).
/// </summary>
[Collection("FileSystem")]
public sealed class PredictBlankLinesTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-predict-blank-" + Guid.NewGuid().ToString("N"));

    public PredictBlankLinesTests()
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
        var csv = new StringBuilder(labelled ? "x,z,y\n" : "x,z\n");
        var rng = new Random(7);
        for (var i = 0; i < rows; i++)
        {
            var x = rng.NextDouble();
            csv.Append(CultureInfo.InvariantCulture, $"{x:F4},{rng.NextDouble():F4}");
            if (labelled) csv.Append(rng.NextDouble() < (x > 0.5 ? 0.9 : 0.1) ? ",yes" : ",no");
            csv.Append('\n');
        }
        return csv.ToString();
    }

    [Theory]
    [InlineData(80, "", "\n")]
    [InlineData(80, "", "\n\n\n")]
    [InlineData(10, "\n", "")]
    public async Task ABlankLine_IsNoRow_AndTheFileIsPredicted(int rows, string blankAfterHeader, string blankAtEnd)
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Rows(200, labelled: true));
        var (trained, trainOut, trainErr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(trained == 0, trainOut + trainErr);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001")).ExitCode);

        var body = Rows(rows, labelled: false);
        var header = body[..(body.IndexOf('\n') + 1)];
        await File.WriteAllTextAsync("datasets/predict.csv", header + blankAfterHeader + body[header.Length..] + blankAtEnd);

        var (exit, stdout, stderr) = await CliRunner.RunAsync("predict", "datasets/predict.csv", "--output", "out.csv");

        Assert.True(exit == 0, stdout + stderr);
        Assert.Equal(rows + 1, File.ReadAllLines("out.csv").Count(l => l.Length > 0));
    }

    [Fact]
    public async Task IncludedFeatures_StayWithTheirOwnPrediction_WhenABlankLineSitsBetweenRows()
    {
        // The features are written beside the predictions by line number, and predict skips the blank line: every row after it
        // came out with the features of the row below.
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "binary-classification")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Rows(200, labelled: true));
        var (trained, trainOut, trainErr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "10");
        Assert.True(trained == 0, trainOut + trainErr);
        Assert.Equal(0, (await CliRunner.RunAsync("promote", "exp-001")).ExitCode);

        var rows = Rows(12, labelled: false).TrimEnd('\n').Split('\n');
        var lines = rows.Take(6).Append("").Concat(rows.Skip(6)).ToArray();
        await File.WriteAllTextAsync("datasets/predict.csv", string.Join('\n', lines) + "\n");

        var (exit, stdout, stderr) = await CliRunner.RunAsync("predict", "datasets/predict.csv", "--include-features", "--output", "out.csv");

        Assert.True(exit == 0, stdout + stderr);
        var written = File.ReadAllLines("out.csv").Skip(1).Select(l => l.Split(',')[0]).ToArray();
        Assert.Equal(rows.Skip(1).Select(l => l.Split(',')[0]), written);
    }
}
