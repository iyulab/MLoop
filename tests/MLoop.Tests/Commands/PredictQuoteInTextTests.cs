namespace MLoop.Tests.Commands;

/// <summary>
/// <c>mloop predict</c> on a file whose text holds a quote (<c>12" pipe</c>) answers every row. The splitter its column steps
/// read each line with opened a quoted field at any quote: the row's other columns went into the first, and rows after it were dropped
/// (40 in, 36 out).
/// </summary>
[Collection("FileSystem")]
public sealed class PredictQuoteInTextTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-predquote-" + Guid.NewGuid().ToString("N"));

    public PredictQuoteInTextTests() => Directory.CreateDirectory(_workspace);

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private static string Rows(int count, Func<int, string> note)
        => string.Join("\n", ["note,x2,y", .. Enumerable.Range(1, count).Select(i => $"{note(i)},{(i * 7) % 13},{((i * 7) % 13 > 6 ? 1 : 0)}")]) + "\n";

    [Fact]
    public async Task AFileWithAQuoteInItsText_IsPredictedRowForRow()
    {
        Directory.SetCurrentDirectory(_workspace);
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        await File.WriteAllTextAsync("datasets/train.csv", Rows(120, i => $"n{i % 7}"));
        var (trained, trainOut, trainErr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "3");
        Assert.True(trained == 0, trainOut + trainErr);
        await File.WriteAllTextAsync("datasets/predict.csv", Rows(40, i => i % 9 == 0 ? "12\" pipe" : $"n{i % 7}"));

        var (exit, stdout, stderr) = await CliRunner.RunAsync("predict", "datasets/predict.csv", "--output", "out.csv");

        Assert.True(exit == 0, stdout + stderr);
        var lines = File.ReadAllLines("out.csv").Where(l => l.Length > 0).ToArray();
        Assert.Equal(41, lines.Length);   // the header and one line per input row
        Assert.Contains(lines, l => l.StartsWith("\"12\"\" pipe\",", StringComparison.Ordinal)); // the text and its own columns
    }
}
