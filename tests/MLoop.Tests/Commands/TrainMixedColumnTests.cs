namespace MLoop.Tests.Commands;

/// <summary>
/// A column of numbers with one word in it trains. The training file and the test file of a pre-split run were each laid out from
/// their own rows: the file without the word read the column as numbers (folded into the feature vector), the file with it as text,
/// and the model fitted on one failed on the other with "Could not find input column".
/// </summary>
[Collection("FileSystem")]
public sealed class TrainMixedColumnTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-mixedcol-" + Guid.NewGuid().ToString("N"));

    public TrainMixedColumnTests() => Directory.CreateDirectory(_workspace);

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    [Fact]
    public async Task ANumericColumnWithOneWordInIt_Trains()
    {
        Directory.SetCurrentDirectory(_workspace);
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", "p", "--task", "binary-classification");
        Assert.True(initExit == 0, initOutput);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var rows = Enumerable.Range(1, 40).Select(i => $"{(i == 5 ? "abc" : i.ToString())},{(i * 7) % 13},{((i * 7) % 13 > 6 ? 1 : 0)}");
        await File.WriteAllLinesAsync("datasets/train.csv", ["size,x2,y", .. rows]);

        var (exit, stdout, stderr) = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "y", "--time", "60", "--max-trials", "3");

        Assert.True(exit == 0, stdout + stderr);
        Assert.DoesNotContain("Could not find input column", stdout + stderr, StringComparison.Ordinal);
    }
}
